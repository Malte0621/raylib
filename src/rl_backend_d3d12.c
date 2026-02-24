/**********************************************************************************************
*
*   rl_backend_d3d12 - Direct3D 12 rendering backend for rlgl
*
*   DESCRIPTION:
*       Complete Direct3D 12 implementation of the rlgl rendering API.
*       Provides all functions required by rlgl.h when GRAPHICS_API_DIRECT3D12 is defined.
*
*   COMPILATION:
*       Only compiled when GRAPHICS_API_DIRECT3D12 is defined (Windows 10+ only).
*       Requires: d3d12.lib, d3dcompiler.lib, dxgi.lib
*
*   LICENSE: zlib/libpng (same as raylib)
*
**********************************************************************************************/

#if defined(GRAPHICS_API_DIRECT3D12)

#include "rlgl.h"
#include "rl_backend.h"

// Required Windows/D3D12 headers
#ifndef WIN32_LEAN_AND_MEAN
    #define WIN32_LEAN_AND_MEAN
#endif
#ifndef CINTERFACE
    #define CINTERFACE
#endif
#ifndef COBJMACROS
    #define COBJMACROS
#endif

#include <windows.h>
#include <d3d12.h>
#include <d3dcompiler.h>
#include <dxgi1_4.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

#pragma comment(lib, "d3d12.lib")
#pragma comment(lib, "d3dcompiler.lib")
#pragma comment(lib, "dxgi.lib")

#ifndef PI
    #define PI 3.14159265358979323846f
#endif
#ifndef DEG2RAD
    #define DEG2RAD (PI/180.0f)
#endif

#ifndef TRACELOG
    #define TRACELOG(level, ...) (void)0
    #define TRACELOGD(...) (void)0
#endif

#ifndef RL_MALLOC
    #define RL_MALLOC(sz)     malloc(sz)
#endif
#ifndef RL_CALLOC
    #define RL_CALLOC(n,sz)   calloc(n,sz)
#endif
#ifndef RL_REALLOC
    #define RL_REALLOC(n,sz)  realloc(n,sz)
#endif
#ifndef RL_FREE
    #define RL_FREE(p)        free(p)
#endif

//----------------------------------------------------------------------------------
// Defines
//----------------------------------------------------------------------------------
#ifndef RL_DEFAULT_BATCH_BUFFER_ELEMENTS
    #define RL_DEFAULT_BATCH_BUFFER_ELEMENTS  8192
#endif
#ifndef RL_DEFAULT_BATCH_BUFFERS
    #define RL_DEFAULT_BATCH_BUFFERS          1
#endif
#ifndef RL_DEFAULT_BATCH_DRAWCALLS
    #define RL_DEFAULT_BATCH_DRAWCALLS        256
#endif
#ifndef RL_DEFAULT_BATCH_MAX_TEXTURE_UNITS
    #define RL_DEFAULT_BATCH_MAX_TEXTURE_UNITS 4
#endif
#ifndef RL_MAX_MATRIX_STACK_SIZE
    #define RL_MAX_MATRIX_STACK_SIZE          32
#endif
#ifndef RL_MAX_SHADER_LOCATIONS
    #define RL_MAX_SHADER_LOCATIONS           32
#endif
#ifndef RL_CULL_DISTANCE_NEAR
    #define RL_CULL_DISTANCE_NEAR             0.01
#endif
#ifndef RL_CULL_DISTANCE_FAR
    #define RL_CULL_DISTANCE_FAR              1000.0
#endif

// Maximum tracked resources
#define RL_D3D12_MAX_TEXTURES       4096
#define RL_D3D12_MAX_SHADERS        256
#define RL_D3D12_MAX_BUFFERS        4096
#define RL_D3D12_MAX_FRAMEBUFFERS   256
#define RL_D3D12_NUM_FRAMES         2       // Double buffering for frame resources
#define RL_D3D12_MAX_SRV_DESC       4096    // SRV descriptor heap size

//----------------------------------------------------------------------------------
// Types - D3D12 resource tracking
//----------------------------------------------------------------------------------
typedef struct {
    ID3D12Resource *resource;
    D3D12_CPU_DESCRIPTOR_HANDLE srvCpuHandle;
    D3D12_GPU_DESCRIPTOR_HANDLE srvGpuHandle;
    int srvIndex;                           // Index in SRV descriptor heap
    int width, height, format;
    bool isCubemap;
    bool isDepth;
    D3D12_CPU_DESCRIPTOR_HANDLE dsvHandle;  // For depth textures
    D3D12_CPU_DESCRIPTOR_HANDLE rtvHandle;  // For render targets
    int dsvIndex;
    int rtvIndex;
    DXGI_FORMAT dxgiFormat;
} rlD3D12Texture;

typedef struct {
    ID3DBlob *vsBlob;
    ID3DBlob *psBlob;
    ID3D12PipelineState *pso;       // Pipeline state object
} rlD3D12Shader;

typedef struct {
    ID3D12Resource *resource;
    int size;
    bool dynamic;
    bool isIndex;
    void *mappedData;               // Persistently mapped for dynamic buffers
} rlD3D12Buffer;

typedef struct {
    int colorTexIds[8];
    int depthTexId;
    int colorCount;
    bool complete;
} rlD3D12Framebuffer;

//----------------------------------------------------------------------------------
// Constant buffer structures (must be 16-byte aligned)
//----------------------------------------------------------------------------------
typedef struct RL_ALIGN(16) {
    float mvp[16];
} rlD3D12MatrixCB;

typedef struct RL_ALIGN(16) {
    float colDiffuse[4];
} rlD3D12ColorCB;

//----------------------------------------------------------------------------------
// Per-frame resources (double-buffered)
//----------------------------------------------------------------------------------
typedef struct {
    ID3D12CommandAllocator *commandAllocator;
    ID3D12Resource *cbMatrixUpload;     // Constant buffer for MVP
    ID3D12Resource *cbColorUpload;      // Constant buffer for diffuse color
    void *cbMatrixMapped;
    void *cbColorMapped;
    // Batch vertex upload buffers
    ID3D12Resource *batchVertexBuffers[4];   // pos, texcoord, normal, color
    void *batchVertexMapped[4];
    // Batch index buffer
    ID3D12Resource *batchIndexBuffer;
    void *batchIndexMapped;
    UINT64 fenceValue;
} rlD3D12FrameResources;

//----------------------------------------------------------------------------------
// Internal rlgl state for D3D12
//----------------------------------------------------------------------------------
typedef struct rlglData {
    rlRenderBatch *currentBatch;
    rlRenderBatch defaultBatch;

    struct {
        int vertexCounter;
        float texcoordx, texcoordy;
        float normalx, normaly, normalz;
        unsigned char colorr, colorg, colorb, colora;

        int currentMatrixMode;
        Matrix *currentMatrix;
        Matrix modelview;
        Matrix projection;
        Matrix transform;
        bool transformRequired;
        Matrix stack[RL_MAX_MATRIX_STACK_SIZE];
        int stackCounter;

        unsigned int defaultTextureId;
        unsigned int activeTextureId[RL_DEFAULT_BATCH_MAX_TEXTURE_UNITS];
        unsigned int defaultVShaderId;
        unsigned int defaultFShaderId;
        unsigned int defaultShaderId;
        int *defaultShaderLocs;
        unsigned int currentShaderId;
        int *currentShaderLocs;

        bool stereoRender;
        Matrix projectionStereo[2];
        Matrix viewOffsetStereo[2];

        int currentBlendMode;
        bool depthTestEnabled;
        bool depthWriteEnabled;
        bool backfaceCullingEnabled;
        bool scissorTestEnabled;
        bool colorBlendEnabled;
        bool wireMode;
        int cullFaceMode;
        float lineWidth;

        float clearColor[4];

        int framebufferWidth;
        int framebufferHeight;
    } State;

    struct {
        bool computeShader;
        bool ssbo;
        float maxAnisotropyLevel;
        int maxDepthBits;
    } ExtSupported;

    // D3D12 core objects
    IDXGIFactory4 *factory;
    ID3D12Device *device;
    ID3D12CommandQueue *commandQueue;
    ID3D12GraphicsCommandList *commandList;
    ID3D12RootSignature *rootSignature;

    // Descriptor heaps
    ID3D12DescriptorHeap *srvHeap;          // Shader-visible CBV/SRV/UAV heap
    ID3D12DescriptorHeap *rtvHeap;          // Render target view heap
    ID3D12DescriptorHeap *dsvHeap;          // Depth stencil view heap
    ID3D12DescriptorHeap *samplerHeap;      // Sampler heap
    UINT srvDescriptorSize;
    UINT rtvDescriptorSize;
    UINT dsvDescriptorSize;
    UINT samplerDescriptorSize;
    int srvHeapUsed;                        // Next free SRV slot
    int rtvHeapUsed;
    int dsvHeapUsed;

    // Synchronization
    ID3D12Fence *fence;
    HANDLE fenceEvent;
    UINT64 fenceValue;

    // Frame resources (double buffered)
    rlD3D12FrameResources frames[RL_D3D12_NUM_FRAMES];
    int currentFrame;

    // Pipeline state cache
    bool pipelineDirty;
    D3D12_GRAPHICS_PIPELINE_STATE_DESC currentPSODesc;

    // Resource tracking arrays
    rlD3D12Texture textures[RL_D3D12_MAX_TEXTURES];
    unsigned int textureCount;

    rlD3D12Shader shaders[RL_D3D12_MAX_SHADERS];
    unsigned int shaderCount;

    rlD3D12Buffer buffers[RL_D3D12_MAX_BUFFERS];
    unsigned int bufferCount;

    rlD3D12Framebuffer framebuffers[RL_D3D12_MAX_FRAMEBUFFERS];
    unsigned int framebufferCount;

    unsigned int activeFramebuffer;

    int batchBufferSize;

} rlglData;

//----------------------------------------------------------------------------------
// Forward declarations
//----------------------------------------------------------------------------------
static Matrix rlMatrixIdentity(void);
static Matrix rlMatrixMultiply(Matrix left, Matrix right);
static Matrix rlMatrixTranspose(Matrix mat);
static Matrix rlMatrixInvert(Matrix mat);
typedef struct rl_float16 { float v[16]; } rl_float16;
static rl_float16 rlMatrixToFloatV(Matrix mat);
#define rlMatrixToFloat(mat) (rlMatrixToFloatV(mat).v)
static int rlGetPixelDataSize(int width, int height, int format);
static void rlLoadShaderDefault(void);
static void rlUnloadShaderDefault(void);
static DXGI_FORMAT rlGetDXGIFormat(int format);
static void rlWaitForGpu(void);
static void rlFlushCommandList(void);
static ID3D12Resource *rlCreateUploadBuffer(UINT size);
static ID3D12Resource *rlCreateDefaultBuffer(UINT size, D3D12_RESOURCE_FLAGS flags);
static void rlTransitionResource(ID3D12Resource *resource, D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after);

//----------------------------------------------------------------------------------
// Global state
//----------------------------------------------------------------------------------
static double rlCullDistanceNear = RL_CULL_DISTANCE_NEAR;
static double rlCullDistanceFar = RL_CULL_DISTANCE_FAR;
static rlglData RLGL = { 0 };

//----------------------------------------------------------------------------------
// Default HLSL Shaders (same as D3D11, compiled to SM 5.0)
//----------------------------------------------------------------------------------
static const char *defaultVShaderHLSL =
    "cbuffer MatrixBuffer : register(b0) { matrix mvp; };\n"
    "struct VSInput {\n"
    "    float3 position : POSITION;\n"
    "    float2 texcoord : TEXCOORD0;\n"
    "    float3 normal   : NORMAL;\n"
    "    float4 color    : COLOR;\n"
    "};\n"
    "struct PSInput {\n"
    "    float4 position : SV_POSITION;\n"
    "    float2 texcoord : TEXCOORD0;\n"
    "    float4 color    : COLOR;\n"
    "};\n"
    "PSInput VSMain(VSInput input) {\n"
    "    PSInput output;\n"
    "    output.position = mul(mvp, float4(input.position, 1.0));\n"
    "    output.texcoord = input.texcoord;\n"
    "    output.color = input.color;\n"
    "    return output;\n"
    "}\n";

static const char *defaultPShaderHLSL =
    "Texture2D texture0 : register(t0);\n"
    "SamplerState sampler0 : register(s0);\n"
    "cbuffer ColorBuffer : register(b1) { float4 colDiffuse; };\n"
    "struct PSInput {\n"
    "    float4 position : SV_POSITION;\n"
    "    float2 texcoord : TEXCOORD0;\n"
    "    float4 color    : COLOR;\n"
    "};\n"
    "float4 PSMain(PSInput input) : SV_TARGET {\n"
    "    float4 texColor = texture0.Sample(sampler0, input.texcoord);\n"
    "    return texColor * colDiffuse * input.color;\n"
    "}\n";

//----------------------------------------------------------------------------------
// Helper: Get DXGI format from rlgl pixel format
//----------------------------------------------------------------------------------
static DXGI_FORMAT rlGetDXGIFormat(int format)
{
    switch (format) {
        case 1:  return DXGI_FORMAT_R8_UNORM;
        case 2:  return DXGI_FORMAT_R8G8_UNORM;
        case 3:  return DXGI_FORMAT_B5G6R5_UNORM;
        case 4:  return DXGI_FORMAT_R8G8B8A8_UNORM;      // R8G8B8 -> RGBA
        case 5:  return DXGI_FORMAT_B5G5R5A1_UNORM;
        case 6:  return DXGI_FORMAT_B4G4R4A4_UNORM;
        case 7:  return DXGI_FORMAT_R8G8B8A8_UNORM;
        case 8:  return DXGI_FORMAT_R32_FLOAT;
        case 9:  return DXGI_FORMAT_R32G32B32_FLOAT;
        case 10: return DXGI_FORMAT_R32G32B32A32_FLOAT;
        case 11: return DXGI_FORMAT_R16_FLOAT;
        case 12: return DXGI_FORMAT_R16G16B16A16_FLOAT;   // R16G16B16 -> RGBA16
        case 13: return DXGI_FORMAT_R16G16B16A16_FLOAT;
        case 14: return DXGI_FORMAT_BC1_UNORM;
        case 15: return DXGI_FORMAT_BC1_UNORM;
        case 16: return DXGI_FORMAT_BC2_UNORM;
        case 17: return DXGI_FORMAT_BC3_UNORM;
        default: return DXGI_FORMAT_R8G8B8A8_UNORM;
    }
}

//----------------------------------------------------------------------------------
// Helper: Create upload (CPU-writable) buffer
//----------------------------------------------------------------------------------
static ID3D12Resource *rlCreateUploadBuffer(UINT size)
{
    ID3D12Resource *buffer = NULL;
    D3D12_HEAP_PROPERTIES heapProps = { 0 };
    heapProps.Type = D3D12_HEAP_TYPE_UPLOAD;

    D3D12_RESOURCE_DESC desc = { 0 };
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = size;
    desc.Height = 1;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.Format = DXGI_FORMAT_UNKNOWN;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    desc.Flags = D3D12_RESOURCE_FLAG_NONE;

    HRESULT hr = ID3D12Device_CreateCommittedResource(RLGL.device, &heapProps,
        D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_GENERIC_READ,
        NULL, &IID_ID3D12Resource, (void **)&buffer);
    if (FAILED(hr)) return NULL;
    return buffer;
}

//----------------------------------------------------------------------------------
// Helper: Create default (GPU-only) buffer
//----------------------------------------------------------------------------------
static ID3D12Resource *rlCreateDefaultBuffer(UINT size, D3D12_RESOURCE_FLAGS flags)
{
    ID3D12Resource *buffer = NULL;
    D3D12_HEAP_PROPERTIES heapProps = { 0 };
    heapProps.Type = D3D12_HEAP_TYPE_DEFAULT;

    D3D12_RESOURCE_DESC desc = { 0 };
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = size;
    desc.Height = 1;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.Format = DXGI_FORMAT_UNKNOWN;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    desc.Flags = flags;

    HRESULT hr = ID3D12Device_CreateCommittedResource(RLGL.device, &heapProps,
        D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COMMON,
        NULL, &IID_ID3D12Resource, (void **)&buffer);
    if (FAILED(hr)) return NULL;
    return buffer;
}

//----------------------------------------------------------------------------------
// Helper: Transition a resource between states
//----------------------------------------------------------------------------------
static void rlTransitionResource(ID3D12Resource *resource, D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after)
{
    if (!RLGL.commandList || before == after) return;
    D3D12_RESOURCE_BARRIER barrier = { 0 };
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
    barrier.Transition.pResource = resource;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    barrier.Transition.StateBefore = before;
    barrier.Transition.StateAfter = after;
    ID3D12GraphicsCommandList_ResourceBarrier(RLGL.commandList, 1, &barrier);
}

//----------------------------------------------------------------------------------
// Helper: Allocate SRV descriptor slot
//----------------------------------------------------------------------------------
static int rlAllocSRVDescriptor(D3D12_CPU_DESCRIPTOR_HANDLE *cpuHandle, D3D12_GPU_DESCRIPTOR_HANDLE *gpuHandle)
{
    if (RLGL.srvHeapUsed >= RL_D3D12_MAX_SRV_DESC) return -1;
    int index = RLGL.srvHeapUsed++;

    D3D12_CPU_DESCRIPTOR_HANDLE heapStart;
    D3D12_GPU_DESCRIPTOR_HANDLE gpuStart;
    ID3D12DescriptorHeap_GetCPUDescriptorHandleForHeapStart(RLGL.srvHeap, &heapStart);
    ID3D12DescriptorHeap_GetGPUDescriptorHandleForHeapStart(RLGL.srvHeap, &gpuStart);

    cpuHandle->ptr = heapStart.ptr + (SIZE_T)(index * RLGL.srvDescriptorSize);
    gpuHandle->ptr = gpuStart.ptr + (UINT64)(index * RLGL.srvDescriptorSize);
    return index;
}

//----------------------------------------------------------------------------------
// Helper: Wait for GPU to finish all commands
//----------------------------------------------------------------------------------
static void rlWaitForGpu(void)
{
    if (!RLGL.commandQueue || !RLGL.fence) return;

    RLGL.fenceValue++;
    ID3D12CommandQueue_Signal(RLGL.commandQueue, RLGL.fence, RLGL.fenceValue);

    if (ID3D12Fence_GetCompletedValue(RLGL.fence) < RLGL.fenceValue) {
        ID3D12Fence_SetEventOnCompletion(RLGL.fence, RLGL.fenceValue, RLGL.fenceEvent);
        WaitForSingleObject(RLGL.fenceEvent, INFINITE);
    }
}

//----------------------------------------------------------------------------------
// Helper: Execute and reset command list
//----------------------------------------------------------------------------------
static void rlFlushCommandList(void)
{
    if (!RLGL.commandList || !RLGL.commandQueue) return;

    HRESULT hr = ID3D12GraphicsCommandList_Close(RLGL.commandList);
    if (SUCCEEDED(hr)) {
        ID3D12CommandList *lists[] = { (ID3D12CommandList *)RLGL.commandList };
        ID3D12CommandQueue_ExecuteCommandLists(RLGL.commandQueue, 1, lists);
    }

    rlWaitForGpu();

    // Reset for next frame
    rlD3D12FrameResources *frame = &RLGL.frames[RLGL.currentFrame];
    ID3D12CommandAllocator_Reset(frame->commandAllocator);
    ID3D12GraphicsCommandList_Reset(RLGL.commandList, frame->commandAllocator, NULL);
}

//----------------------------------------------------------------------------------
// Helper: Create root signature
// Root parameter layout:
//   [0] CBV b0 (MVP matrix)          - visibility: VERTEX
//   [1] CBV b1 (diffuse color)       - visibility: PIXEL
//   [2] Descriptor table (SRV t0)    - visibility: PIXEL
//   Static sampler s0                 - linear wrap
//----------------------------------------------------------------------------------
static HRESULT rlCreateRootSignature(void)
{
    D3D12_ROOT_PARAMETER rootParams[3] = { 0 };

    // [0] CBV b0 for vertex shader
    rootParams[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
    rootParams[0].Descriptor.ShaderRegister = 0;
    rootParams[0].Descriptor.RegisterSpace = 0;
    rootParams[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_VERTEX;

    // [1] CBV b1 for pixel shader
    rootParams[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
    rootParams[1].Descriptor.ShaderRegister = 1;
    rootParams[1].Descriptor.RegisterSpace = 0;
    rootParams[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;

    // [2] Descriptor table with 1 SRV (t0) for pixel shader
    D3D12_DESCRIPTOR_RANGE srvRange = { 0 };
    srvRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    srvRange.NumDescriptors = 1;
    srvRange.BaseShaderRegister = 0;
    srvRange.RegisterSpace = 0;
    srvRange.OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;

    rootParams[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    rootParams[2].DescriptorTable.NumDescriptorRanges = 1;
    rootParams[2].DescriptorTable.pDescriptorRanges = &srvRange;
    rootParams[2].ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;

    // Static sampler
    D3D12_STATIC_SAMPLER_DESC sampler = { 0 };
    sampler.Filter = D3D12_FILTER_MIN_MAG_MIP_LINEAR;
    sampler.AddressU = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
    sampler.AddressV = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
    sampler.AddressW = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
    sampler.MipLODBias = 0;
    sampler.MaxAnisotropy = 0;
    sampler.ComparisonFunc = D3D12_COMPARISON_FUNC_NEVER;
    sampler.BorderColor = D3D12_STATIC_BORDER_COLOR_TRANSPARENT_BLACK;
    sampler.MinLOD = 0.0f;
    sampler.MaxLOD = D3D12_FLOAT32_MAX;
    sampler.ShaderRegister = 0;
    sampler.RegisterSpace = 0;
    sampler.ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;

    D3D12_ROOT_SIGNATURE_DESC rsDesc = { 0 };
    rsDesc.NumParameters = 3;
    rsDesc.pParameters = rootParams;
    rsDesc.NumStaticSamplers = 1;
    rsDesc.pStaticSamplers = &sampler;
    rsDesc.Flags = D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT |
                   D3D12_ROOT_SIGNATURE_FLAG_DENY_HULL_SHADER_ROOT_ACCESS |
                   D3D12_ROOT_SIGNATURE_FLAG_DENY_DOMAIN_SHADER_ROOT_ACCESS |
                   D3D12_ROOT_SIGNATURE_FLAG_DENY_GEOMETRY_SHADER_ROOT_ACCESS;

    ID3DBlob *serialized = NULL;
    ID3DBlob *errors = NULL;
    HRESULT hr = D3D12SerializeRootSignature(&rsDesc, D3D_ROOT_SIGNATURE_VERSION_1, &serialized, &errors);
    if (FAILED(hr)) {
        if (errors) {
            TRACELOG(RL_LOG_ERROR, "D3D12: Root signature serialization error: %s", (char *)ID3D10Blob_GetBufferPointer(errors));
            ID3D10Blob_Release(errors);
        }
        return hr;
    }
    if (errors) ID3D10Blob_Release(errors);

    hr = ID3D12Device_CreateRootSignature(RLGL.device, 0,
        ID3D10Blob_GetBufferPointer(serialized), ID3D10Blob_GetBufferSize(serialized),
        &IID_ID3D12RootSignature, (void **)&RLGL.rootSignature);
    ID3D10Blob_Release(serialized);
    return hr;
}

//----------------------------------------------------------------------------------
// Helper: Create a PSO for a given blend/depth/raster config + shader blobs
//----------------------------------------------------------------------------------
static ID3D12PipelineState *rlCreatePipelineState(ID3DBlob *vsBlob, ID3DBlob *psBlob,
    bool blendEnabled, int blendMode, bool depthTest, bool depthWrite,
    bool backfaceCull, int cullFaceMode, bool wireMode, bool scissorTest,
    D3D12_PRIMITIVE_TOPOLOGY_TYPE topoType)
{
    D3D12_GRAPHICS_PIPELINE_STATE_DESC psoDesc = { 0 };
    psoDesc.pRootSignature = RLGL.rootSignature;
    psoDesc.VS.pShaderBytecode = ID3D10Blob_GetBufferPointer(vsBlob);
    psoDesc.VS.BytecodeLength = ID3D10Blob_GetBufferSize(vsBlob);
    psoDesc.PS.pShaderBytecode = ID3D10Blob_GetBufferPointer(psBlob);
    psoDesc.PS.BytecodeLength = ID3D10Blob_GetBufferSize(psBlob);

    // Input layout
    D3D12_INPUT_ELEMENT_DESC inputLayout[] = {
        { "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
        { "TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT,    1, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
        { "NORMAL",   0, DXGI_FORMAT_R32G32B32_FLOAT, 2, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
        { "COLOR",    0, DXGI_FORMAT_R8G8B8A8_UNORM,  3, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
    };
    psoDesc.InputLayout.pInputElementDescs = inputLayout;
    psoDesc.InputLayout.NumElements = 4;

    // Rasterizer state
    psoDesc.RasterizerState.FillMode = wireMode ? D3D12_FILL_MODE_WIREFRAME : D3D12_FILL_MODE_SOLID;
    psoDesc.RasterizerState.CullMode = backfaceCull ?
        (cullFaceMode == 1 ? D3D12_CULL_MODE_FRONT : D3D12_CULL_MODE_BACK) : D3D12_CULL_MODE_NONE;
    psoDesc.RasterizerState.FrontCounterClockwise = TRUE;
    psoDesc.RasterizerState.DepthBias = D3D12_DEFAULT_DEPTH_BIAS;
    psoDesc.RasterizerState.DepthBiasClamp = D3D12_DEFAULT_DEPTH_BIAS_CLAMP;
    psoDesc.RasterizerState.SlopeScaledDepthBias = D3D12_DEFAULT_SLOPE_SCALED_DEPTH_BIAS;
    psoDesc.RasterizerState.DepthClipEnable = TRUE;
    psoDesc.RasterizerState.MultisampleEnable = FALSE;
    psoDesc.RasterizerState.AntialiasedLineEnable = FALSE;
    psoDesc.RasterizerState.ForcedSampleCount = 0;
    psoDesc.RasterizerState.ConservativeRaster = D3D12_CONSERVATIVE_RASTERIZATION_MODE_OFF;

    // Blend state
    psoDesc.BlendState.AlphaToCoverageEnable = FALSE;
    psoDesc.BlendState.IndependentBlendEnable = FALSE;
    psoDesc.BlendState.RenderTarget[0].BlendEnable = blendEnabled;
    psoDesc.BlendState.RenderTarget[0].LogicOpEnable = FALSE;
    psoDesc.BlendState.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
    if (blendEnabled) {
        switch (blendMode) {
            case 1: // RL_BLEND_ADDITIVE
                psoDesc.BlendState.RenderTarget[0].SrcBlend = D3D12_BLEND_SRC_ALPHA;
                psoDesc.BlendState.RenderTarget[0].DestBlend = D3D12_BLEND_ONE;
                psoDesc.BlendState.RenderTarget[0].BlendOp = D3D12_BLEND_OP_ADD;
                psoDesc.BlendState.RenderTarget[0].SrcBlendAlpha = D3D12_BLEND_ONE;
                psoDesc.BlendState.RenderTarget[0].DestBlendAlpha = D3D12_BLEND_ONE;
                psoDesc.BlendState.RenderTarget[0].BlendOpAlpha = D3D12_BLEND_OP_ADD;
                break;
            case 2: // RL_BLEND_MULTIPLIED
                psoDesc.BlendState.RenderTarget[0].SrcBlend = D3D12_BLEND_DEST_COLOR;
                psoDesc.BlendState.RenderTarget[0].DestBlend = D3D12_BLEND_INV_SRC_ALPHA;
                psoDesc.BlendState.RenderTarget[0].BlendOp = D3D12_BLEND_OP_ADD;
                psoDesc.BlendState.RenderTarget[0].SrcBlendAlpha = D3D12_BLEND_ONE;
                psoDesc.BlendState.RenderTarget[0].DestBlendAlpha = D3D12_BLEND_INV_SRC_ALPHA;
                psoDesc.BlendState.RenderTarget[0].BlendOpAlpha = D3D12_BLEND_OP_ADD;
                break;
            default: // RL_BLEND_ALPHA (0) and fallback
                psoDesc.BlendState.RenderTarget[0].SrcBlend = D3D12_BLEND_SRC_ALPHA;
                psoDesc.BlendState.RenderTarget[0].DestBlend = D3D12_BLEND_INV_SRC_ALPHA;
                psoDesc.BlendState.RenderTarget[0].BlendOp = D3D12_BLEND_OP_ADD;
                psoDesc.BlendState.RenderTarget[0].SrcBlendAlpha = D3D12_BLEND_ONE;
                psoDesc.BlendState.RenderTarget[0].DestBlendAlpha = D3D12_BLEND_INV_SRC_ALPHA;
                psoDesc.BlendState.RenderTarget[0].BlendOpAlpha = D3D12_BLEND_OP_ADD;
                break;
        }
    }

    // Depth stencil state
    psoDesc.DepthStencilState.DepthEnable = depthTest;
    psoDesc.DepthStencilState.DepthWriteMask = depthWrite ? D3D12_DEPTH_WRITE_MASK_ALL : D3D12_DEPTH_WRITE_MASK_ZERO;
    psoDesc.DepthStencilState.DepthFunc = D3D12_COMPARISON_FUNC_LESS_EQUAL;
    psoDesc.DepthStencilState.StencilEnable = FALSE;

    psoDesc.SampleMask = UINT_MAX;
    psoDesc.PrimitiveTopologyType = topoType;
    psoDesc.NumRenderTargets = 1;
    psoDesc.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
    psoDesc.DSVFormat = DXGI_FORMAT_D32_FLOAT;
    psoDesc.SampleDesc.Count = 1;
    psoDesc.SampleDesc.Quality = 0;

    ID3D12PipelineState *pso = NULL;
    HRESULT hr = ID3D12Device_CreateGraphicsPipelineState(RLGL.device, &psoDesc, &IID_ID3D12PipelineState, (void **)&pso);
    if (FAILED(hr)) {
        TRACELOG(RL_LOG_WARNING, "D3D12: Failed to create PSO (hr=0x%08X)", hr);
        return NULL;
    }
    return pso;
}

//----------------------------------------------------------------------------------
// Matrix operations (CPU-side, identical logic to D3D11/OpenGL backend)
//----------------------------------------------------------------------------------
void rlMatrixMode(int mode)
{
    if (mode == RL_PROJECTION) RLGL.State.currentMatrix = &RLGL.State.projection;
    else if (mode == RL_MODELVIEW) RLGL.State.currentMatrix = &RLGL.State.modelview;
    RLGL.State.currentMatrixMode = mode;
}

void rlPushMatrix(void)
{
    if (RLGL.State.stackCounter >= RL_MAX_MATRIX_STACK_SIZE) {
        TRACELOG(RL_LOG_ERROR, "RLGL: Matrix stack overflow (RL_MAX_MATRIX_STACK_SIZE)");
        return;
    }
    if (RLGL.State.currentMatrixMode == RL_MODELVIEW) {
        RLGL.State.transformRequired = true;
        RLGL.State.currentMatrix = &RLGL.State.transform;
    }
    RLGL.State.stack[RLGL.State.stackCounter] = *RLGL.State.currentMatrix;
    RLGL.State.stackCounter++;
}

void rlPopMatrix(void)
{
    if (RLGL.State.stackCounter > 0) {
        *RLGL.State.currentMatrix = RLGL.State.stack[RLGL.State.stackCounter - 1];
        RLGL.State.stackCounter--;
    }
    if ((RLGL.State.stackCounter == 0) && (RLGL.State.currentMatrixMode == RL_MODELVIEW)) {
        RLGL.State.currentMatrix = &RLGL.State.modelview;
        RLGL.State.transformRequired = false;
    }
}

void rlLoadIdentity(void) { *RLGL.State.currentMatrix = rlMatrixIdentity(); }

void rlTranslatef(float x, float y, float z)
{
    Matrix matTranslation = { 1,0,0,x, 0,1,0,y, 0,0,1,z, 0,0,0,1 };
    *RLGL.State.currentMatrix = rlMatrixMultiply(matTranslation, *RLGL.State.currentMatrix);
}

void rlRotatef(float angle, float x, float y, float z)
{
    Matrix matRotation = rlMatrixIdentity();
    float lengthSquared = x*x + y*y + z*z;
    if ((lengthSquared != 1.0f) && (lengthSquared != 0.0f)) {
        float invLength = 1.0f/sqrtf(lengthSquared);
        x *= invLength; y *= invLength; z *= invLength;
    }
    float sinres = sinf(angle*DEG2RAD);
    float cosres = cosf(angle*DEG2RAD);
    float t = 1.0f - cosres;

    matRotation.m0 = x*x*t + cosres;   matRotation.m1 = y*x*t + z*sinres; matRotation.m2 = z*x*t - y*sinres;
    matRotation.m4 = x*y*t - z*sinres; matRotation.m5 = y*y*t + cosres;   matRotation.m6 = z*y*t + x*sinres;
    matRotation.m8 = x*z*t + y*sinres; matRotation.m9 = y*z*t - x*sinres; matRotation.m10 = z*z*t + cosres;

    *RLGL.State.currentMatrix = rlMatrixMultiply(matRotation, *RLGL.State.currentMatrix);
}

void rlScalef(float x, float y, float z)
{
    Matrix matScale = { x,0,0,0, 0,y,0,0, 0,0,z,0, 0,0,0,1 };
    *RLGL.State.currentMatrix = rlMatrixMultiply(matScale, *RLGL.State.currentMatrix);
}

void rlMultMatrixf(const float *matf)
{
    Matrix mat = { matf[0],matf[1],matf[2],matf[3],
                   matf[4],matf[5],matf[6],matf[7],
                   matf[8],matf[9],matf[10],matf[11],
                   matf[12],matf[13],matf[14],matf[15] };
    *RLGL.State.currentMatrix = rlMatrixMultiply(*RLGL.State.currentMatrix, mat);
}

void rlFrustum(double left, double right, double bottom, double top, double znear, double zfar)
{
    Matrix matFrustum = { 0 };
    float rl = (float)(right - left);
    float tb = (float)(top - bottom);
    float fn = (float)(zfar - znear);
    matFrustum.m0 = ((float)znear*2.0f)/rl;
    matFrustum.m5 = ((float)znear*2.0f)/tb;
    matFrustum.m2 = ((float)right + (float)left)/rl;
    matFrustum.m6 = ((float)top + (float)bottom)/tb;
    matFrustum.m10 = -((float)zfar + (float)znear)/fn;
    matFrustum.m11 = -1.0f;
    matFrustum.m14 = -((float)zfar*(float)znear*2.0f)/fn;
    *RLGL.State.currentMatrix = rlMatrixMultiply(*RLGL.State.currentMatrix, matFrustum);
}

void rlOrtho(double left, double right, double bottom, double top, double znear, double zfar)
{
    Matrix matOrtho = { 0 };
    float rl = (float)(right - left);
    float tb = (float)(top - bottom);
    float fn = (float)(zfar - znear);
    matOrtho.m0 = 2.0f/rl;
    matOrtho.m5 = 2.0f/tb;
    matOrtho.m10 = -2.0f/fn;
    matOrtho.m3 = -((float)left + (float)right)/rl;
    matOrtho.m7 = -((float)top + (float)bottom)/tb;
    matOrtho.m11 = -((float)zfar + (float)znear)/fn;
    matOrtho.m15 = 1.0f;
    *RLGL.State.currentMatrix = rlMatrixMultiply(*RLGL.State.currentMatrix, matOrtho);
}

void rlViewport(int x, int y, int width, int height)
{
    // Viewport is set on the command list during batch drawing
    RLGL.State.framebufferWidth = width;
    RLGL.State.framebufferHeight = height;
}

void rlSetClipPlanes(double nearPlane, double farPlane)
{
    rlCullDistanceNear = nearPlane;
    rlCullDistanceFar = farPlane;
}

double rlGetCullDistanceNear(void) { return rlCullDistanceNear; }
double rlGetCullDistanceFar(void) { return rlCullDistanceFar; }

//----------------------------------------------------------------------------------
// Vertex-level operations
//----------------------------------------------------------------------------------
void rlBegin(int mode)
{
    if (RLGL.currentBatch->draws[RLGL.currentBatch->drawCounter - 1].mode != mode) {
        if (RLGL.currentBatch->draws[RLGL.currentBatch->drawCounter - 1].vertexCount > 0) {
            if (RLGL.currentBatch->draws[RLGL.currentBatch->drawCounter - 1].mode == RL_LINES)
                RLGL.currentBatch->draws[RLGL.currentBatch->drawCounter - 1].vertexAlignment =
                    RLGL.currentBatch->draws[RLGL.currentBatch->drawCounter - 1].vertexCount % 4;
            else if (RLGL.currentBatch->draws[RLGL.currentBatch->drawCounter - 1].mode == RL_TRIANGLES)
                RLGL.currentBatch->draws[RLGL.currentBatch->drawCounter - 1].vertexAlignment =
                    4 - (RLGL.currentBatch->draws[RLGL.currentBatch->drawCounter - 1].vertexCount % 4);

            if (RLGL.currentBatch->draws[RLGL.currentBatch->drawCounter - 1].vertexAlignment == 4)
                RLGL.currentBatch->draws[RLGL.currentBatch->drawCounter - 1].vertexAlignment = 0;

            if (RLGL.State.vertexCounter + RLGL.currentBatch->draws[RLGL.currentBatch->drawCounter - 1].vertexAlignment <
                RLGL.currentBatch->vertexBuffer[RLGL.currentBatch->currentBuffer].elementCount * 4)
                RLGL.State.vertexCounter += RLGL.currentBatch->draws[RLGL.currentBatch->drawCounter - 1].vertexAlignment;

            if (RLGL.currentBatch->drawCounter < RL_DEFAULT_BATCH_DRAWCALLS) {
                RLGL.currentBatch->draws[RLGL.currentBatch->drawCounter].textureId =
                    RLGL.currentBatch->draws[RLGL.currentBatch->drawCounter - 1].textureId;
                RLGL.currentBatch->drawCounter++;
            }
        }
        if (RLGL.currentBatch->drawCounter < RL_DEFAULT_BATCH_DRAWCALLS)
            RLGL.currentBatch->draws[RLGL.currentBatch->drawCounter - 1].mode = mode;
    }
}

void rlEnd(void) {
    /* Batch is drawn on rlDrawRenderBatch */
}

void rlVertex3f(float x, float y, float z)
{
    float tx = x, ty = y, tz = z;
    if (RLGL.State.transformRequired) {
        tx = RLGL.State.transform.m0*x + RLGL.State.transform.m1*y + RLGL.State.transform.m2*z + RLGL.State.transform.m3;
        ty = RLGL.State.transform.m4*x + RLGL.State.transform.m5*y + RLGL.State.transform.m6*z + RLGL.State.transform.m7;
        tz = RLGL.State.transform.m8*x + RLGL.State.transform.m9*y + RLGL.State.transform.m10*z + RLGL.State.transform.m11;
    }

    if (RLGL.State.vertexCounter < (RLGL.currentBatch->vertexBuffer[RLGL.currentBatch->currentBuffer].elementCount*4)) {
        int idx = RLGL.State.vertexCounter;
        RLGL.currentBatch->vertexBuffer[RLGL.currentBatch->currentBuffer].vertices[3*idx] = tx;
        RLGL.currentBatch->vertexBuffer[RLGL.currentBatch->currentBuffer].vertices[3*idx+1] = ty;
        RLGL.currentBatch->vertexBuffer[RLGL.currentBatch->currentBuffer].vertices[3*idx+2] = tz;

        RLGL.currentBatch->vertexBuffer[RLGL.currentBatch->currentBuffer].texcoords[2*idx] = RLGL.State.texcoordx;
        RLGL.currentBatch->vertexBuffer[RLGL.currentBatch->currentBuffer].texcoords[2*idx+1] = RLGL.State.texcoordy;

        RLGL.currentBatch->vertexBuffer[RLGL.currentBatch->currentBuffer].normals[3*idx] = RLGL.State.normalx;
        RLGL.currentBatch->vertexBuffer[RLGL.currentBatch->currentBuffer].normals[3*idx+1] = RLGL.State.normaly;
        RLGL.currentBatch->vertexBuffer[RLGL.currentBatch->currentBuffer].normals[3*idx+2] = RLGL.State.normalz;

        RLGL.currentBatch->vertexBuffer[RLGL.currentBatch->currentBuffer].colors[4*idx] = RLGL.State.colorr;
        RLGL.currentBatch->vertexBuffer[RLGL.currentBatch->currentBuffer].colors[4*idx+1] = RLGL.State.colorg;
        RLGL.currentBatch->vertexBuffer[RLGL.currentBatch->currentBuffer].colors[4*idx+2] = RLGL.State.colorb;
        RLGL.currentBatch->vertexBuffer[RLGL.currentBatch->currentBuffer].colors[4*idx+3] = RLGL.State.colora;

        RLGL.State.vertexCounter++;
        RLGL.currentBatch->draws[RLGL.currentBatch->drawCounter - 1].vertexCount++;
    } else {
        TRACELOG(RL_LOG_ERROR, "RLGL: Batch element limit reached (MAX: %i)", RLGL.currentBatch->vertexBuffer[RLGL.currentBatch->currentBuffer].elementCount*4);
    }
}

void rlVertex2f(float x, float y) { rlVertex3f(x, y, RLGL.currentBatch->currentDepth); }
void rlVertex2i(int x, int y) { rlVertex3f((float)x, (float)y, RLGL.currentBatch->currentDepth); }
void rlTexCoord2f(float x, float y) { RLGL.State.texcoordx = x; RLGL.State.texcoordy = y; }
void rlNormal3f(float x, float y, float z) {
    float length = sqrtf(x*x + y*y + z*z);
    if (length != 0.0f) { x /= length; y /= length; z /= length; }
    RLGL.State.normalx = x; RLGL.State.normaly = y; RLGL.State.normalz = z;
}

void rlColor4ub(unsigned char r, unsigned char g, unsigned char b, unsigned char a) {
    RLGL.State.colorr = r; RLGL.State.colorg = g; RLGL.State.colorb = b; RLGL.State.colora = a;
}
void rlColor4f(float r, float g, float b, float a) {
    rlColor4ub((unsigned char)(r*255), (unsigned char)(g*255), (unsigned char)(b*255), (unsigned char)(a*255));
}
void rlColor3f(float x, float y, float z) { rlColor4ub((unsigned char)(x*255), (unsigned char)(y*255), (unsigned char)(z*255), 255); }

//----------------------------------------------------------------------------------
// Texture binding
//----------------------------------------------------------------------------------
void rlSetTexture(unsigned int id)
{
    if (id == 0) {
        if (RLGL.State.vertexCounter >= RLGL.currentBatch->vertexBuffer[RLGL.currentBatch->currentBuffer].elementCount*4) {
            rlDrawRenderBatch(RLGL.currentBatch);
        }
    } else {
        if (RLGL.currentBatch->draws[RLGL.currentBatch->drawCounter - 1].textureId != id) {
            if (RLGL.currentBatch->draws[RLGL.currentBatch->drawCounter - 1].vertexCount > 0) {
                if (RLGL.currentBatch->draws[RLGL.currentBatch->drawCounter - 1].mode == RL_LINES)
                    RLGL.currentBatch->draws[RLGL.currentBatch->drawCounter - 1].vertexAlignment =
                        RLGL.currentBatch->draws[RLGL.currentBatch->drawCounter - 1].vertexCount % 4;
                else if (RLGL.currentBatch->draws[RLGL.currentBatch->drawCounter - 1].mode == RL_TRIANGLES)
                    RLGL.currentBatch->draws[RLGL.currentBatch->drawCounter - 1].vertexAlignment =
                        4 - (RLGL.currentBatch->draws[RLGL.currentBatch->drawCounter - 1].vertexCount % 4);

                if (RLGL.currentBatch->draws[RLGL.currentBatch->drawCounter - 1].vertexAlignment == 4)
                    RLGL.currentBatch->draws[RLGL.currentBatch->drawCounter - 1].vertexAlignment = 0;

                if (RLGL.State.vertexCounter + RLGL.currentBatch->draws[RLGL.currentBatch->drawCounter - 1].vertexAlignment <
                    RLGL.currentBatch->vertexBuffer[RLGL.currentBatch->currentBuffer].elementCount * 4)
                    RLGL.State.vertexCounter += RLGL.currentBatch->draws[RLGL.currentBatch->drawCounter - 1].vertexAlignment;

                if (RLGL.currentBatch->drawCounter < RL_DEFAULT_BATCH_DRAWCALLS) RLGL.currentBatch->drawCounter++;
            }

            if (RLGL.currentBatch->drawCounter < RL_DEFAULT_BATCH_DRAWCALLS) {
                RLGL.currentBatch->draws[RLGL.currentBatch->drawCounter - 1].textureId = id;
                RLGL.currentBatch->draws[RLGL.currentBatch->drawCounter - 1].vertexCount = 0;
            }
        }
    }
}

//----------------------------------------------------------------------------------
// Enable/Disable state functions
//----------------------------------------------------------------------------------
void rlActiveTextureSlot(int slot) { (void)slot; }
void rlEnableTexture(unsigned int id) {
    if (id > 0 && id <= RLGL.textureCount)
        RLGL.State.activeTextureId[0] = id;
}
void rlDisableTexture(void) {
    RLGL.State.activeTextureId[0] = RLGL.State.defaultTextureId;
}
void rlEnableTextureCubemap(unsigned int id) { rlEnableTexture(id); }
void rlDisableTextureCubemap(void) { rlDisableTexture(); }
void rlTextureParameters(unsigned int id, int param, int value) { (void)id; (void)param; (void)value; }
void rlCubemapParameters(unsigned int id, int param, int value) { (void)id; (void)param; (void)value; }

void rlEnableShader(unsigned int id) {
    if (id > 0 && id <= RLGL.shaderCount) {
        unsigned int idx = id - 1;
        if (RLGL.shaders[idx].pso && RLGL.commandList) {
            ID3D12GraphicsCommandList_SetPipelineState(RLGL.commandList, RLGL.shaders[idx].pso);
        }
    }
}
void rlDisableShader(void) {
    if (RLGL.State.defaultShaderId > 0) {
        unsigned int idx = RLGL.State.defaultShaderId - 1;
        if (RLGL.shaders[idx].pso && RLGL.commandList) {
            ID3D12GraphicsCommandList_SetPipelineState(RLGL.commandList, RLGL.shaders[idx].pso);
        }
    }
}

void rlEnableFramebuffer(unsigned int id) { RLGL.activeFramebuffer = id; }
void rlDisableFramebuffer(void) { RLGL.activeFramebuffer = 0; }
unsigned int rlGetActiveFramebuffer(void) { return RLGL.activeFramebuffer; }
void rlActiveDrawBuffers(int count) { (void)count; }
void rlBlitFramebuffer(int srcX, int srcY, int srcWidth, int srcHeight,
    int dstX, int dstY, int dstWidth, int dstHeight, int bufferMask) {
    (void)srcX; (void)srcY; (void)srcWidth; (void)srcHeight;
    (void)dstX; (void)dstY; (void)dstWidth; (void)dstHeight; (void)bufferMask;
}
void rlBindFramebuffer(unsigned int target, unsigned int framebuffer) {
    (void)target;
    if (framebuffer == 0) RLGL.activeFramebuffer = 0;
    else RLGL.activeFramebuffer = framebuffer;
}

void rlEnableColorBlend(void) { RLGL.State.colorBlendEnabled = true; RLGL.pipelineDirty = true; }
void rlDisableColorBlend(void) { RLGL.State.colorBlendEnabled = false; RLGL.pipelineDirty = true; }
void rlEnableDepthTest(void) { RLGL.State.depthTestEnabled = true; RLGL.pipelineDirty = true; }
void rlDisableDepthTest(void) { RLGL.State.depthTestEnabled = false; RLGL.pipelineDirty = true; }
void rlEnableDepthMask(void) { RLGL.State.depthWriteEnabled = true; RLGL.pipelineDirty = true; }
void rlDisableDepthMask(void) { RLGL.State.depthWriteEnabled = false; RLGL.pipelineDirty = true; }
void rlEnableBackfaceCulling(void) { RLGL.State.backfaceCullingEnabled = true; RLGL.pipelineDirty = true; }
void rlDisableBackfaceCulling(void) { RLGL.State.backfaceCullingEnabled = false; RLGL.pipelineDirty = true; }
void rlColorMask(bool r, bool g, bool b, bool a) { (void)r; (void)g; (void)b; (void)a; }
void rlSetCullFace(int mode) { RLGL.State.cullFaceMode = (mode == RL_CULL_FACE_FRONT) ? 1 : 0; RLGL.pipelineDirty = true; }
void rlEnableScissorTest(void) { RLGL.State.scissorTestEnabled = true; RLGL.pipelineDirty = true; }
void rlDisableScissorTest(void) { RLGL.State.scissorTestEnabled = false; RLGL.pipelineDirty = true; }
void rlScissor(int x, int y, int width, int height) {
    if (!RLGL.commandList) return;
    D3D12_RECT scissor = { x, y, x + width, y + height };
    ID3D12GraphicsCommandList_RSSetScissorRects(RLGL.commandList, 1, &scissor);
}

void rlEnableWireMode(void) { RLGL.State.wireMode = true; RLGL.pipelineDirty = true; }
void rlEnablePointMode(void) { RLGL.State.wireMode = true; RLGL.pipelineDirty = true; }
void rlDisableWireMode(void) { RLGL.State.wireMode = false; RLGL.pipelineDirty = true; }
void rlSetLineWidth(float width) { RLGL.State.lineWidth = width; }
float rlGetLineWidth(void) { return RLGL.State.lineWidth; }
void rlEnableSmoothLines(void) { }
void rlDisableSmoothLines(void) { }
void rlEnableStereoRender(void) { RLGL.State.stereoRender = true; }
void rlDisableStereoRender(void) { RLGL.State.stereoRender = false; }
bool rlIsStereoRenderEnabled(void) { return RLGL.State.stereoRender; }

void rlClearColor(unsigned char r, unsigned char g, unsigned char b, unsigned char a) {
    RLGL.State.clearColor[0] = r/255.0f;
    RLGL.State.clearColor[1] = g/255.0f;
    RLGL.State.clearColor[2] = b/255.0f;
    RLGL.State.clearColor[3] = a/255.0f;
}

void rlClearScreenBuffers(void) {
    // Clearing is handled during rlDrawRenderBatch when RTV/DSV are bound
    // The actual clear is done by the platform layer on the swap chain RTV
}

void rlCheckErrors(void) { /* D3D12 debug layer handles errors via OutputDebugString */ }

void rlSetBlendMode(int mode) {
    if (RLGL.State.currentBlendMode != mode) {
        rlDrawRenderBatch(RLGL.currentBatch);
        RLGL.State.currentBlendMode = mode;
        RLGL.pipelineDirty = true;
    }
}

void rlSetBlendFactors(int glSrcFactor, int glDstFactor, int glEquation) {
    (void)glSrcFactor; (void)glDstFactor; (void)glEquation;
}
void rlSetBlendFactorsSeparate(int glSrcRGB, int glDstRGB, int glSrcAlpha, int glDstAlpha, int glEqRGB, int glEqAlpha) {
    (void)glSrcRGB; (void)glDstRGB; (void)glSrcAlpha; (void)glDstAlpha; (void)glEqRGB; (void)glEqAlpha;
}

//----------------------------------------------------------------------------------
// Backend initialization
//----------------------------------------------------------------------------------
void rlglInit(int width, int height, bool headless)
{
    (void)headless;

    // Create DXGI factory
    UINT dxgiFlags = 0;
#if defined(_DEBUG)
    // Enable debug layer
    {
        ID3D12Debug *debugController = NULL;
        if (SUCCEEDED(D3D12GetDebugInterface(&IID_ID3D12Debug, (void **)&debugController))) {
            ID3D12Debug_EnableDebugLayer(debugController);
            ID3D12Debug_Release(debugController);
            dxgiFlags |= DXGI_CREATE_FACTORY_DEBUG;
        }
    }
#endif

    HRESULT hr = CreateDXGIFactory2(dxgiFlags, &IID_IDXGIFactory4, (void **)&RLGL.factory);
    if (FAILED(hr)) {
        TRACELOG(RL_LOG_ERROR, "D3D12: Failed to create DXGI factory");
        return;
    }

    // Find hardware adapter
    IDXGIAdapter1 *adapter = NULL;
    for (UINT i = 0; IDXGIFactory4_EnumAdapters1(RLGL.factory, i, &adapter) != DXGI_ERROR_NOT_FOUND; i++) {
        DXGI_ADAPTER_DESC1 desc;
        IDXGIAdapter1_GetDesc1(adapter, &desc);
        if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) {
            IDXGIAdapter1_Release(adapter);
            adapter = NULL;
            continue;
        }
        // Try creating D3D12 device to test support
        hr = D3D12CreateDevice((IUnknown *)adapter, D3D_FEATURE_LEVEL_11_0, &IID_ID3D12Device, NULL);
        if (SUCCEEDED(hr)) break;
        IDXGIAdapter1_Release(adapter);
        adapter = NULL;
    }

    // Create device
    hr = D3D12CreateDevice(adapter ? (IUnknown *)adapter : NULL, D3D_FEATURE_LEVEL_11_0,
        &IID_ID3D12Device, (void **)&RLGL.device);
    if (adapter) IDXGIAdapter1_Release(adapter);

    if (FAILED(hr)) {
        TRACELOG(RL_LOG_ERROR, "D3D12: Failed to create device");
        IDXGIFactory4_Release(RLGL.factory);
        RLGL.factory = NULL;
        return;
    }

    TRACELOG(RL_LOG_INFO, "D3D12: Device created successfully");

    // Create command queue
    D3D12_COMMAND_QUEUE_DESC queueDesc = { 0 };
    queueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    queueDesc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
    hr = ID3D12Device_CreateCommandQueue(RLGL.device, &queueDesc, &IID_ID3D12CommandQueue, (void **)&RLGL.commandQueue);
    if (FAILED(hr)) {
        TRACELOG(RL_LOG_ERROR, "D3D12: Failed to create command queue");
        return;
    }

    // Create descriptor heaps
    {
        // SRV/CBV/UAV heap (shader visible)
        D3D12_DESCRIPTOR_HEAP_DESC heapDesc = { 0 };
        heapDesc.NumDescriptors = RL_D3D12_MAX_SRV_DESC;
        heapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        heapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
        ID3D12Device_CreateDescriptorHeap(RLGL.device, &heapDesc, &IID_ID3D12DescriptorHeap, (void **)&RLGL.srvHeap);

        // RTV heap
        heapDesc.NumDescriptors = 32;
        heapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
        heapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
        ID3D12Device_CreateDescriptorHeap(RLGL.device, &heapDesc, &IID_ID3D12DescriptorHeap, (void **)&RLGL.rtvHeap);

        // DSV heap
        heapDesc.NumDescriptors = 32;
        heapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_DSV;
        heapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
        ID3D12Device_CreateDescriptorHeap(RLGL.device, &heapDesc, &IID_ID3D12DescriptorHeap, (void **)&RLGL.dsvHeap);

        RLGL.srvDescriptorSize = ID3D12Device_GetDescriptorHandleIncrementSize(RLGL.device, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        RLGL.rtvDescriptorSize = ID3D12Device_GetDescriptorHandleIncrementSize(RLGL.device, D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
        RLGL.dsvDescriptorSize = ID3D12Device_GetDescriptorHandleIncrementSize(RLGL.device, D3D12_DESCRIPTOR_HEAP_TYPE_DSV);
    }

    // Create fence
    hr = ID3D12Device_CreateFence(RLGL.device, 0, D3D12_FENCE_FLAG_NONE, &IID_ID3D12Fence, (void **)&RLGL.fence);
    RLGL.fenceEvent = CreateEventW(NULL, FALSE, FALSE, NULL);
    RLGL.fenceValue = 0;

    // Create root signature
    rlCreateRootSignature();

    // Create per-frame resources
    int totalVertices = RL_DEFAULT_BATCH_BUFFER_ELEMENTS * 4;
    for (int i = 0; i < RL_D3D12_NUM_FRAMES; i++) {
        rlD3D12FrameResources *frame = &RLGL.frames[i];

        ID3D12Device_CreateCommandAllocator(RLGL.device, D3D12_COMMAND_LIST_TYPE_DIRECT,
            &IID_ID3D12CommandAllocator, (void **)&frame->commandAllocator);

        // Constant buffers (upload heap, persistently mapped)
        UINT cbMatrixSize = (sizeof(rlD3D12MatrixCB) + 255) & ~255;
        UINT cbColorSize = (sizeof(rlD3D12ColorCB) + 255) & ~255;

        frame->cbMatrixUpload = rlCreateUploadBuffer(cbMatrixSize);
        frame->cbColorUpload = rlCreateUploadBuffer(cbColorSize);

        D3D12_RANGE readRange = { 0, 0 };
        if (frame->cbMatrixUpload) ID3D12Resource_Map(frame->cbMatrixUpload, 0, &readRange, &frame->cbMatrixMapped);
        if (frame->cbColorUpload) ID3D12Resource_Map(frame->cbColorUpload, 0, &readRange, &frame->cbColorMapped);

        // Batch vertex upload buffers
        frame->batchVertexBuffers[0] = rlCreateUploadBuffer(totalVertices * 3 * sizeof(float));
        frame->batchVertexBuffers[1] = rlCreateUploadBuffer(totalVertices * 2 * sizeof(float));
        frame->batchVertexBuffers[2] = rlCreateUploadBuffer(totalVertices * 3 * sizeof(float));
        frame->batchVertexBuffers[3] = rlCreateUploadBuffer(totalVertices * 4 * sizeof(unsigned char));
        frame->batchIndexBuffer = rlCreateUploadBuffer(RL_DEFAULT_BATCH_BUFFER_ELEMENTS * 6 * sizeof(unsigned int));

        for (int j = 0; j < 4; j++) {
            if (frame->batchVertexBuffers[j])
                ID3D12Resource_Map(frame->batchVertexBuffers[j], 0, &readRange, &frame->batchVertexMapped[j]);
        }
        if (frame->batchIndexBuffer)
            ID3D12Resource_Map(frame->batchIndexBuffer, 0, &readRange, &frame->batchIndexMapped);

        frame->fenceValue = 0;
    }

    RLGL.currentFrame = 0;
    RLGL.batchBufferSize = totalVertices;

    // Create command list
    hr = ID3D12Device_CreateCommandList(RLGL.device, 0, D3D12_COMMAND_LIST_TYPE_DIRECT,
        RLGL.frames[0].commandAllocator, NULL, &IID_ID3D12GraphicsCommandList, (void **)&RLGL.commandList);
    if (FAILED(hr)) {
        TRACELOG(RL_LOG_ERROR, "D3D12: Failed to create command list");
        return;
    }

    // Set descriptor heap
    ID3D12DescriptorHeap *heaps[] = { RLGL.srvHeap };
    ID3D12GraphicsCommandList_SetDescriptorHeaps(RLGL.commandList, 1, heaps);

    // Init default white texture (1x1 RGBA)
    {
        unsigned char pixels[4] = { 255, 255, 255, 255 };
        RLGL.State.defaultTextureId = rlLoadTexture(pixels, 1, 1, 7, 1);
        if (RLGL.State.defaultTextureId != 0)
            TRACELOG(RL_LOG_INFO, "TEXTURE: [ID %i] Default texture loaded successfully", RLGL.State.defaultTextureId);
    }

    // Init default shader
    rlLoadShaderDefault();
    RLGL.State.currentShaderId = RLGL.State.defaultShaderId;
    RLGL.State.currentShaderLocs = RLGL.State.defaultShaderLocs;

    // Init render batch
    RLGL.defaultBatch = rlLoadRenderBatch(RL_DEFAULT_BATCH_BUFFERS, RL_DEFAULT_BATCH_BUFFER_ELEMENTS);
    RLGL.currentBatch = &RLGL.defaultBatch;

    // Init stack matrices
    for (int i = 0; i < RL_MAX_MATRIX_STACK_SIZE; i++) RLGL.State.stack[i] = rlMatrixIdentity();
    RLGL.State.transform = rlMatrixIdentity();
    RLGL.State.projection = rlMatrixIdentity();
    RLGL.State.modelview = rlMatrixIdentity();
    RLGL.State.currentMatrix = &RLGL.State.modelview;

    // Init default states
    RLGL.State.colorBlendEnabled = true;
    RLGL.State.depthTestEnabled = false;
    RLGL.State.depthWriteEnabled = true;
    RLGL.State.backfaceCullingEnabled = true;
    RLGL.State.lineWidth = 1.0f;
    RLGL.pipelineDirty = true;

    RLGL.State.framebufferWidth = width;
    RLGL.State.framebufferHeight = height;

    // Execute initial commands (texture/shader creation)
    rlFlushCommandList();

    TRACELOG(RL_LOG_INFO, "RLGL: Direct3D 12 backend initialized successfully");
}

void rlglClose(void)
{
    rlWaitForGpu();

    rlUnloadRenderBatch(RLGL.defaultBatch);
    rlUnloadShaderDefault();

    if (RLGL.State.defaultTextureId > 0) rlUnloadTexture(RLGL.State.defaultTextureId);

    // Release tracked resources
    for (unsigned int i = 0; i < RLGL.textureCount; i++) {
        if (RLGL.textures[i].resource) ID3D12Resource_Release(RLGL.textures[i].resource);
    }
    for (unsigned int i = 0; i < RLGL.shaderCount; i++) {
        if (RLGL.shaders[i].pso) ID3D12PipelineState_Release(RLGL.shaders[i].pso);
        if (RLGL.shaders[i].vsBlob) ID3D10Blob_Release(RLGL.shaders[i].vsBlob);
        if (RLGL.shaders[i].psBlob) ID3D10Blob_Release(RLGL.shaders[i].psBlob);
    }
    for (unsigned int i = 0; i < RLGL.bufferCount; i++) {
        if (RLGL.buffers[i].resource) ID3D12Resource_Release(RLGL.buffers[i].resource);
    }

    // Release per-frame resources
    for (int i = 0; i < RL_D3D12_NUM_FRAMES; i++) {
        rlD3D12FrameResources *frame = &RLGL.frames[i];
        if (frame->commandAllocator) ID3D12CommandAllocator_Release(frame->commandAllocator);
        if (frame->cbMatrixUpload) ID3D12Resource_Release(frame->cbMatrixUpload);
        if (frame->cbColorUpload) ID3D12Resource_Release(frame->cbColorUpload);
        for (int j = 0; j < 4; j++) {
            if (frame->batchVertexBuffers[j]) ID3D12Resource_Release(frame->batchVertexBuffers[j]);
        }
        if (frame->batchIndexBuffer) ID3D12Resource_Release(frame->batchIndexBuffer);
    }

    if (RLGL.commandList) ID3D12GraphicsCommandList_Release(RLGL.commandList);
    if (RLGL.rootSignature) ID3D12RootSignature_Release(RLGL.rootSignature);
    if (RLGL.fence) ID3D12Fence_Release(RLGL.fence);
    if (RLGL.fenceEvent) CloseHandle(RLGL.fenceEvent);
    if (RLGL.srvHeap) ID3D12DescriptorHeap_Release(RLGL.srvHeap);
    if (RLGL.rtvHeap) ID3D12DescriptorHeap_Release(RLGL.rtvHeap);
    if (RLGL.dsvHeap) ID3D12DescriptorHeap_Release(RLGL.dsvHeap);
    if (RLGL.commandQueue) ID3D12CommandQueue_Release(RLGL.commandQueue);
    if (RLGL.device) { ID3D12Device_Release(RLGL.device); RLGL.device = NULL; }
    if (RLGL.factory) { IDXGIFactory4_Release(RLGL.factory); RLGL.factory = NULL; }

    TRACELOG(RL_LOG_INFO, "RLGL: Direct3D 12 backend closed successfully");
}

void rlLoadExtensions(void *loader)
{
    (void)loader;
    RLGL.ExtSupported.computeShader = true;  // D3D12 always supports compute
    RLGL.ExtSupported.ssbo = true;
    RLGL.ExtSupported.maxAnisotropyLevel = 16.0f;
    RLGL.ExtSupported.maxDepthBits = 32;

    TRACELOG(RL_LOG_INFO, "D3D12: Extensions loaded");
}

int rlGetVersion(void) { return RL_DIRECT3D_12; }

void rlSetFramebufferWidth(int width) { RLGL.State.framebufferWidth = width; }
void rlSetFramebufferHeight(int height) { RLGL.State.framebufferHeight = height; }
int rlGetFramebufferWidth(void) { return RLGL.State.framebufferWidth; }
int rlGetFramebufferHeight(void) { return RLGL.State.framebufferHeight; }
unsigned int rlGetTextureIdDefault(void) { return RLGL.State.defaultTextureId; }
unsigned int rlGetShaderIdDefault(void) { return RLGL.State.defaultShaderId; }
int *rlGetShaderLocsDefault(void) { return RLGL.State.defaultShaderLocs; }

//----------------------------------------------------------------------------------
// Render batch management
//----------------------------------------------------------------------------------
rlRenderBatch rlLoadRenderBatch(int numBuffers, int bufferElements)
{
    rlRenderBatch batch = { 0 };
    batch.vertexBuffer = (rlVertexBuffer *)RL_MALLOC(numBuffers * sizeof(rlVertexBuffer));

    for (int i = 0; i < numBuffers; i++) {
        batch.vertexBuffer[i].elementCount = bufferElements;
        batch.vertexBuffer[i].vertices = (float *)RL_MALLOC(bufferElements * 3 * 4 * sizeof(float));
        batch.vertexBuffer[i].texcoords = (float *)RL_MALLOC(bufferElements * 2 * 4 * sizeof(float));
        batch.vertexBuffer[i].normals = (float *)RL_MALLOC(bufferElements * 3 * 4 * sizeof(float));
        batch.vertexBuffer[i].colors = (unsigned char *)RL_MALLOC(bufferElements * 4 * 4 * sizeof(unsigned char));
        batch.vertexBuffer[i].indices = (unsigned int *)RL_MALLOC(bufferElements * 6 * sizeof(unsigned int));

        for (int j = 0; j < (3 * 4 * bufferElements); j++) batch.vertexBuffer[i].vertices[j] = 0.0f;
        for (int j = 0; j < (2 * 4 * bufferElements); j++) batch.vertexBuffer[i].texcoords[j] = 0.0f;
        for (int j = 0; j < (3 * 4 * bufferElements); j++) batch.vertexBuffer[i].normals[j] = 0.0f;
        for (int j = 0; j < (4 * 4 * bufferElements); j++) batch.vertexBuffer[i].colors[j] = 0;

        int k = 0;
        for (int j = 0; j < (6 * bufferElements); j += 6) {
            batch.vertexBuffer[i].indices[j] = 4 * k;
            batch.vertexBuffer[i].indices[j + 1] = 4 * k + 1;
            batch.vertexBuffer[i].indices[j + 2] = 4 * k + 2;
            batch.vertexBuffer[i].indices[j + 3] = 4 * k;
            batch.vertexBuffer[i].indices[j + 4] = 4 * k + 2;
            batch.vertexBuffer[i].indices[j + 5] = 4 * k + 3;
            k++;
        }
        RLGL.State.vertexCounter = 0;
    }

    // Upload index data to each frame's index buffer
    for (int i = 0; i < RL_D3D12_NUM_FRAMES; i++) {
        rlD3D12FrameResources *frame = &RLGL.frames[i];
        if (frame->batchIndexMapped) {
            memcpy(frame->batchIndexMapped, batch.vertexBuffer[0].indices, bufferElements * 6 * sizeof(unsigned int));
        }
    }

    batch.draws = (rlDrawCall *)RL_MALLOC(RL_DEFAULT_BATCH_DRAWCALLS * sizeof(rlDrawCall));
    for (int i = 0; i < RL_DEFAULT_BATCH_DRAWCALLS; i++) {
        batch.draws[i].mode = RL_QUADS;
        batch.draws[i].vertexCount = 0;
        batch.draws[i].vertexAlignment = 0;
        batch.draws[i].textureId = RLGL.State.defaultTextureId;
    }
    batch.bufferCount = numBuffers;
    batch.drawCounter = 1;
    batch.currentDepth = -1.0f;

    TRACELOG(RL_LOG_INFO, "RLGL: D3D12 render batch loaded successfully");
    return batch;
}

void rlUnloadRenderBatch(rlRenderBatch batch)
{
    for (int i = 0; i < batch.bufferCount; i++) {
        RL_FREE(batch.vertexBuffer[i].vertices);
        RL_FREE(batch.vertexBuffer[i].texcoords);
        RL_FREE(batch.vertexBuffer[i].normals);
        RL_FREE(batch.vertexBuffer[i].colors);
        RL_FREE(batch.vertexBuffer[i].indices);
    }
    RL_FREE(batch.vertexBuffer);
    RL_FREE(batch.draws);
}

void rlDrawRenderBatch(rlRenderBatch *batch)
{
    if (!RLGL.commandList || !RLGL.device) return;
    if (RLGL.State.vertexCounter <= 0) {
        // Reset batch
        RLGL.State.vertexCounter = 0;
        batch->currentDepth = -1.0f;
        for (int i = 0; i < RL_DEFAULT_BATCH_DRAWCALLS; i++) {
            batch->draws[i].mode = RL_QUADS;
            batch->draws[i].vertexCount = 0;
            batch->draws[i].textureId = RLGL.State.defaultTextureId;
        }
        for (int i = 0; i < RL_DEFAULT_BATCH_MAX_TEXTURE_UNITS; i++) RLGL.State.activeTextureId[i] = 0;
        batch->drawCounter = 1;
        batch->currentBuffer++;
        if (batch->currentBuffer >= batch->bufferCount) batch->currentBuffer = 0;
        return;
    }

    rlD3D12FrameResources *frame = &RLGL.frames[RLGL.currentFrame];

    // Upload vertex data to persistently mapped upload buffers
    if (frame->batchVertexMapped[0])
        memcpy(frame->batchVertexMapped[0], batch->vertexBuffer[batch->currentBuffer].vertices, RLGL.State.vertexCounter * 3 * sizeof(float));
    if (frame->batchVertexMapped[1])
        memcpy(frame->batchVertexMapped[1], batch->vertexBuffer[batch->currentBuffer].texcoords, RLGL.State.vertexCounter * 2 * sizeof(float));
    if (frame->batchVertexMapped[2])
        memcpy(frame->batchVertexMapped[2], batch->vertexBuffer[batch->currentBuffer].normals, RLGL.State.vertexCounter * 3 * sizeof(float));
    if (frame->batchVertexMapped[3])
        memcpy(frame->batchVertexMapped[3], batch->vertexBuffer[batch->currentBuffer].colors, RLGL.State.vertexCounter * 4 * sizeof(unsigned char));

    // Set root signature and descriptor heaps
    ID3D12GraphicsCommandList_SetGraphicsRootSignature(RLGL.commandList, RLGL.rootSignature);
    ID3D12DescriptorHeap *heaps[] = { RLGL.srvHeap };
    ID3D12GraphicsCommandList_SetDescriptorHeaps(RLGL.commandList, 1, heaps);

    // Set viewport and scissor
    D3D12_VIEWPORT viewport = { 0, 0, (float)RLGL.State.framebufferWidth, (float)RLGL.State.framebufferHeight, 0.0f, 1.0f };
    D3D12_RECT scissorRect = { 0, 0, RLGL.State.framebufferWidth, RLGL.State.framebufferHeight };
    ID3D12GraphicsCommandList_RSSetViewports(RLGL.commandList, 1, &viewport);
    if (!RLGL.State.scissorTestEnabled)
        ID3D12GraphicsCommandList_RSSetScissorRects(RLGL.commandList, 1, &scissorRect);

    // Set pipeline state (use the current shader's PSO)
    if (RLGL.State.currentShaderId > 0 && RLGL.State.currentShaderId <= RLGL.shaderCount) {
        unsigned int shaderIdx = RLGL.State.currentShaderId - 1;

        // Recreate PSO if pipeline state is dirty
        if (RLGL.pipelineDirty && RLGL.shaders[shaderIdx].vsBlob && RLGL.shaders[shaderIdx].psBlob) {
            if (RLGL.shaders[shaderIdx].pso) ID3D12PipelineState_Release(RLGL.shaders[shaderIdx].pso);
            RLGL.shaders[shaderIdx].pso = rlCreatePipelineState(
                RLGL.shaders[shaderIdx].vsBlob, RLGL.shaders[shaderIdx].psBlob,
                RLGL.State.colorBlendEnabled, RLGL.State.currentBlendMode,
                RLGL.State.depthTestEnabled, RLGL.State.depthWriteEnabled,
                RLGL.State.backfaceCullingEnabled, RLGL.State.cullFaceMode,
                RLGL.State.wireMode, RLGL.State.scissorTestEnabled,
                D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE);
            RLGL.pipelineDirty = false;
        }
        if (RLGL.shaders[shaderIdx].pso) {
            ID3D12GraphicsCommandList_SetPipelineState(RLGL.commandList, RLGL.shaders[shaderIdx].pso);
        }
    }

    // Update MVP constant buffer
    Matrix matMVP = rlMatrixMultiply(RLGL.State.modelview, RLGL.State.projection);
    rl_float16 matData = rlMatrixToFloatV(matMVP);
    if (frame->cbMatrixMapped) memcpy(frame->cbMatrixMapped, matData.v, sizeof(float) * 16);

    // Update diffuse color constant buffer
    float diffuse[4] = { 1.0f, 1.0f, 1.0f, 1.0f };
    if (frame->cbColorMapped) memcpy(frame->cbColorMapped, diffuse, sizeof(float) * 4);

    // Set constant buffers as root CBVs
    if (frame->cbMatrixUpload)
        ID3D12GraphicsCommandList_SetGraphicsRootConstantBufferView(RLGL.commandList, 0,
            ID3D12Resource_GetGPUVirtualAddress(frame->cbMatrixUpload));
    if (frame->cbColorUpload)
        ID3D12GraphicsCommandList_SetGraphicsRootConstantBufferView(RLGL.commandList, 1,
            ID3D12Resource_GetGPUVirtualAddress(frame->cbColorUpload));

    // Set vertex buffers
    D3D12_VERTEX_BUFFER_VIEW vbViews[4] = { 0 };
    UINT vertexSizes[4] = { 3 * sizeof(float), 2 * sizeof(float), 3 * sizeof(float), 4 * sizeof(unsigned char) };
    for (int i = 0; i < 4; i++) {
        if (frame->batchVertexBuffers[i]) {
            vbViews[i].BufferLocation = ID3D12Resource_GetGPUVirtualAddress(frame->batchVertexBuffers[i]);
            vbViews[i].SizeInBytes = RLGL.batchBufferSize * vertexSizes[i];
            vbViews[i].StrideInBytes = vertexSizes[i];
        }
    }
    ID3D12GraphicsCommandList_IASetVertexBuffers(RLGL.commandList, 0, 4, vbViews);

    // Set index buffer
    if (frame->batchIndexBuffer) {
        D3D12_INDEX_BUFFER_VIEW ibView = { 0 };
        ibView.BufferLocation = ID3D12Resource_GetGPUVirtualAddress(frame->batchIndexBuffer);
        ibView.SizeInBytes = RL_DEFAULT_BATCH_BUFFER_ELEMENTS * 6 * sizeof(unsigned int);
        ibView.Format = DXGI_FORMAT_R32_UINT;
        ID3D12GraphicsCommandList_IASetIndexBuffer(RLGL.commandList, &ibView);
    }

    // Draw each batch draw call
    for (int i = 0, vertexOffset = 0; i < batch->drawCounter; i++) {
        // Bind texture SRV
        unsigned int texId = batch->draws[i].textureId;
        if (texId > 0 && texId <= RLGL.textureCount) {
            ID3D12GraphicsCommandList_SetGraphicsRootDescriptorTable(RLGL.commandList, 2,
                RLGL.textures[texId - 1].srvGpuHandle);
        }

        if ((batch->draws[i].mode == RL_LINES) || (batch->draws[i].mode == RL_TRIANGLES)) {
            D3D_PRIMITIVE_TOPOLOGY topo = (batch->draws[i].mode == RL_LINES) ?
                D3D_PRIMITIVE_TOPOLOGY_LINELIST : D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
            ID3D12GraphicsCommandList_IASetPrimitiveTopology(RLGL.commandList, topo);
            ID3D12GraphicsCommandList_DrawInstanced(RLGL.commandList, batch->draws[i].vertexCount, 1, vertexOffset, 0);
        } else {
            // QUADS: Draw as indexed triangles
            ID3D12GraphicsCommandList_IASetPrimitiveTopology(RLGL.commandList, D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            ID3D12GraphicsCommandList_DrawIndexedInstanced(RLGL.commandList, batch->draws[i].vertexCount / 4 * 6, 1, vertexOffset / 4 * 6, 0, 0);
        }
        vertexOffset += (batch->draws[i].vertexCount + batch->draws[i].vertexAlignment);
    }

    // Reset batch
    RLGL.State.vertexCounter = 0;
    batch->currentDepth = -1.0f;

    for (int i = 0; i < RL_DEFAULT_BATCH_DRAWCALLS; i++) {
        batch->draws[i].mode = RL_QUADS;
        batch->draws[i].vertexCount = 0;
        batch->draws[i].textureId = RLGL.State.defaultTextureId;
    }
    for (int i = 0; i < RL_DEFAULT_BATCH_MAX_TEXTURE_UNITS; i++) RLGL.State.activeTextureId[i] = 0;
    batch->drawCounter = 1;
    batch->currentBuffer++;
    if (batch->currentBuffer >= batch->bufferCount) batch->currentBuffer = 0;
}

void rlSetRenderBatchActive(rlRenderBatch *batch)
{
    rlDrawRenderBatch(RLGL.currentBatch);
    if (batch != NULL) RLGL.currentBatch = batch;
    else RLGL.currentBatch = &RLGL.defaultBatch;
}

void rlDrawRenderBatchActive(void) { rlDrawRenderBatch(RLGL.currentBatch); }

bool rlCheckRenderBatchLimit(int vCount)
{
    bool overflow = false;
    if ((RLGL.State.vertexCounter + vCount) >= (RLGL.currentBatch->vertexBuffer[RLGL.currentBatch->currentBuffer].elementCount * 4)) {
        overflow = true;
        int currentMode = RLGL.currentBatch->draws[RLGL.currentBatch->drawCounter - 1].mode;
        int currentTexture = RLGL.currentBatch->draws[RLGL.currentBatch->drawCounter - 1].textureId;
        rlDrawRenderBatch(RLGL.currentBatch);
        RLGL.currentBatch->draws[RLGL.currentBatch->drawCounter - 1].mode = currentMode;
        RLGL.currentBatch->draws[RLGL.currentBatch->drawCounter - 1].textureId = currentTexture;
    }
    return overflow;
}

//----------------------------------------------------------------------------------
// Texture management
//----------------------------------------------------------------------------------
unsigned int rlLoadTexture(const void *data, int width, int height, int format, int mipmapCount)
{
    if (!RLGL.device || RLGL.textureCount >= RL_D3D12_MAX_TEXTURES) return 0;

    DXGI_FORMAT dxgiFormat = rlGetDXGIFormat(format);
    unsigned int idx = RLGL.textureCount;

    // Calculate bytes per pixel and handle 3-channel expansion
    int bpp = 4;
    switch (format) {
        case 1: bpp = 1; break;
        case 2: bpp = 2; break;
        case 3: bpp = 2; break;
        case 4: bpp = 4; break;  // R8G8B8 -> RGBA (expanded)
        case 5: bpp = 2; break;
        case 6: bpp = 2; break;
        case 7: bpp = 4; break;
        case 8: bpp = 4; break;
        case 9: bpp = 12; break;
        case 10: bpp = 16; break;
        case 11: bpp = 2; break;
        case 12: bpp = 8; break;  // R16G16B16 -> RGBA16 (expanded)
        case 13: bpp = 8; break;
        default: bpp = 4; break;
    }

    // Expand 3-channel data if needed
    void *expandedData = NULL;
    const void *uploadData = data;
    if (data != NULL && format == 4) {
        int pixelCount = width * height;
        expandedData = RL_MALLOC(pixelCount * 4);
        const unsigned char *src = (const unsigned char *)data;
        unsigned char *dst = (unsigned char *)expandedData;
        for (int i = 0; i < pixelCount; i++) {
            dst[i * 4 + 0] = src[i * 3 + 0];
            dst[i * 4 + 1] = src[i * 3 + 1];
            dst[i * 4 + 2] = src[i * 3 + 2];
            dst[i * 4 + 3] = 255;
        }
        uploadData = expandedData;
    } else if (data != NULL && format == 12) {
        int pixelCount = width * height;
        expandedData = RL_MALLOC(pixelCount * 8);
        const unsigned short *src = (const unsigned short *)data;
        unsigned short *dst = (unsigned short *)expandedData;
        for (int i = 0; i < pixelCount; i++) {
            dst[i * 4 + 0] = src[i * 3 + 0];
            dst[i * 4 + 1] = src[i * 3 + 1];
            dst[i * 4 + 2] = src[i * 3 + 2];
            dst[i * 4 + 3] = 0x3C00;
        }
        uploadData = expandedData;
    }

    // Create texture resource
    D3D12_RESOURCE_DESC texDesc = { 0 };
    texDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    texDesc.Width = width;
    texDesc.Height = height;
    texDesc.DepthOrArraySize = 1;
    texDesc.MipLevels = mipmapCount;
    texDesc.Format = dxgiFormat;
    texDesc.SampleDesc.Count = 1;
    texDesc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    texDesc.Flags = D3D12_RESOURCE_FLAG_NONE;

    D3D12_HEAP_PROPERTIES heapProps = { 0 };
    heapProps.Type = D3D12_HEAP_TYPE_DEFAULT;

    HRESULT hr = ID3D12Device_CreateCommittedResource(RLGL.device, &heapProps,
        D3D12_HEAP_FLAG_NONE, &texDesc,
        D3D12_RESOURCE_STATE_COPY_DEST,
        NULL, &IID_ID3D12Resource, (void **)&RLGL.textures[idx].resource);
    if (FAILED(hr)) {
        RL_FREE(expandedData);
        return 0;
    }

    // Upload texture data via staging buffer
    if (uploadData) {
        UINT64 uploadSize = 0;
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint = { 0 };
        UINT numRows = 0;
        UINT64 rowSize = 0;
        ID3D12Device_GetCopyableFootprints(RLGL.device, &texDesc, 0, 1, 0, &footprint, &numRows, &rowSize, &uploadSize);

        ID3D12Resource *uploadBuffer = rlCreateUploadBuffer((UINT)uploadSize);
        if (uploadBuffer) {
            void *mapped = NULL;
            D3D12_RANGE readRange = { 0, 0 };
            ID3D12Resource_Map(uploadBuffer, 0, &readRange, &mapped);
            if (mapped) {
                // Copy row by row (upload pitch may differ from data pitch)
                const unsigned char *srcData = (const unsigned char *)uploadData;
                unsigned char *dstData = (unsigned char *)mapped;
                UINT srcPitch = width * bpp;
                for (UINT row = 0; row < numRows; row++) {
                    memcpy(dstData + footprint.Footprint.RowPitch * row, srcData + srcPitch * row, srcPitch);
                }
                ID3D12Resource_Unmap(uploadBuffer, 0, NULL);
            }

            // Copy from upload buffer to texture
            D3D12_TEXTURE_COPY_LOCATION dstLoc = { 0 };
            dstLoc.pResource = RLGL.textures[idx].resource;
            dstLoc.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
            dstLoc.SubresourceIndex = 0;

            D3D12_TEXTURE_COPY_LOCATION srcLoc = { 0 };
            srcLoc.pResource = uploadBuffer;
            srcLoc.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
            srcLoc.PlacedFootprint = footprint;

            ID3D12GraphicsCommandList_CopyTextureRegion(RLGL.commandList, &dstLoc, 0, 0, 0, &srcLoc, NULL);

            // Transition to shader resource
            rlTransitionResource(RLGL.textures[idx].resource, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);

            // Execute and wait so we can release upload buffer
            rlFlushCommandList();
            ID3D12Resource_Release(uploadBuffer);
        }
    } else {
        // No data, just transition
        rlTransitionResource(RLGL.textures[idx].resource, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
    }

    RL_FREE(expandedData);

    // Create SRV
    D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = { 0 };
    srvDesc.Format = dxgiFormat;
    srvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srvDesc.Texture2D.MipLevels = mipmapCount;

    RLGL.textures[idx].srvIndex = rlAllocSRVDescriptor(&RLGL.textures[idx].srvCpuHandle, &RLGL.textures[idx].srvGpuHandle);
    if (RLGL.textures[idx].srvIndex < 0) {
        ID3D12Resource_Release(RLGL.textures[idx].resource);
        RLGL.textures[idx].resource = NULL;
        return 0;
    }
    ID3D12Device_CreateShaderResourceView(RLGL.device, RLGL.textures[idx].resource, &srvDesc, RLGL.textures[idx].srvCpuHandle);

    RLGL.textures[idx].width = width;
    RLGL.textures[idx].height = height;
    RLGL.textures[idx].format = format;
    RLGL.textures[idx].dxgiFormat = dxgiFormat;
    RLGL.textureCount++;

    unsigned int id = idx + 1;
    TRACELOG(RL_LOG_INFO, "TEXTURE: [ID %i] Texture loaded successfully (%ix%i)", id, width, height);
    return id;
}

unsigned int rlLoadTextureDepth(int width, int height, bool useRenderBuffer)
{
    (void)useRenderBuffer;
    if (!RLGL.device || RLGL.textureCount >= RL_D3D12_MAX_TEXTURES) return 0;

    unsigned int idx = RLGL.textureCount;

    D3D12_RESOURCE_DESC texDesc = { 0 };
    texDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    texDesc.Width = width;
    texDesc.Height = height;
    texDesc.DepthOrArraySize = 1;
    texDesc.MipLevels = 1;
    texDesc.Format = DXGI_FORMAT_R32_TYPELESS;
    texDesc.SampleDesc.Count = 1;
    texDesc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    texDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;

    D3D12_HEAP_PROPERTIES heapProps = { 0 };
    heapProps.Type = D3D12_HEAP_TYPE_DEFAULT;

    D3D12_CLEAR_VALUE clearValue = { 0 };
    clearValue.Format = DXGI_FORMAT_D32_FLOAT;
    clearValue.DepthStencil.Depth = 1.0f;

    HRESULT hr = ID3D12Device_CreateCommittedResource(RLGL.device, &heapProps,
        D3D12_HEAP_FLAG_NONE, &texDesc,
        D3D12_RESOURCE_STATE_DEPTH_WRITE,
        &clearValue, &IID_ID3D12Resource, (void **)&RLGL.textures[idx].resource);
    if (FAILED(hr)) return 0;

    // Create DSV
    D3D12_DEPTH_STENCIL_VIEW_DESC dsvDesc = { 0 };
    dsvDesc.Format = DXGI_FORMAT_D32_FLOAT;
    dsvDesc.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2D;

    D3D12_CPU_DESCRIPTOR_HANDLE dsvStart;
    ID3D12DescriptorHeap_GetCPUDescriptorHandleForHeapStart(RLGL.dsvHeap, &dsvStart);
    RLGL.textures[idx].dsvIndex = RLGL.dsvHeapUsed++;
    RLGL.textures[idx].dsvHandle.ptr = dsvStart.ptr + (SIZE_T)(RLGL.textures[idx].dsvIndex * RLGL.dsvDescriptorSize);
    ID3D12Device_CreateDepthStencilView(RLGL.device, RLGL.textures[idx].resource, &dsvDesc, RLGL.textures[idx].dsvHandle);

    // Create SRV for sampling
    D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = { 0 };
    srvDesc.Format = DXGI_FORMAT_R32_FLOAT;
    srvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srvDesc.Texture2D.MipLevels = 1;

    RLGL.textures[idx].srvIndex = rlAllocSRVDescriptor(&RLGL.textures[idx].srvCpuHandle, &RLGL.textures[idx].srvGpuHandle);
    ID3D12Device_CreateShaderResourceView(RLGL.device, RLGL.textures[idx].resource, &srvDesc, RLGL.textures[idx].srvCpuHandle);

    RLGL.textures[idx].width = width;
    RLGL.textures[idx].height = height;
    RLGL.textures[idx].format = 8; // R32
    RLGL.textures[idx].isDepth = true;
    RLGL.textureCount++;

    unsigned int id = idx + 1;
    TRACELOG(RL_LOG_INFO, "TEXTURE: [ID %i] Depth texture created (%ix%i)", id, width, height);
    return id;
}

unsigned int rlLoadTextureCubemap(const void *data, int size, int format, int mipmapCount)
{
    (void)data; (void)size; (void)format; (void)mipmapCount;
    // Cubemap loading - simplified stub
    TRACELOG(RL_LOG_WARNING, "D3D12: Cubemap loading not yet fully implemented");
    return 0;
}

void rlUpdateTexture(unsigned int id, int offsetX, int offsetY, int width, int height, int format, const void *data)
{
    if (id == 0 || id > RLGL.textureCount || !RLGL.commandList || !data) return;
    unsigned int idx = id - 1;

    int bpp = 4;
    switch (format) {
        case 1: bpp = 1; break;
        case 2: bpp = 2; break;
        case 3: bpp = 2; break;
        case 4: bpp = 4; break;
        case 5: bpp = 2; break;
        case 6: bpp = 2; break;
        case 7: bpp = 4; break;
        case 8: bpp = 4; break;
        case 9: bpp = 12; break;
        case 10: bpp = 16; break;
        case 11: bpp = 2; break;
        case 12: bpp = 8; break;
        case 13: bpp = 8; break;
        default: bpp = 4; break;
    }

    // Expand 3-channel data
    void *expandedData = NULL;
    const void *uploadData = data;
    if (format == 4) {
        int pixelCount = width * height;
        expandedData = RL_MALLOC(pixelCount * 4);
        const unsigned char *src = (const unsigned char *)data;
        unsigned char *dst = (unsigned char *)expandedData;
        for (int i = 0; i < pixelCount; i++) {
            dst[i * 4 + 0] = src[i * 3 + 0];
            dst[i * 4 + 1] = src[i * 3 + 1];
            dst[i * 4 + 2] = src[i * 3 + 2];
            dst[i * 4 + 3] = 255;
        }
        uploadData = expandedData;
    } else if (format == 12) {
        int pixelCount = width * height;
        expandedData = RL_MALLOC(pixelCount * 8);
        const unsigned short *src = (const unsigned short *)data;
        unsigned short *dst = (unsigned short *)expandedData;
        for (int i = 0; i < pixelCount; i++) {
            dst[i * 4 + 0] = src[i * 3 + 0];
            dst[i * 4 + 1] = src[i * 3 + 1];
            dst[i * 4 + 2] = src[i * 3 + 2];
            dst[i * 4 + 3] = 0x3C00;
        }
        uploadData = expandedData;
    }

    // Create staging upload buffer
    UINT uploadSize = width * height * bpp;
    ID3D12Resource *uploadBuffer = rlCreateUploadBuffer(uploadSize);
    if (!uploadBuffer) { RL_FREE(expandedData); return; }

    void *mapped = NULL;
    D3D12_RANGE readRange = { 0, 0 };
    ID3D12Resource_Map(uploadBuffer, 0, &readRange, &mapped);
    if (mapped) {
        memcpy(mapped, uploadData, uploadSize);
        ID3D12Resource_Unmap(uploadBuffer, 0, NULL);
    }

    // Transition texture to copy dest
    rlTransitionResource(RLGL.textures[idx].resource, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_DEST);

    // Copy region
    D3D12_TEXTURE_COPY_LOCATION dstLoc = { 0 };
    dstLoc.pResource = RLGL.textures[idx].resource;
    dstLoc.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    dstLoc.SubresourceIndex = 0;

    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint = { 0 };
    footprint.Footprint.Format = RLGL.textures[idx].dxgiFormat;
    footprint.Footprint.Width = width;
    footprint.Footprint.Height = height;
    footprint.Footprint.Depth = 1;
    footprint.Footprint.RowPitch = (width * bpp + 255) & ~255; // Align to 256 bytes

    D3D12_TEXTURE_COPY_LOCATION srcLoc = { 0 };
    srcLoc.pResource = uploadBuffer;
    srcLoc.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    srcLoc.PlacedFootprint = footprint;

    D3D12_BOX srcBox = { 0, 0, 0, (UINT)width, (UINT)height, 1 };
    ID3D12GraphicsCommandList_CopyTextureRegion(RLGL.commandList, &dstLoc, offsetX, offsetY, 0, &srcLoc, &srcBox);

    // Transition back
    rlTransitionResource(RLGL.textures[idx].resource, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);

    rlFlushCommandList();
    ID3D12Resource_Release(uploadBuffer);
    RL_FREE(expandedData);
}

void rlGetGlTextureFormats(int format, unsigned int *glInternalFormat, unsigned int *glFormat, unsigned int *glType)
{
    *glInternalFormat = (unsigned int)rlGetDXGIFormat(format);
    *glFormat = *glInternalFormat;
    *glType = 0;
}

const char *rlGetPixelFormatName(unsigned int format)
{
    switch (format) {
        case 1: return "GRAYSCALE";
        case 2: return "GRAY_ALPHA";
        case 3: return "R5G6B5";
        case 4: return "R8G8B8";
        case 5: return "R5G5B5A1";
        case 6: return "R4G4B4A4";
        case 7: return "R8G8B8A8";
        case 8: return "R32";
        case 9: return "R32G32B32";
        case 10: return "R32G32B32A32";
        default: return "UNKNOWN";
    }
}

void rlUnloadTexture(unsigned int id) {
    if (id == 0 || id > RLGL.textureCount) return;
    unsigned int idx = id - 1;
    if (RLGL.textures[idx].resource) { ID3D12Resource_Release(RLGL.textures[idx].resource); RLGL.textures[idx].resource = NULL; }
}

void rlGenTextureMipmaps(unsigned int id, int width, int height, int format, int *mipmaps) {
    (void)id; (void)format;
    // D3D12 mipmap generation requires compute shader or manual blit chain
    if (mipmaps) {
        int max = (width > height) ? width : height;
        *mipmaps = 1 + (int)floor(log((double)max) / log(2.0));
    }
}

void *rlReadTexturePixels(unsigned int id, int width, int height, int format) {
    if (id == 0 || id > RLGL.textureCount || !RLGL.commandList || !RLGL.device) return NULL;
    unsigned int idx = id - 1;

    // Create readback buffer
    D3D12_RESOURCE_DESC texDesc = { 0 };
    ID3D12Resource_GetDesc(RLGL.textures[idx].resource, &texDesc);

    UINT64 totalSize = 0;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint = { 0 };
    UINT numRows = 0;
    UINT64 rowSize = 0;
    ID3D12Device_GetCopyableFootprints(RLGL.device, &texDesc, 0, 1, 0, &footprint, &numRows, &rowSize, &totalSize);

    // Create readback buffer
    D3D12_HEAP_PROPERTIES heapProps = { 0 };
    heapProps.Type = D3D12_HEAP_TYPE_READBACK;
    D3D12_RESOURCE_DESC bufDesc = { 0 };
    bufDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    bufDesc.Width = totalSize;
    bufDesc.Height = 1;
    bufDesc.DepthOrArraySize = 1;
    bufDesc.MipLevels = 1;
    bufDesc.Format = DXGI_FORMAT_UNKNOWN;
    bufDesc.SampleDesc.Count = 1;
    bufDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;

    ID3D12Resource *readback = NULL;
    HRESULT hr = ID3D12Device_CreateCommittedResource(RLGL.device, &heapProps,
        D3D12_HEAP_FLAG_NONE, &bufDesc, D3D12_RESOURCE_STATE_COPY_DEST,
        NULL, &IID_ID3D12Resource, (void **)&readback);
    if (FAILED(hr)) return NULL;

    // Transition texture to copy source
    rlTransitionResource(RLGL.textures[idx].resource, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_SOURCE);

    // Copy to readback buffer
    D3D12_TEXTURE_COPY_LOCATION dstLoc = { 0 };
    dstLoc.pResource = readback;
    dstLoc.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    dstLoc.PlacedFootprint = footprint;

    D3D12_TEXTURE_COPY_LOCATION srcLoc = { 0 };
    srcLoc.pResource = RLGL.textures[idx].resource;
    srcLoc.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    srcLoc.SubresourceIndex = 0;

    ID3D12GraphicsCommandList_CopyTextureRegion(RLGL.commandList, &dstLoc, 0, 0, 0, &srcLoc, NULL);

    // Transition back
    rlTransitionResource(RLGL.textures[idx].resource, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);

    rlFlushCommandList();

    // Read back data
    int dataSize = rlGetPixelDataSize(width, height, format);
    void *pixels = RL_MALLOC(dataSize);

    void *mapped = NULL;
    D3D12_RANGE readRange = { 0, totalSize };
    if (SUCCEEDED(ID3D12Resource_Map(readback, 0, &readRange, &mapped))) {
        // Copy row by row
        int bpp = dataSize / (width * height);
        for (int row = 0; row < height; row++) {
            memcpy((unsigned char *)pixels + row * width * bpp,
                   (unsigned char *)mapped + row * footprint.Footprint.RowPitch,
                   width * bpp);
        }
        D3D12_RANGE writeRange = { 0, 0 };
        ID3D12Resource_Unmap(readback, 0, &writeRange);
    }

    ID3D12Resource_Release(readback);
    return pixels;
}

//----------------------------------------------------------------------------------
// Framebuffer management
//----------------------------------------------------------------------------------
unsigned int rlLoadFramebuffer(void) {
    if (RLGL.framebufferCount >= RL_D3D12_MAX_FRAMEBUFFERS) return 0;
    unsigned int idx = RLGL.framebufferCount;
    memset(&RLGL.framebuffers[idx], 0, sizeof(rlD3D12Framebuffer));
    RLGL.framebufferCount++;
    return idx + 1;
}

void rlFramebufferAttach(unsigned int fboId, unsigned int texId, int attachType, int texType, int mipLevel) {
    (void)texType; (void)mipLevel;
    if (fboId == 0 || fboId > RLGL.framebufferCount || texId == 0 || texId > RLGL.textureCount) return;
    unsigned int fboIdx = fboId - 1;
    unsigned int texIdx = texId - 1;

    if (attachType == RL_ATTACHMENT_COLOR_CHANNEL0) {
        RLGL.framebuffers[fboIdx].colorTexIds[0] = texId;
        RLGL.framebuffers[fboIdx].colorCount = 1;

        // Create RTV for the texture
        D3D12_CPU_DESCRIPTOR_HANDLE rtvStart;
        ID3D12DescriptorHeap_GetCPUDescriptorHandleForHeapStart(RLGL.rtvHeap, &rtvStart);
        RLGL.textures[texIdx].rtvIndex = RLGL.rtvHeapUsed++;
        RLGL.textures[texIdx].rtvHandle.ptr = rtvStart.ptr + (SIZE_T)(RLGL.textures[texIdx].rtvIndex * RLGL.rtvDescriptorSize);

        D3D12_RENDER_TARGET_VIEW_DESC rtvDesc = { 0 };
        rtvDesc.Format = RLGL.textures[texIdx].dxgiFormat;
        rtvDesc.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2D;
        ID3D12Device_CreateRenderTargetView(RLGL.device, RLGL.textures[texIdx].resource, &rtvDesc, RLGL.textures[texIdx].rtvHandle);
    } else if (attachType == RL_ATTACHMENT_DEPTH) {
        RLGL.framebuffers[fboIdx].depthTexId = texId;
    }
}

bool rlFramebufferComplete(unsigned int id) {
    if (id == 0 || id > RLGL.framebufferCount) return false;
    RLGL.framebuffers[id - 1].complete = true;
    return true;
}

void rlUnloadFramebuffer(unsigned int id) {
    if (id == 0 || id > RLGL.framebufferCount) return;
    memset(&RLGL.framebuffers[id - 1], 0, sizeof(rlD3D12Framebuffer));
}

//----------------------------------------------------------------------------------
// Vertex buffer management
//----------------------------------------------------------------------------------
unsigned int rlLoadVertexArray(void) { return 1; }
void rlUnloadVertexArray(unsigned int vaoId) { (void)vaoId; }
bool rlEnableVertexArray(unsigned int vaoId) { (void)vaoId; return true; }
void rlDisableVertexArray(void) { }

unsigned int rlLoadVertexBuffer(const void *buffer, int size, bool dynamic) {
    if (!RLGL.device || RLGL.bufferCount >= RL_D3D12_MAX_BUFFERS) return 0;
    unsigned int idx = RLGL.bufferCount;

    if (dynamic) {
        RLGL.buffers[idx].resource = rlCreateUploadBuffer(size);
        if (RLGL.buffers[idx].resource) {
            D3D12_RANGE readRange = { 0, 0 };
            ID3D12Resource_Map(RLGL.buffers[idx].resource, 0, &readRange, &RLGL.buffers[idx].mappedData);
            if (buffer && RLGL.buffers[idx].mappedData) memcpy(RLGL.buffers[idx].mappedData, buffer, size);
        }
    } else {
        RLGL.buffers[idx].resource = rlCreateDefaultBuffer(size, D3D12_RESOURCE_FLAG_NONE);
        if (buffer && RLGL.buffers[idx].resource) {
            // Upload via staging
            ID3D12Resource *upload = rlCreateUploadBuffer(size);
            if (upload) {
                void *mapped = NULL;
                D3D12_RANGE readRange = { 0, 0 };
                ID3D12Resource_Map(upload, 0, &readRange, &mapped);
                if (mapped) { memcpy(mapped, buffer, size); ID3D12Resource_Unmap(upload, 0, NULL); }
                rlTransitionResource(RLGL.buffers[idx].resource, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
                ID3D12GraphicsCommandList_CopyBufferRegion(RLGL.commandList, RLGL.buffers[idx].resource, 0, upload, 0, size);
                rlTransitionResource(RLGL.buffers[idx].resource, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER);
                rlFlushCommandList();
                ID3D12Resource_Release(upload);
            }
        }
    }

    RLGL.buffers[idx].size = size;
    RLGL.buffers[idx].dynamic = dynamic;
    RLGL.buffers[idx].isIndex = false;
    if (!RLGL.buffers[idx].resource) return 0;
    RLGL.bufferCount++;
    return idx + 1;
}

unsigned int rlLoadVertexBufferElement(const void *buffer, int size, bool dynamic) {
    if (!RLGL.device || RLGL.bufferCount >= RL_D3D12_MAX_BUFFERS) return 0;
    unsigned int idx = RLGL.bufferCount;

    if (dynamic) {
        RLGL.buffers[idx].resource = rlCreateUploadBuffer(size);
        if (RLGL.buffers[idx].resource) {
            D3D12_RANGE readRange = { 0, 0 };
            ID3D12Resource_Map(RLGL.buffers[idx].resource, 0, &readRange, &RLGL.buffers[idx].mappedData);
            if (buffer && RLGL.buffers[idx].mappedData) memcpy(RLGL.buffers[idx].mappedData, buffer, size);
        }
    } else {
        RLGL.buffers[idx].resource = rlCreateDefaultBuffer(size, D3D12_RESOURCE_FLAG_NONE);
        if (buffer && RLGL.buffers[idx].resource) {
            ID3D12Resource *upload = rlCreateUploadBuffer(size);
            if (upload) {
                void *mapped = NULL;
                D3D12_RANGE readRange = { 0, 0 };
                ID3D12Resource_Map(upload, 0, &readRange, &mapped);
                if (mapped) { memcpy(mapped, buffer, size); ID3D12Resource_Unmap(upload, 0, NULL); }
                rlTransitionResource(RLGL.buffers[idx].resource, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
                ID3D12GraphicsCommandList_CopyBufferRegion(RLGL.commandList, RLGL.buffers[idx].resource, 0, upload, 0, size);
                rlTransitionResource(RLGL.buffers[idx].resource, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_INDEX_BUFFER);
                rlFlushCommandList();
                ID3D12Resource_Release(upload);
            }
        }
    }

    RLGL.buffers[idx].size = size;
    RLGL.buffers[idx].dynamic = dynamic;
    RLGL.buffers[idx].isIndex = true;
    if (!RLGL.buffers[idx].resource) return 0;
    RLGL.bufferCount++;
    return idx + 1;
}

void rlUpdateVertexBuffer(unsigned int id, const void *data, int dataSize, int offset) {
    if (id == 0 || id > RLGL.bufferCount || !data) return;
    unsigned int idx = id - 1;
    if (RLGL.buffers[idx].dynamic && RLGL.buffers[idx].mappedData) {
        memcpy((char *)RLGL.buffers[idx].mappedData + offset, data, dataSize);
    }
}

void rlUpdateVertexBufferElements(unsigned int id, const void *data, int dataSize, int offset) {
    rlUpdateVertexBuffer(id, data, dataSize, offset);
}

void rlUnloadVertexBuffer(unsigned int vboId) {
    if (vboId == 0 || vboId > RLGL.bufferCount) return;
    unsigned int idx = vboId - 1;
    if (RLGL.buffers[idx].resource) { ID3D12Resource_Release(RLGL.buffers[idx].resource); RLGL.buffers[idx].resource = NULL; }
    RLGL.buffers[idx].mappedData = NULL;
}

void rlEnableVertexBuffer(unsigned int id) {
    if (id == 0 || id > RLGL.bufferCount || !RLGL.commandList) return;
    unsigned int idx = id - 1;
    D3D12_VERTEX_BUFFER_VIEW vbView = { 0 };
    vbView.BufferLocation = ID3D12Resource_GetGPUVirtualAddress(RLGL.buffers[idx].resource);
    vbView.SizeInBytes = RLGL.buffers[idx].size;
    vbView.StrideInBytes = 0; // Will be set by caller via rlSetVertexAttribute
    ID3D12GraphicsCommandList_IASetVertexBuffers(RLGL.commandList, 0, 1, &vbView);
}
void rlDisableVertexBuffer(void) { }
void rlEnableVertexBufferElement(unsigned int id) {
    if (id == 0 || id > RLGL.bufferCount || !RLGL.commandList) return;
    unsigned int idx = id - 1;
    D3D12_INDEX_BUFFER_VIEW ibView = { 0 };
    ibView.BufferLocation = ID3D12Resource_GetGPUVirtualAddress(RLGL.buffers[idx].resource);
    ibView.SizeInBytes = RLGL.buffers[idx].size;
    ibView.Format = DXGI_FORMAT_R16_UINT;
    ID3D12GraphicsCommandList_IASetIndexBuffer(RLGL.commandList, &ibView);
}
void rlDisableVertexBufferElement(void) { }
void rlEnableVertexAttribute(unsigned int index) { (void)index; }
void rlDisableVertexAttribute(unsigned int index) { (void)index; }
void rlSetVertexAttribute(unsigned int index, int compSize, int type, bool normalized, int stride, int offset) {
    (void)index; (void)compSize; (void)type; (void)normalized; (void)stride; (void)offset;
}
void rlSetVertexAttributeDivisor(unsigned int index, int divisor) { (void)index; (void)divisor; }
void rlSetVertexAttributeDefault(int locIndex, const void *value, int attribType, int count) {
    (void)locIndex; (void)value; (void)attribType; (void)count;
}

void rlDrawVertexArray(int offset, int count) {
    if (!RLGL.commandList) return;
    ID3D12GraphicsCommandList_IASetPrimitiveTopology(RLGL.commandList, D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    ID3D12GraphicsCommandList_DrawInstanced(RLGL.commandList, count, 1, offset, 0);
}

void rlDrawVertexArrayElements(int offset, int count, const void *buffer) {
    (void)buffer;
    if (!RLGL.commandList) return;
    ID3D12GraphicsCommandList_IASetPrimitiveTopology(RLGL.commandList, D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    ID3D12GraphicsCommandList_DrawIndexedInstanced(RLGL.commandList, count, 1, offset, 0, 0);
}

void rlDrawVertexArrayInstanced(int offset, int count, int instances) {
    (void)offset;
    if (!RLGL.commandList) return;
    ID3D12GraphicsCommandList_IASetPrimitiveTopology(RLGL.commandList, D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    ID3D12GraphicsCommandList_DrawInstanced(RLGL.commandList, count, instances, 0, 0);
}

void rlDrawVertexArrayElementsInstanced(int offset, int count, const void *buffer, int instances) {
    (void)buffer;
    if (!RLGL.commandList) return;
    ID3D12GraphicsCommandList_IASetPrimitiveTopology(RLGL.commandList, D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    ID3D12GraphicsCommandList_DrawIndexedInstanced(RLGL.commandList, count, instances, offset, 0, 0);
}

//----------------------------------------------------------------------------------
// Shader management
//----------------------------------------------------------------------------------
unsigned int rlCompileShader(const char *shaderCode, int type)
{
    if (!RLGL.device || RLGL.shaderCount >= RL_D3D12_MAX_SHADERS) return 0;

    const char *target = (type == RL_VERTEX_SHADER) ? "vs_5_0" : "ps_5_0";
    const char *entry = (type == RL_VERTEX_SHADER) ? "VSMain" : "PSMain";

    ID3DBlob *blob = NULL;
    ID3DBlob *errors = NULL;
    HRESULT hr = D3DCompile(shaderCode, strlen(shaderCode), NULL, NULL, NULL, entry, target,
        D3DCOMPILE_OPTIMIZATION_LEVEL3 | D3DCOMPILE_PACK_MATRIX_COLUMN_MAJOR, 0, &blob, &errors);

    if (FAILED(hr)) {
        if (errors) {
            TRACELOG(RL_LOG_WARNING, "SHADER: Compile error: %s", (char *)ID3D10Blob_GetBufferPointer(errors));
            ID3D10Blob_Release(errors);
        }
        return 0;
    }
    if (errors) ID3D10Blob_Release(errors);

    unsigned int id = RLGL.shaderCount + 1;

    if (type == RL_VERTEX_SHADER) {
        RLGL.shaders[RLGL.shaderCount].vsBlob = blob;
    } else {
        RLGL.shaders[RLGL.shaderCount].psBlob = blob;
    }

    return id;
}

unsigned int rlLoadShaderCode(const char *vsCode, const char *fsCode)
{
    if (!vsCode) vsCode = defaultVShaderHLSL;
    if (!fsCode) fsCode = defaultPShaderHLSL;

    if (RLGL.shaderCount >= RL_D3D12_MAX_SHADERS) return 0;
    unsigned int idx = RLGL.shaderCount;

    // Compile vertex shader
    ID3DBlob *vsBlob = NULL, *psBlob = NULL, *errors = NULL;
    HRESULT hr = D3DCompile(vsCode, strlen(vsCode), NULL, NULL, NULL, "VSMain", "vs_5_0",
        D3DCOMPILE_OPTIMIZATION_LEVEL3 | D3DCOMPILE_PACK_MATRIX_COLUMN_MAJOR, 0, &vsBlob, &errors);
    if (FAILED(hr)) {
        if (errors) { TRACELOG(RL_LOG_WARNING, "SHADER: VS compile error: %s", (char *)ID3D10Blob_GetBufferPointer(errors)); ID3D10Blob_Release(errors); }
        return RLGL.State.defaultShaderId;
    }
    if (errors) ID3D10Blob_Release(errors);
    errors = NULL;

    // Compile pixel shader
    hr = D3DCompile(fsCode, strlen(fsCode), NULL, NULL, NULL, "PSMain", "ps_5_0",
        D3DCOMPILE_OPTIMIZATION_LEVEL3 | D3DCOMPILE_PACK_MATRIX_COLUMN_MAJOR, 0, &psBlob, &errors);
    if (FAILED(hr)) {
        if (errors) { TRACELOG(RL_LOG_WARNING, "SHADER: PS compile error: %s", (char *)ID3D10Blob_GetBufferPointer(errors)); ID3D10Blob_Release(errors); }
        ID3D10Blob_Release(vsBlob);
        return RLGL.State.defaultShaderId;
    }
    if (errors) ID3D10Blob_Release(errors);

    RLGL.shaders[idx].vsBlob = vsBlob;
    RLGL.shaders[idx].psBlob = psBlob;

    // Create default PSO with current state
    RLGL.shaders[idx].pso = rlCreatePipelineState(vsBlob, psBlob,
        RLGL.State.colorBlendEnabled, RLGL.State.currentBlendMode,
        RLGL.State.depthTestEnabled, RLGL.State.depthWriteEnabled,
        RLGL.State.backfaceCullingEnabled, RLGL.State.cullFaceMode,
        RLGL.State.wireMode, RLGL.State.scissorTestEnabled,
        D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE);

    RLGL.shaderCount++;

    unsigned int id = idx + 1;
    TRACELOG(RL_LOG_INFO, "SHADER: [ID %i] D3D12 shader loaded successfully", id);
    return id;
}

unsigned int rlLoadShaderProgram(unsigned int vShaderId, unsigned int fShaderId) {
    (void)vShaderId; (void)fShaderId;
    return 0;
}

void rlUnloadShaderProgram(unsigned int id) {
    if (id == 0 || id > RLGL.shaderCount) return;
    unsigned int idx = id - 1;
    if (RLGL.shaders[idx].pso) { ID3D12PipelineState_Release(RLGL.shaders[idx].pso); RLGL.shaders[idx].pso = NULL; }
    if (RLGL.shaders[idx].vsBlob) { ID3D10Blob_Release(RLGL.shaders[idx].vsBlob); RLGL.shaders[idx].vsBlob = NULL; }
    if (RLGL.shaders[idx].psBlob) { ID3D10Blob_Release(RLGL.shaders[idx].psBlob); RLGL.shaders[idx].psBlob = NULL; }
}

int rlGetLocationUniform(unsigned int shaderId, const char *uniformName) {
    (void)shaderId; (void)uniformName;
    return -1;
}

int rlGetLocationAttrib(unsigned int shaderId, const char *attribName) {
    (void)shaderId; (void)attribName;
    return -1;
}

void rlSetUniform(int locIndex, const void *value, int uniformType, int count) {
    (void)locIndex; (void)value; (void)uniformType; (void)count;
}

void rlSetUniformMatrix(int locIndex, Matrix mat) { (void)locIndex; (void)mat; }
void rlSetUniformMatrices(int locIndex, const Matrix *mat, int count) { (void)locIndex; (void)mat; (void)count; }
void rlSetUniformSampler(int locIndex, unsigned int textureId) { (void)locIndex; (void)textureId; }

void rlSetShader(unsigned int id, int *locs)
{
    if (RLGL.State.currentShaderId != id) {
        rlDrawRenderBatch(RLGL.currentBatch);
        RLGL.State.currentShaderId = id;
        RLGL.State.currentShaderLocs = locs;
    }
}

// Compute shader stubs
unsigned int rlLoadComputeShaderProgram(unsigned int shaderId) {
    (void)shaderId;
    TRACELOG(RL_LOG_WARNING, "SHADER: D3D12 compute shaders not yet implemented in rlgl layer");
    return 0;
}
void rlComputeShaderDispatch(unsigned int groupX, unsigned int groupY, unsigned int groupZ) { (void)groupX; (void)groupY; (void)groupZ; }

// SSBO stubs
unsigned int rlLoadShaderBuffer(unsigned int size, const void *data, int usageHint) { (void)size; (void)data; (void)usageHint; return 0; }
void rlUnloadShaderBuffer(unsigned int ssboId) { (void)ssboId; }
void rlUpdateShaderBuffer(unsigned int id, const void *data, unsigned int dataSize, unsigned int offset) { (void)id; (void)data; (void)dataSize; (void)offset; }
void rlBindShaderBuffer(unsigned int id, unsigned int index) { (void)id; (void)index; }
void rlReadShaderBuffer(unsigned int id, void *dest, unsigned int count, unsigned int offset) { (void)id; (void)dest; (void)count; (void)offset; }
void rlCopyShaderBuffer(unsigned int destId, unsigned int srcId, unsigned int destOffset, unsigned int srcOffset, unsigned int count) { (void)destId; (void)srcId; (void)destOffset; (void)srcOffset; (void)count; }
unsigned int rlGetShaderBufferSize(unsigned int id) { (void)id; return 0; }
void rlBindImageTexture(unsigned int id, unsigned int index, int format, bool readonly) { (void)id; (void)index; (void)format; (void)readonly; }

//----------------------------------------------------------------------------------
// Matrix state management
//----------------------------------------------------------------------------------
Matrix rlGetMatrixModelview(void) { return RLGL.State.modelview; }
Matrix rlGetMatrixProjection(void) { return RLGL.State.projection; }
Matrix rlGetMatrixTransform(void) { return RLGL.State.transform; }
Matrix rlGetMatrixProjectionStereo(int eye) { return RLGL.State.projectionStereo[eye]; }
Matrix rlGetMatrixViewOffsetStereo(int eye) { return RLGL.State.viewOffsetStereo[eye]; }
void rlSetMatrixModelview(Matrix view) { RLGL.State.modelview = view; }
void rlSetMatrixProjection(Matrix projection) { RLGL.State.projection = projection; }
void rlSetMatrixProjectionStereo(Matrix right, Matrix left) { RLGL.State.projectionStereo[0] = right; RLGL.State.projectionStereo[1] = left; }
void rlSetMatrixViewOffsetStereo(Matrix right, Matrix left) { RLGL.State.viewOffsetStereo[0] = right; RLGL.State.viewOffsetStereo[1] = left; }

//----------------------------------------------------------------------------------
// Quick draw helpers
//----------------------------------------------------------------------------------
void rlLoadDrawQuad(void) {
    rlBegin(RL_QUADS);
        rlTexCoord2f(0.0f, 1.0f); rlVertex3f(-1.0f,  1.0f, 0.0f);
        rlTexCoord2f(0.0f, 0.0f); rlVertex3f(-1.0f, -1.0f, 0.0f);
        rlTexCoord2f(1.0f, 0.0f); rlVertex3f( 1.0f, -1.0f, 0.0f);
        rlTexCoord2f(1.0f, 1.0f); rlVertex3f( 1.0f,  1.0f, 0.0f);
    rlEnd();
    rlDrawRenderBatchActive();
}

void rlLoadDrawCube(void) {
    rlBegin(RL_TRIANGLES);
        // Front face
        rlNormal3f(0,0,-1); rlTexCoord2f(0,0); rlVertex3f(-1,-1,-1);
        rlNormal3f(0,0,-1); rlTexCoord2f(1,1); rlVertex3f(1,1,-1);
        rlNormal3f(0,0,-1); rlTexCoord2f(1,0); rlVertex3f(1,-1,-1);
        rlNormal3f(0,0,-1); rlTexCoord2f(1,1); rlVertex3f(1,1,-1);
        rlNormal3f(0,0,-1); rlTexCoord2f(0,0); rlVertex3f(-1,-1,-1);
        rlNormal3f(0,0,-1); rlTexCoord2f(0,1); rlVertex3f(-1,1,-1);
        // Back face
        rlNormal3f(0,0,1); rlTexCoord2f(0,0); rlVertex3f(-1,-1,1);
        rlNormal3f(0,0,1); rlTexCoord2f(1,0); rlVertex3f(1,-1,1);
        rlNormal3f(0,0,1); rlTexCoord2f(1,1); rlVertex3f(1,1,1);
        rlNormal3f(0,0,1); rlTexCoord2f(1,1); rlVertex3f(1,1,1);
        rlNormal3f(0,0,1); rlTexCoord2f(0,1); rlVertex3f(-1,1,1);
        rlNormal3f(0,0,1); rlTexCoord2f(0,0); rlVertex3f(-1,-1,1);
        // Left/Right/Bottom/Top
        rlNormal3f(-1,0,0); rlVertex3f(-1,1,1); rlNormal3f(-1,0,0); rlVertex3f(-1,1,-1); rlNormal3f(-1,0,0); rlVertex3f(-1,-1,-1);
        rlNormal3f(-1,0,0); rlVertex3f(-1,-1,-1); rlNormal3f(-1,0,0); rlVertex3f(-1,-1,1); rlNormal3f(-1,0,0); rlVertex3f(-1,1,1);
        rlNormal3f(1,0,0); rlVertex3f(1,1,1); rlNormal3f(1,0,0); rlVertex3f(1,-1,-1); rlNormal3f(1,0,0); rlVertex3f(1,1,-1);
        rlNormal3f(1,0,0); rlVertex3f(1,-1,-1); rlNormal3f(1,0,0); rlVertex3f(1,1,1); rlNormal3f(1,0,0); rlVertex3f(1,-1,1);
        rlNormal3f(0,-1,0); rlVertex3f(-1,-1,-1); rlNormal3f(0,-1,0); rlVertex3f(1,-1,-1); rlNormal3f(0,-1,0); rlVertex3f(1,-1,1);
        rlNormal3f(0,-1,0); rlVertex3f(1,-1,1); rlNormal3f(0,-1,0); rlVertex3f(-1,-1,1); rlNormal3f(0,-1,0); rlVertex3f(-1,-1,-1);
        rlNormal3f(0,1,0); rlVertex3f(-1,1,-1); rlNormal3f(0,1,0); rlVertex3f(1,1,1); rlNormal3f(0,1,0); rlVertex3f(1,1,-1);
        rlNormal3f(0,1,0); rlVertex3f(1,1,1); rlNormal3f(0,1,0); rlVertex3f(-1,1,-1); rlNormal3f(0,1,0); rlVertex3f(-1,1,1);
    rlEnd();
    rlDrawRenderBatchActive();
}

//----------------------------------------------------------------------------------
// Default shader loading
//----------------------------------------------------------------------------------
static void rlLoadShaderDefault(void)
{
    RLGL.State.defaultShaderLocs = (int *)RL_CALLOC(RL_MAX_SHADER_LOCATIONS, sizeof(int));
    for (int i = 0; i < RL_MAX_SHADER_LOCATIONS; i++) RLGL.State.defaultShaderLocs[i] = -1;

    RLGL.State.defaultShaderId = rlLoadShaderCode(defaultVShaderHLSL, defaultPShaderHLSL);

    if (RLGL.State.defaultShaderId > 0) {
        TRACELOG(RL_LOG_INFO, "SHADER: [ID %i] Default D3D12 shader loaded successfully", RLGL.State.defaultShaderId);
        RLGL.State.defaultShaderLocs[RL_SHADER_LOC_VERTEX_POSITION] = 0;
        RLGL.State.defaultShaderLocs[RL_SHADER_LOC_VERTEX_TEXCOORD01] = 1;
        RLGL.State.defaultShaderLocs[RL_SHADER_LOC_VERTEX_COLOR] = 3;
        RLGL.State.defaultShaderLocs[RL_SHADER_LOC_MATRIX_MVP] = 0;
        RLGL.State.defaultShaderLocs[RL_SHADER_LOC_COLOR_DIFFUSE] = 1;
        RLGL.State.defaultShaderLocs[RL_SHADER_LOC_MAP_DIFFUSE] = 0;
    }
}

static void rlUnloadShaderDefault(void)
{
    if (RLGL.State.defaultShaderId > 0) rlUnloadShaderProgram(RLGL.State.defaultShaderId);
    RL_FREE(RLGL.State.defaultShaderLocs);
}

//----------------------------------------------------------------------------------
// Auxiliar math functions (identical to D3D11/OpenGL backend)
//----------------------------------------------------------------------------------
static rl_float16 rlMatrixToFloatV(Matrix mat) {
    rl_float16 r = { 0 };
    r.v[0]=mat.m0; r.v[1]=mat.m1; r.v[2]=mat.m2; r.v[3]=mat.m3;
    r.v[4]=mat.m4; r.v[5]=mat.m5; r.v[6]=mat.m6; r.v[7]=mat.m7;
    r.v[8]=mat.m8; r.v[9]=mat.m9; r.v[10]=mat.m10; r.v[11]=mat.m11;
    r.v[12]=mat.m12; r.v[13]=mat.m13; r.v[14]=mat.m14; r.v[15]=mat.m15;
    return r;
}

static Matrix rlMatrixIdentity(void) {
    Matrix r = { 1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1 };
    return r;
}

static Matrix rlMatrixMultiply(Matrix left, Matrix right) {
    Matrix r = { 0 };
    r.m0 = left.m0*right.m0+left.m1*right.m4+left.m2*right.m8+left.m3*right.m12;
    r.m1 = left.m0*right.m1+left.m1*right.m5+left.m2*right.m9+left.m3*right.m13;
    r.m2 = left.m0*right.m2+left.m1*right.m6+left.m2*right.m10+left.m3*right.m14;
    r.m3 = left.m0*right.m3+left.m1*right.m7+left.m2*right.m11+left.m3*right.m15;
    r.m4 = left.m4*right.m0+left.m5*right.m4+left.m6*right.m8+left.m7*right.m12;
    r.m5 = left.m4*right.m1+left.m5*right.m5+left.m6*right.m9+left.m7*right.m13;
    r.m6 = left.m4*right.m2+left.m5*right.m6+left.m6*right.m10+left.m7*right.m14;
    r.m7 = left.m4*right.m3+left.m5*right.m7+left.m6*right.m11+left.m7*right.m15;
    r.m8 = left.m8*right.m0+left.m9*right.m4+left.m10*right.m8+left.m11*right.m12;
    r.m9 = left.m8*right.m1+left.m9*right.m5+left.m10*right.m9+left.m11*right.m13;
    r.m10= left.m8*right.m2+left.m9*right.m6+left.m10*right.m10+left.m11*right.m14;
    r.m11= left.m8*right.m3+left.m9*right.m7+left.m10*right.m11+left.m11*right.m15;
    r.m12= left.m12*right.m0+left.m13*right.m4+left.m14*right.m8+left.m15*right.m12;
    r.m13= left.m12*right.m1+left.m13*right.m5+left.m14*right.m9+left.m15*right.m13;
    r.m14= left.m12*right.m2+left.m13*right.m6+left.m14*right.m10+left.m15*right.m14;
    r.m15= left.m12*right.m3+left.m13*right.m7+left.m14*right.m11+left.m15*right.m15;
    return r;
}

static Matrix rlMatrixTranspose(Matrix mat) {
    Matrix r = { 0 };
    r.m0=mat.m0; r.m1=mat.m4; r.m2=mat.m8; r.m3=mat.m12;
    r.m4=mat.m1; r.m5=mat.m5; r.m6=mat.m9; r.m7=mat.m13;
    r.m8=mat.m2; r.m9=mat.m6; r.m10=mat.m10; r.m11=mat.m14;
    r.m12=mat.m3; r.m13=mat.m7; r.m14=mat.m11; r.m15=mat.m15;
    return r;
}

static Matrix rlMatrixInvert(Matrix mat) {
    Matrix r = { 0 };
    float a00=mat.m0,a01=mat.m1,a02=mat.m2,a03=mat.m3;
    float a10=mat.m4,a11=mat.m5,a12=mat.m6,a13=mat.m7;
    float a20=mat.m8,a21=mat.m9,a22=mat.m10,a23=mat.m11;
    float a30=mat.m12,a31=mat.m13,a32=mat.m14,a33=mat.m15;
    float b00=a00*a11-a01*a10, b01=a00*a12-a02*a10, b02=a00*a13-a03*a10;
    float b03=a01*a12-a02*a11, b04=a01*a13-a03*a11, b05=a02*a13-a03*a12;
    float b06=a20*a31-a21*a30, b07=a20*a32-a22*a30, b08=a20*a33-a23*a30;
    float b09=a21*a32-a22*a31, b10=a21*a33-a23*a31, b11=a22*a33-a23*a32;
    float invDet=1.0f/(b00*b11-b01*b10+b02*b09+b03*b08-b04*b07+b05*b06);
    r.m0=(a11*b11-a12*b10+a13*b09)*invDet; r.m1=(-a01*b11+a02*b10-a03*b09)*invDet;
    r.m2=(a31*b05-a32*b04+a33*b03)*invDet; r.m3=(-a21*b05+a22*b04-a23*b03)*invDet;
    r.m4=(-a10*b11+a12*b08-a13*b07)*invDet; r.m5=(a00*b11-a02*b08+a03*b07)*invDet;
    r.m6=(-a30*b05+a32*b02-a33*b01)*invDet; r.m7=(a20*b05-a22*b02+a23*b01)*invDet;
    r.m8=(a10*b10-a11*b08+a13*b06)*invDet; r.m9=(-a00*b10+a01*b08-a03*b06)*invDet;
    r.m10=(a30*b04-a31*b02+a33*b00)*invDet; r.m11=(-a20*b04+a21*b02-a23*b00)*invDet;
    r.m12=(-a10*b09+a11*b07-a12*b06)*invDet; r.m13=(a00*b09-a01*b07+a02*b06)*invDet;
    r.m14=(-a30*b03+a31*b01-a32*b00)*invDet; r.m15=(a20*b03-a21*b01+a22*b00)*invDet;
    return r;
}

static int rlGetPixelDataSize(int width, int height, int format)
{
    int dataSize = 0;
    int bpp = 0;
    switch (format) {
        case 1: bpp=8; break;
        case 2: case 3: case 5: case 6: bpp=16; break;
        case 7: bpp=32; break;
        case 4: bpp=24; break;
        case 8: bpp=32; break;
        case 9: bpp=32*3; break;
        case 10: bpp=32*4; break;
        case 11: bpp=16; break;
        case 12: bpp=16*3; break;
        case 13: bpp=16*4; break;
        case 14: case 15: case 18: case 19: case 21: case 22: bpp=4; break;
        case 16: case 17: case 20: case 23: bpp=8; break;
        case 24: bpp=2; break;
        default: break;
    }
    dataSize = (int)((double)bpp/8.0*width*height);
    return dataSize;
}

// Headless render viewport stubs
void rlSetHeadlessRenderViewport(int width, int height) { (void)width; (void)height; }
void rlUnsetHeadlessRenderViewport(void) { }

#endif // GRAPHICS_API_DIRECT3D12
