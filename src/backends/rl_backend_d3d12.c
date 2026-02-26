/**********************************************************************************************
*
*   rl_backend_d3d12 - Direct3D 12 rendering backend for rlgl
*
*   DESCRIPTION:
*       Complete Direct3D 12 implementation of the rlgl rendering API.
*       Provides all functions required by rlgl.h when GRAPHICS_API_DIRECT3D12 is defined.
*       Includes full shader reflection, per-uniform cbuffer management, cubemap support,
*       swap chain management, and mesh rendering path (ported from D3D11 backend).
*
*   COMPILATION:
*       Only compiled when GRAPHICS_API_DIRECT3D12 is defined (Windows 10+ only).
*       Requires: d3d12.lib, d3dcompiler.lib, dxgi.lib, dxguid.lib
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
#include <d3d11shader.h>        // For shader reflection (SM 5.0 bytecode is compatible)
#include <d3dcompiler.h>
#include <dxgi1_4.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <stdio.h>

#pragma comment(lib, "d3d12.lib")
#pragma comment(lib, "d3dcompiler.lib")
#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "dxguid.lib")

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
#define RL_D3D12_NUM_FRAMES         2
#define RL_D3D12_MAX_SRV_DESC       4096
#define RL_D3D12_MAX_CBUFFERS       16
#define RL_D3D12_VS_CBV_COUNT       4       // VS cbuffer root params (b0-b3, SHADER_VISIBILITY_VERTEX)
#define RL_D3D12_PS_CBV_COUNT       8       // PS cbuffer root params (b0-b7, SHADER_VISIBILITY_PIXEL)
#define RL_D3D12_ROOT_CBV_COUNT     (RL_D3D12_VS_CBV_COUNT + RL_D3D12_PS_CBV_COUNT)  // Total CBV root params
#define RL_D3D12_ROOT_SRV_TABLE_IDX RL_D3D12_ROOT_CBV_COUNT  // Root parameter index for the SRV descriptor table
#define RL_D3D12_MAX_DEFERRED_RELEASES 512  // Max PSOs awaiting GPU completion before release
#define RL_D3D12_PSO_CACHE_SIZE 256         // Hash table size for cached PSOs (power of 2)
#define RL_D3D12_MAX_UNIFORM_VARS   64
#define RL_D3D12_MAX_TEXTURE_BINDINGS 16
#define RL_D3D12_TEX_LOC_OFFSET     10000
#define RL_MAX_MATERIAL_MAPS        12

//----------------------------------------------------------------------------------
// Types - D3D12 resource tracking
//----------------------------------------------------------------------------------
typedef struct {
    ID3D12Resource *resource;
    D3D12_CPU_DESCRIPTOR_HANDLE srvCpuHandle;
    D3D12_GPU_DESCRIPTOR_HANDLE srvGpuHandle;
    int srvIndex;
    int width, height, format;
    bool isCubemap;
    bool isDepth;
    D3D12_CPU_DESCRIPTOR_HANDLE dsvHandle;
    D3D12_CPU_DESCRIPTOR_HANDLE rtvHandle;
    int dsvIndex;
    int rtvIndex;
    DXGI_FORMAT dxgiFormat;
} rlD3D12Texture;

// Per-constant-buffer info (tracked per shader via reflection)
typedef struct {
    int stage;               // 0 = VS, 1 = PS
    int registerSlot;        // register(bN) slot
    int byteSize;            // Total cbuffer size (16-byte aligned)
    ID3D12Resource *uploadBuffer;   // Upload heap buffer (persistently mapped)
    unsigned char *cpuData;         // CPU-side copy for partial updates
    void *mappedData;               // Persistently mapped pointer
    bool dirty;                     // True if cpuData changed since last GPU bind
} rlD3D12CBufferInfo;

// Per-uniform variable info (tracked per shader via reflection)
typedef struct {
    char name[128];
    int cbufferIndex;        // Index into shader's cbuffers[] array
    int byteOffset;          // Offset within the constant buffer
    int byteSize;            // Size of this variable in bytes
} rlD3D12UniformVar;

// Per-shader texture binding info (from reflection)
typedef struct {
    char name[128];
    int registerSlot;        // register(tN) slot in HLSL
} rlD3D12TextureBindingInfo;

typedef struct {
    ID3DBlob *vsBlob;
    ID3DBlob *psBlob;
    ID3D12PipelineState *pso;       // PSO for triangle topology
    ID3D12PipelineState *psoLine;   // PSO for line topology
    // Constant buffer tracking (populated by shader reflection)
    rlD3D12CBufferInfo cbuffers[RL_D3D12_MAX_CBUFFERS];
    int cbufferCount;
    // Uniform variable tracking (populated by shader reflection)
    rlD3D12UniformVar uniforms[RL_D3D12_MAX_UNIFORM_VARS];
    int uniformCount;
    // Texture binding tracking (populated by shader reflection)
    rlD3D12TextureBindingInfo textureBindings[RL_D3D12_MAX_TEXTURE_BINDINGS];
    int textureBindingCount;
    // Material-map-index -> D3D12 register slot mapping
    int texMaterialMapToRegister[RL_MAX_MATERIAL_MAPS];
} rlD3D12Shader;

typedef struct {
    ID3D12Resource *resource;
    int size;
    bool dynamic;
    bool isIndex;
    void *mappedData;
} rlD3D12Buffer;

typedef struct {
    int colorTexIds[8];
    int depthTexId;
    int colorCount;
    bool complete;
} rlD3D12Framebuffer;

// PSO cache entry for hash-table-based PSO reuse
typedef struct {
    uint64_t key;           // 0 = empty slot
    ID3D12PipelineState *pso;
} rlD3D12PsoCacheEntry;

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
#define RL_D3D12_MAX_BATCH_DRAWS_PER_FRAME 256   // Max rlDrawRenderBatch calls per frame (sub-allocated within upload buffers)
#define RL_D3D12_MAX_PENDING_UPLOADS 512          // Max upload buffers awaiting GPU completion before release
#define RL_D3D12_CB_RING_SIZE (64 * 1024 * 1024)  // Per-frame ring buffer for mesh shader cbuffers (64 MB)

typedef struct {
    ID3D12CommandAllocator *commandAllocator;
    ID3D12Resource *cbMatrixUpload;
    ID3D12Resource *cbColorUpload;
    void *cbMatrixMapped;
    void *cbColorMapped;
    ID3D12Resource *batchVertexBuffers[4];
    void *batchVertexMapped[4];
    ID3D12Resource *batchIndexBuffer;
    void *batchIndexMapped;
    UINT64 fenceValue;
    // Per-frame sub-allocation tracking: each rlDrawRenderBatch call within a
    // single frame advances these offsets so that consecutive batch draws write
    // into distinct regions of the upload buffers.  Without this, the second
    // batch draw (e.g. 2D overlay) overwrites the first (3D scene) before the
    // GPU has read either, since commands execute only at Present time.
    UINT batchCBSlot;       // Next 256-byte aligned CB slot index  (0..MAX-1)
    UINT batchVBOffset;     // Cumulative vertex offset into VB uploads

    // Deferred upload buffer release: upload buffers created for texture/buffer
    // initialisation are kept alive until the GPU has consumed them.
    ID3D12Resource *pendingUploads[RL_D3D12_MAX_PENDING_UPLOADS];
    int pendingUploadCount;

    // Per-frame constant buffer ring buffer for mesh shader uniforms.
    // Eliminates the race condition where CPU overwrites shared shader cbuffers
    // while the GPU is still reading them from the previous frame's command list.
    ID3D12Resource *cbRingBuffer;
    void *cbRingMapped;
    UINT cbRingOffset;
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

        // Tracked scissor rect (saved so we can re-apply after command list Reset)
        int scissorX, scissorY, scissorWidth, scissorHeight;
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
    ID3D12DescriptorHeap *srvHeap;
    ID3D12DescriptorHeap *rtvHeap;
    ID3D12DescriptorHeap *dsvHeap;
    UINT srvDescriptorSize;
    UINT rtvDescriptorSize;
    UINT dsvDescriptorSize;
    int srvHeapUsed;
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

    // PSO hash-table cache: avoids expensive re-creation when render state toggles
    rlD3D12PsoCacheEntry psoCache[RL_D3D12_PSO_CACHE_SIZE];

    // Swap chain
    IDXGISwapChain3 *swapChain;
    ID3D12Resource *renderTargets[RL_D3D12_NUM_FRAMES];
    D3D12_CPU_DESCRIPTOR_HANDLE renderTargetHandles[RL_D3D12_NUM_FRAMES];
    ID3D12Resource *depthStencilBuffer;
    D3D12_CPU_DESCRIPTOR_HANDLE dsvBackbuffer;
    int swapChainRTVBase;
    int swapChainDSVIndex;

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

    // Deferred PSO release queue (PSOs awaiting GPU completion before safe release)
    struct {
        ID3D12PipelineState *pso;
        UINT64 fenceValue;          // Release is safe once GPU completes this fence value
    } deferredPsoReleases[RL_D3D12_MAX_DEFERRED_RELEASES];
    int deferredPsoReleaseCount;

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
static void rlRestoreStateAfterFlush(void);  // Re-establish D3D12 state after mid-frame command list reset
static ID3D12Resource *rlCreateUploadBuffer(UINT size);
static ID3D12Resource *rlCreateDefaultBuffer(UINT size, D3D12_RESOURCE_FLAGS flags);
static void rlTransitionResource(ID3D12Resource *resource, D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after);
static void rlReflectShaderUniforms(unsigned int shaderIdx);
static void rlUpdateD3D12PipelineState(void);
static void rlDeferPsoRelease(ID3D12PipelineState *pso);
static void rlFlushDeferredPsoReleases(void);
static void rlDeferUploadRelease(ID3D12Resource *upload);
static void rlReleasePendingUploads(rlD3D12FrameResources *frame);
static D3D12_GPU_VIRTUAL_ADDRESS rlSuballocateCB(const unsigned char *data, UINT byteSize);
static void rlFlushDirtyCBuffers(void);    // Bind all dirty cbuffers before a draw call
static void rlEnsurePipelineState(void);   // Re-select PSO if render state changed since rlEnableShader
static ID3D12PipelineState *rlGetOrCreatePso(unsigned int shaderId,
    bool blendEnabled, int blendMode, bool depthTest, bool depthWrite,
    bool backfaceCull, int cullFaceMode, bool wireMode, bool scissorTest,
    D3D12_PRIMITIVE_TOPOLOGY_TYPE topoType);
static bool rlIsPsoCached(ID3D12PipelineState *pso);
static void rlInvalidatePsoCacheForShader(unsigned int shaderId);
static void rlClearPsoCache(void);
static void rlEnsureGraphicsStateForMesh(void);

// External: raylib provides this to get the native window handle (HWND)
extern void *GetWindowHandle(void);

//----------------------------------------------------------------------------------
// Global state
//----------------------------------------------------------------------------------
static double rlCullDistanceNear = RL_CULL_DISTANCE_NEAR;
static double rlCullDistanceFar = RL_CULL_DISTANCE_FAR;
static rlglData RLGL = { 0 };

// D3D12-specific rendering state (mesh path)
static unsigned int d3d12_activeShaderId = 0;
static unsigned int d3d12_lastEnabledShader = 0;  // Persists across rlDisableShader for perf optimization
static int d3d12_activeTextureSlot = 0;
static unsigned int d3d12_pendingVBO = 0;

// Track last bound VB/IB state for restoration after mid-frame command list flush
static D3D12_VERTEX_BUFFER_VIEW d3d12_lastVBViews[4] = { 0 };
static D3D12_INDEX_BUFFER_VIEW d3d12_lastIBView = { 0 };
static bool d3d12_hasVBBound = false;
static bool d3d12_hasIBBound = false;

// Track whether base D3D12 state (root sig, heaps, viewport, scissor, RT) has been
// set on the current command list.  Reset to false on command list Reset(), set to
// true after batch draw or rlEnableShader sets it.  Allows rlEnableShader to skip
// redundant per-object state setup for 10K+ object scenes.
static bool d3d12_baseStateSet = false;

// Track the last PSO bound to the command list to skip redundant SetPipelineState
// calls when drawing 10K+ objects with the same shader/state.
static ID3D12PipelineState *d3d12_lastBoundPSO = NULL;

// One-shot diagnostic: logs depth state on first mesh draw
static bool d3d12_depthDiagLogged = false;

// Cached back-buffer index: updated once per frame in rlClearScreenBuffers to avoid
// calling IDXGISwapChain3_GetCurrentBackBufferIndex (a DXGI IPC call) in every draw.
static UINT d3d12_cachedBackBufferIndex = 0;

// Default white vertex buffer for meshes without vertex colors
static ID3D12Resource *d3d12_defaultWhiteVBO = NULL;
static D3D12_VERTEX_BUFFER_VIEW d3d12_defaultWhiteVBOView = { 0 };
#define RL_D3D12_DEFAULT_WHITE_VBO_VERTS 65536

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
        case 4:  return DXGI_FORMAT_R8G8B8A8_UNORM;
        case 5:  return DXGI_FORMAT_B5G5R5A1_UNORM;
        case 6:  return DXGI_FORMAT_B4G4R4A4_UNORM;
        case 7:  return DXGI_FORMAT_R8G8B8A8_UNORM;
        case 8:  return DXGI_FORMAT_R32_FLOAT;
        case 9:  return DXGI_FORMAT_R32G32B32_FLOAT;
        case 10: return DXGI_FORMAT_R32G32B32A32_FLOAT;
        case 11: return DXGI_FORMAT_R16_FLOAT;
        case 12: return DXGI_FORMAT_R16G16B16A16_FLOAT;
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
    rlD3D12FrameResources *frame = &RLGL.frames[RLGL.currentFrame];
    ID3D12CommandAllocator_Reset(frame->commandAllocator);
    ID3D12GraphicsCommandList_Reset(RLGL.commandList, frame->commandAllocator, NULL);
    d3d12_baseStateSet = false;
    d3d12_lastBoundPSO = NULL;
    d3d12_lastEnabledShader = 0;
}
static void rlDeferUploadRelease(ID3D12Resource *upload)
{
    if (!upload) return;
    rlD3D12FrameResources *frame = &RLGL.frames[RLGL.currentFrame];
    if (frame->pendingUploadCount < RL_D3D12_MAX_PENDING_UPLOADS) {
        frame->pendingUploads[frame->pendingUploadCount++] = upload;
    } else {
        // Overflow: force GPU sync and release immediately
        rlFlushCommandList();
        ID3D12Resource_Release(upload);
    }
}

//----------------------------------------------------------------------------------
// Helper: Release all pending upload buffers for a frame (called after GPU fence wait)
//----------------------------------------------------------------------------------
static void rlReleasePendingUploads(rlD3D12FrameResources *frame)
{
    for (int i = 0; i < frame->pendingUploadCount; i++) {
        if (frame->pendingUploads[i]) {
            ID3D12Resource_Release(frame->pendingUploads[i]);
            frame->pendingUploads[i] = NULL;
        }
    }
    frame->pendingUploadCount = 0;
}

//----------------------------------------------------------------------------------
// Helper: Sub-allocate from the per-frame constant buffer ring buffer.
// Returns the GPU virtual address of the sub-allocation, or 0 on failure.
// This eliminates the race condition where CPU overwrites shared shader cbuffers
// while the GPU is still reading them from the previous frame's command list.
//----------------------------------------------------------------------------------
// Helper: Re-establish all D3D12 state on the command list after a mid-frame flush.
// After rlFlushCommandList() resets the allocator+list, ALL state is lost (PSO,
// root sig, descriptor heaps, viewport, scissor, RT/DSV binding, topology, VBs/IBs).
// We must restore enough state so that subsequent draw calls work correctly.
static void rlRestoreStateAfterFlush(void)
{
    if (!RLGL.commandList) return;

    // Root signature + descriptor heaps (required for any SetGraphicsRoot* call)
    if (RLGL.rootSignature)
        ID3D12GraphicsCommandList_SetGraphicsRootSignature(RLGL.commandList, RLGL.rootSignature);
    if (RLGL.srvHeap) {
        ID3D12DescriptorHeap *heaps[] = { RLGL.srvHeap };
        ID3D12GraphicsCommandList_SetDescriptorHeaps(RLGL.commandList, 1, heaps);
    }

    // Viewport + scissor
    D3D12_VIEWPORT vp = { 0, 0, (float)RLGL.State.framebufferWidth, (float)RLGL.State.framebufferHeight, 0.0f, 1.0f };
    ID3D12GraphicsCommandList_RSSetViewports(RLGL.commandList, 1, &vp);
    D3D12_RECT sr = { 0, 0, RLGL.State.framebufferWidth, RLGL.State.framebufferHeight };
    if (RLGL.State.scissorTestEnabled) {
        sr.left = RLGL.State.scissorX; sr.top = RLGL.State.scissorY;
        sr.right = RLGL.State.scissorX + RLGL.State.scissorWidth;
        sr.bottom = RLGL.State.scissorY + RLGL.State.scissorHeight;
    }
    ID3D12GraphicsCommandList_RSSetScissorRects(RLGL.commandList, 1, &sr);

    // Render target binding (backbuffer is already in RENDER_TARGET state from rlClearScreenBuffers,
    // and ExecuteCommandLists does NOT change resource states — barriers are only recorded in the
    // command list, so the GPU-side state persists across submits).
    if (RLGL.activeFramebuffer > 0 && RLGL.activeFramebuffer <= RLGL.framebufferCount) {
        rlD3D12Framebuffer *fb = &RLGL.framebuffers[RLGL.activeFramebuffer - 1];
        if (fb->colorCount > 0 && fb->colorTexIds[0] > 0 && fb->colorTexIds[0] <= RLGL.textureCount) {
            D3D12_CPU_DESCRIPTOR_HANDLE rtvH = RLGL.textures[fb->colorTexIds[0] - 1].rtvHandle;
            D3D12_CPU_DESCRIPTOR_HANDLE *dsvPtr = NULL;
            D3D12_CPU_DESCRIPTOR_HANDLE dsvH = { 0 };
            if (fb->depthTexId > 0 && fb->depthTexId <= RLGL.textureCount) {
                dsvH = RLGL.textures[fb->depthTexId - 1].dsvHandle;
                dsvPtr = &dsvH;
            }
            ID3D12GraphicsCommandList_OMSetRenderTargets(RLGL.commandList, 1, &rtvH, FALSE, dsvPtr);
        }
    } else if (RLGL.swapChain) {
        UINT backIdx = IDXGISwapChain3_GetCurrentBackBufferIndex(RLGL.swapChain);
        if (RLGL.renderTargets[backIdx]) {
            ID3D12GraphicsCommandList_OMSetRenderTargets(RLGL.commandList, 1,
                &RLGL.renderTargetHandles[backIdx], FALSE,
                RLGL.depthStencilBuffer ? &RLGL.dsvBackbuffer : NULL);
        }
    }

    // Re-bind PSO for the active shader
    unsigned int sid = d3d12_activeShaderId;
    if (sid == 0) sid = RLGL.State.currentShaderId;  // batch path fallback
    if (sid > 0 && sid <= RLGL.shaderCount && RLGL.shaders[sid - 1].pso) {
        ID3D12GraphicsCommandList_SetPipelineState(RLGL.commandList, RLGL.shaders[sid - 1].pso);
        d3d12_lastBoundPSO = RLGL.shaders[sid - 1].pso;
        d3d12_lastEnabledShader = sid;  // Prevent redundant PSO re-lookup after flush
    }

    // Mark all cbuffers dirty so they get re-bound before the next draw
    if (sid > 0 && sid <= RLGL.shaderCount) {
        rlD3D12Shader *s = &RLGL.shaders[sid - 1];
        for (int i = 0; i < s->cbufferCount; i++) {
            s->cbuffers[i].dirty = true;
        }
    }

    ID3D12GraphicsCommandList_IASetPrimitiveTopology(RLGL.commandList, D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

    // Restore VB/IB bindings that were set before the flush
    if (d3d12_hasVBBound) {
        ID3D12GraphicsCommandList_IASetVertexBuffers(RLGL.commandList, 0, 4, d3d12_lastVBViews);
    }
    if (d3d12_hasIBBound) {
        ID3D12GraphicsCommandList_IASetIndexBuffer(RLGL.commandList, &d3d12_lastIBView);
    }

    // Restore texture SRV binding
    unsigned int texId = RLGL.State.activeTextureId[0];
    if (texId > 0 && texId <= RLGL.textureCount) {
        ID3D12GraphicsCommandList_SetGraphicsRootDescriptorTable(RLGL.commandList, RL_D3D12_ROOT_SRV_TABLE_IDX,
            RLGL.textures[texId - 1].srvGpuHandle);
    }

    d3d12_baseStateSet = true;
}

static D3D12_GPU_VIRTUAL_ADDRESS rlSuballocateCB(const unsigned char *data, UINT byteSize)
{
    if (!data || byteSize == 0) return 0;
    rlD3D12FrameResources *frame = &RLGL.frames[RLGL.currentFrame];
    if (!frame->cbRingBuffer || !frame->cbRingMapped) return 0;

    UINT alignedSize = (byteSize + 255) & ~255;  // D3D12 CB alignment requirement
    if (frame->cbRingOffset + alignedSize > RL_D3D12_CB_RING_SIZE) {
        // Ring buffer exhausted — flush the command list to the GPU so it reads all
        // recorded draws, then reset the ring.  This is safe because rlFlushCommandList()
        // calls rlWaitForGpu() which ensures the GPU has finished reading everything.
        TRACELOG(RL_LOG_WARNING, "D3D12: CB ring buffer full (used=%u, needed=%u), flushing command list",
            frame->cbRingOffset, alignedSize);
        rlFlushCommandList();
        frame->cbRingOffset = 0;
        // After command list reset, ALL D3D12 state is lost — restore it
        rlRestoreStateAfterFlush();
    }

    UINT offset = frame->cbRingOffset;
    memcpy((char *)frame->cbRingMapped + offset, data, byteSize);
    frame->cbRingOffset = offset + alignedSize;

    return ID3D12Resource_GetGPUVirtualAddress(frame->cbRingBuffer) + offset;
}

//----------------------------------------------------------------------------------
// Helper: Flush all dirty constant buffers for the currently active shader.
// Called just before draw calls to ensure the GPU sees the latest uniform data.
// This deferred approach eliminates redundant ring buffer suballocations that
// occurred when rlEnableShader re-bound ALL cbuffers and then individual
// rlSetUniform* calls re-bound the same cbuffers with updated data.
//----------------------------------------------------------------------------------
static void rlFlushDirtyCBuffers(void)
{
    if (!RLGL.commandList) return;
    unsigned int activeShader = d3d12_activeShaderId;
    if (activeShader == 0 || activeShader > RLGL.shaderCount) return;
    rlD3D12Shader *s = &RLGL.shaders[activeShader - 1];

    // We must restart the loop if a ring buffer wrap causes rlRestoreStateAfterFlush()
    // to mark all cbuffers dirty again (the ones we already processed in this loop
    // iteration will have been submitted in the PREVIOUS command list and need to be
    // re-bound in the new one).
    bool needRestart;
    do {
        needRestart = false;
        UINT ringOffsetBefore = RLGL.frames[RLGL.currentFrame].cbRingOffset;
        for (int i = 0; i < s->cbufferCount; i++) {
            if (s->cbuffers[i].dirty && s->cbuffers[i].cpuData && s->cbuffers[i].byteSize > 0) {
                int rootParamIdx = rlGetRootParamForCBuffer(s->cbuffers[i].stage, s->cbuffers[i].registerSlot);
                if (rootParamIdx >= 0) {
                    D3D12_GPU_VIRTUAL_ADDRESS gpuAddr = rlSuballocateCB(s->cbuffers[i].cpuData, s->cbuffers[i].byteSize);
                    if (gpuAddr) {
                        ID3D12GraphicsCommandList_SetGraphicsRootConstantBufferView(RLGL.commandList, rootParamIdx, gpuAddr);
                    }
                    // Detect if rlSuballocateCB caused a ring wrap + flush:
                    // ring offset went backwards → command list was reset → need to restart
                    if (RLGL.frames[RLGL.currentFrame].cbRingOffset < ringOffsetBefore) {
                        needRestart = true;
                        break;  // Restart the entire loop (rlRestoreStateAfterFlush already marked all dirty)
                    }
                    ringOffsetBefore = RLGL.frames[RLGL.currentFrame].cbRingOffset;
                }
                s->cbuffers[i].dirty = false;
            }
        }
    } while (needRestart);
}

//----------------------------------------------------------------------------------
// Helper: Queue a PSO for deferred release (safe after GPU finishes current work)
//----------------------------------------------------------------------------------
static void rlDeferPsoRelease(ID3D12PipelineState *pso)
{
    if (!pso) return;
    if (RLGL.deferredPsoReleaseCount < RL_D3D12_MAX_DEFERRED_RELEASES) {
        RLGL.deferredPsoReleases[RLGL.deferredPsoReleaseCount].pso = pso;
        // Tag with fenceValue+1: the current command list hasn't been submitted yet,
        // so it will complete at the NEXT signal (fenceValue+1).
        RLGL.deferredPsoReleases[RLGL.deferredPsoReleaseCount].fenceValue = RLGL.fenceValue + 1;
        RLGL.deferredPsoReleaseCount++;
    } else {
        // Overflow: force GPU wait and release immediately
        rlWaitForGpu();
        ID3D12PipelineState_Release(pso);
    }
}

//----------------------------------------------------------------------------------
// Helper: Release all deferred PSOs whose fence value has been reached by the GPU
//----------------------------------------------------------------------------------
static void rlFlushDeferredPsoReleases(void)
{
    if (RLGL.deferredPsoReleaseCount == 0) return;
    UINT64 completedValue = ID3D12Fence_GetCompletedValue(RLGL.fence);
    int writeIdx = 0;
    for (int i = 0; i < RLGL.deferredPsoReleaseCount; i++) {
        if (completedValue >= RLGL.deferredPsoReleases[i].fenceValue) {
            ID3D12PipelineState_Release(RLGL.deferredPsoReleases[i].pso);
        } else {
            RLGL.deferredPsoReleases[writeIdx++] = RLGL.deferredPsoReleases[i];
        }
    }
    RLGL.deferredPsoReleaseCount = writeIdx;
}

//----------------------------------------------------------------------------------
// Helper: Create root signature
// Expanded layout for mesh rendering:
//   [0] Root CBV b0 (MVP matrix) - VERTEX visibility
//   [1] Root CBV b1 (color/uniforms) - PIXEL visibility
//   [2] Descriptor table (SRV range t0-t15) - ALL visibility
//   [3] Root CBV b2+ (extra VS cbuffers) - VERTEX visibility
//   [4] Root CBV b2+ (extra PS cbuffers) - PIXEL visibility
//   Static samplers s0-s3 (linear wrap)
//----------------------------------------------------------------------------------
// Helper: compute the root parameter index for a constant buffer.
// VS cbuffers (stage 0) use root params 0..VS_CBV_COUNT-1.
// PS cbuffers (stage 1) use root params VS_CBV_COUNT..VS_CBV_COUNT+PS_CBV_COUNT-1.
// This mirrors D3D11's independent VSSetConstantBuffers/PSSetConstantBuffers.
static int rlGetRootParamForCBuffer(int stage, int registerSlot)
{
    if (stage == 0) {
        // Vertex shader — root params 0..VS_CBV_COUNT-1
        if (registerSlot >= 0 && registerSlot < RL_D3D12_VS_CBV_COUNT) return registerSlot;
    } else {
        // Pixel shader — root params VS_CBV_COUNT..VS_CBV_COUNT+PS_CBV_COUNT-1
        if (registerSlot >= 0 && registerSlot < RL_D3D12_PS_CBV_COUNT) return RL_D3D12_VS_CBV_COUNT + registerSlot;
    }
    return -1;  // Out of range
}

static HRESULT rlCreateRootSignature(void)
{
    // Root layout:
    //   [0..VS_CBV_COUNT-1]                         = VS CBVs (b0-b3, VERTEX only)
    //   [VS_CBV_COUNT..VS_CBV_COUNT+PS_CBV_COUNT-1] = PS CBVs (b0-b7, PIXEL only)
    //   [ROOT_SRV_TABLE_IDX]                         = SRV descriptor table (t0-t15)
    D3D12_ROOT_PARAMETER rootParams[RL_D3D12_ROOT_CBV_COUNT + 1] = { 0 };

    // VS CBVs — separate register namespace, SHADER_VISIBILITY_VERTEX
    for (int i = 0; i < RL_D3D12_VS_CBV_COUNT; i++) {
        rootParams[i].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
        rootParams[i].Descriptor.ShaderRegister = i;
        rootParams[i].Descriptor.RegisterSpace = 0;
        rootParams[i].ShaderVisibility = D3D12_SHADER_VISIBILITY_VERTEX;
    }

    // PS CBVs — separate register namespace, SHADER_VISIBILITY_PIXEL
    for (int i = 0; i < RL_D3D12_PS_CBV_COUNT; i++) {
        int idx = RL_D3D12_VS_CBV_COUNT + i;
        rootParams[idx].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
        rootParams[idx].Descriptor.ShaderRegister = i;
        rootParams[idx].Descriptor.RegisterSpace = 0;
        rootParams[idx].ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
    }

    // [RL_D3D12_ROOT_CBV_COUNT] Descriptor table with SRVs (t0-t15)
    D3D12_DESCRIPTOR_RANGE srvRange = { 0 };
    srvRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    srvRange.NumDescriptors = 16;
    srvRange.BaseShaderRegister = 0;
    srvRange.RegisterSpace = 0;
    srvRange.OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;

    rootParams[RL_D3D12_ROOT_CBV_COUNT].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    rootParams[RL_D3D12_ROOT_CBV_COUNT].DescriptorTable.NumDescriptorRanges = 1;
    rootParams[RL_D3D12_ROOT_CBV_COUNT].DescriptorTable.pDescriptorRanges = &srvRange;
    rootParams[RL_D3D12_ROOT_CBV_COUNT].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

    // Static samplers (s0-s3, all linear wrap)
    D3D12_STATIC_SAMPLER_DESC samplers[4] = { 0 };
    for (int i = 0; i < 4; i++) {
        samplers[i].Filter = D3D12_FILTER_MIN_MAG_MIP_LINEAR;
        samplers[i].AddressU = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
        samplers[i].AddressV = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
        samplers[i].AddressW = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
        samplers[i].ComparisonFunc = D3D12_COMPARISON_FUNC_NEVER;
        samplers[i].BorderColor = D3D12_STATIC_BORDER_COLOR_TRANSPARENT_BLACK;
        samplers[i].MinLOD = 0.0f;
        samplers[i].MaxLOD = D3D12_FLOAT32_MAX;
        samplers[i].ShaderRegister = i;
        samplers[i].RegisterSpace = 0;
        samplers[i].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    }

    D3D12_ROOT_SIGNATURE_DESC rsDesc = { 0 };
    rsDesc.NumParameters = RL_D3D12_ROOT_CBV_COUNT + 1;
    rsDesc.pParameters = rootParams;
    rsDesc.NumStaticSamplers = 4;
    rsDesc.pStaticSamplers = samplers;
    rsDesc.Flags = D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;

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
// Helper: Create a PSO for a given state + shader blobs
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

    D3D12_INPUT_ELEMENT_DESC inputLayout[] = {
        { "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
        { "TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT,    1, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
        { "NORMAL",   0, DXGI_FORMAT_R32G32B32_FLOAT, 2, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
        { "COLOR",    0, DXGI_FORMAT_R8G8B8A8_UNORM,  3, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
    };
    psoDesc.InputLayout.pInputElementDescs = inputLayout;
    psoDesc.InputLayout.NumElements = 4;

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

    psoDesc.BlendState.AlphaToCoverageEnable = FALSE;
    psoDesc.BlendState.IndependentBlendEnable = FALSE;
    psoDesc.BlendState.RenderTarget[0].BlendEnable = blendEnabled;
    psoDesc.BlendState.RenderTarget[0].LogicOpEnable = FALSE;
    psoDesc.BlendState.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
    if (blendEnabled) {
        switch (blendMode) {
            case 1:
                psoDesc.BlendState.RenderTarget[0].SrcBlend = D3D12_BLEND_SRC_ALPHA;
                psoDesc.BlendState.RenderTarget[0].DestBlend = D3D12_BLEND_ONE;
                psoDesc.BlendState.RenderTarget[0].BlendOp = D3D12_BLEND_OP_ADD;
                psoDesc.BlendState.RenderTarget[0].SrcBlendAlpha = D3D12_BLEND_ONE;
                psoDesc.BlendState.RenderTarget[0].DestBlendAlpha = D3D12_BLEND_ONE;
                psoDesc.BlendState.RenderTarget[0].BlendOpAlpha = D3D12_BLEND_OP_ADD;
                break;
            case 2:
                psoDesc.BlendState.RenderTarget[0].SrcBlend = D3D12_BLEND_DEST_COLOR;
                psoDesc.BlendState.RenderTarget[0].DestBlend = D3D12_BLEND_INV_SRC_ALPHA;
                psoDesc.BlendState.RenderTarget[0].BlendOp = D3D12_BLEND_OP_ADD;
                psoDesc.BlendState.RenderTarget[0].SrcBlendAlpha = D3D12_BLEND_ONE;
                psoDesc.BlendState.RenderTarget[0].DestBlendAlpha = D3D12_BLEND_INV_SRC_ALPHA;
                psoDesc.BlendState.RenderTarget[0].BlendOpAlpha = D3D12_BLEND_OP_ADD;
                break;
            default:
                psoDesc.BlendState.RenderTarget[0].SrcBlend = D3D12_BLEND_SRC_ALPHA;
                psoDesc.BlendState.RenderTarget[0].DestBlend = D3D12_BLEND_INV_SRC_ALPHA;
                psoDesc.BlendState.RenderTarget[0].BlendOp = D3D12_BLEND_OP_ADD;
                psoDesc.BlendState.RenderTarget[0].SrcBlendAlpha = D3D12_BLEND_ONE;
                psoDesc.BlendState.RenderTarget[0].DestBlendAlpha = D3D12_BLEND_INV_SRC_ALPHA;
                psoDesc.BlendState.RenderTarget[0].BlendOpAlpha = D3D12_BLEND_OP_ADD;
                break;
        }
    }

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
    TRACELOG(RL_LOG_DEBUG, "D3D12: PSO created — depth=%d dwrite=%d cull=%d blend=%d topo=%d",
             depthTest, depthWrite, backfaceCull, blendEnabled, (int)topoType);
    return pso;
}

//----------------------------------------------------------------------------------
// Helper: PSO cache – look up or create a PSO keyed by (shader, render state, topology)
// Avoids expensive CreateGraphicsPipelineState calls when render state toggles each frame.
//----------------------------------------------------------------------------------
static uint64_t rlComputePsoCacheKey(unsigned int shaderId, bool blend, int blendMode,
    bool depthTest, bool depthWrite, bool backfaceCull, int cullFaceMode,
    bool wireMode, bool scissorTest, D3D12_PRIMITIVE_TOPOLOGY_TYPE topoType)
{
    uint64_t key = ((uint64_t)shaderId) |
                   ((uint64_t)blend << 16) |
                   ((uint64_t)(blendMode & 7) << 17) |
                   ((uint64_t)depthTest << 20) |
                   ((uint64_t)depthWrite << 21) |
                   ((uint64_t)backfaceCull << 22) |
                   ((uint64_t)(cullFaceMode & 1) << 23) |
                   ((uint64_t)wireMode << 24) |
                   ((uint64_t)scissorTest << 25) |
                   ((uint64_t)(topoType & 3) << 26);
    if (key == 0) key = 1;  // 0 means empty slot
    return key;
}

static ID3D12PipelineState *rlGetOrCreatePso(unsigned int shaderId,
    bool blendEnabled, int blendMode, bool depthTest, bool depthWrite,
    bool backfaceCull, int cullFaceMode, bool wireMode, bool scissorTest,
    D3D12_PRIMITIVE_TOPOLOGY_TYPE topoType)
{
    if (shaderId == 0 || shaderId > RLGL.shaderCount) return NULL;
    unsigned int shaderIdx = shaderId - 1;
    if (!RLGL.shaders[shaderIdx].vsBlob || !RLGL.shaders[shaderIdx].psBlob) return NULL;

    uint64_t key = rlComputePsoCacheKey(shaderId, blendEnabled, blendMode,
        depthTest, depthWrite, backfaceCull, cullFaceMode, wireMode, scissorTest, topoType);

    // Linear-probe lookup in hash table
    unsigned int startSlot = (unsigned int)(key % RL_D3D12_PSO_CACHE_SIZE);
    for (unsigned int i = 0; i < RL_D3D12_PSO_CACHE_SIZE; i++) {
        unsigned int slot = (startSlot + i) % RL_D3D12_PSO_CACHE_SIZE;
        if (RLGL.psoCache[slot].key == key) return RLGL.psoCache[slot].pso;  // Cache hit
        if (RLGL.psoCache[slot].key == 0) {
            // Empty slot — create PSO and cache it
            ID3D12PipelineState *pso = rlCreatePipelineState(
                RLGL.shaders[shaderIdx].vsBlob, RLGL.shaders[shaderIdx].psBlob,
                blendEnabled, blendMode, depthTest, depthWrite,
                backfaceCull, cullFaceMode, wireMode, scissorTest, topoType);
            if (pso) {
                RLGL.psoCache[slot].key = key;
                RLGL.psoCache[slot].pso = pso;
            }
            return pso;
        }
    }
    // Cache full (shouldn't happen), create without caching
    return rlCreatePipelineState(
        RLGL.shaders[shaderIdx].vsBlob, RLGL.shaders[shaderIdx].psBlob,
        blendEnabled, blendMode, depthTest, depthWrite,
        backfaceCull, cullFaceMode, wireMode, scissorTest, topoType);
}

static bool rlIsPsoCached(ID3D12PipelineState *pso)
{
    if (!pso) return false;
    for (unsigned int i = 0; i < RL_D3D12_PSO_CACHE_SIZE; i++) {
        if (RLGL.psoCache[i].key != 0 && RLGL.psoCache[i].pso == pso) return true;
    }
    return false;
}


static void rlInvalidatePsoCacheForShader(unsigned int shaderId)
{
    if (shaderId == 0) return;
    for (unsigned int i = 0; i < RL_D3D12_PSO_CACHE_SIZE; i++) {
        if (RLGL.psoCache[i].key != 0 && (RLGL.psoCache[i].key & 0xFFFFu) == shaderId) {
            if (RLGL.psoCache[i].pso) rlDeferPsoRelease(RLGL.psoCache[i].pso);
            ID3D12PipelineState *pso = RLGL.psoCache[i].pso;
            for (unsigned int s = 0; s < RLGL.shaderCount; s++) {
                if (RLGL.shaders[s].pso == pso) RLGL.shaders[s].pso = NULL;
                if (RLGL.shaders[s].psoLine == pso) RLGL.shaders[s].psoLine = NULL;
            }
            RLGL.psoCache[i].key = 0;
            RLGL.psoCache[i].pso = NULL;
        }
    }
}

static void rlClearPsoCache(void)
{
    for (unsigned int i = 0; i < RL_D3D12_PSO_CACHE_SIZE; i++) {
        if (RLGL.psoCache[i].key != 0) {
            ID3D12PipelineState *pso = RLGL.psoCache[i].pso;
            if (pso) {
                ID3D12PipelineState_Release(pso);
                for (unsigned int s = 0; s < RLGL.shaderCount; s++) {
                    if (RLGL.shaders[s].pso == pso) RLGL.shaders[s].pso = NULL;
                    if (RLGL.shaders[s].psoLine == pso) RLGL.shaders[s].psoLine = NULL;
                }
            }
            RLGL.psoCache[i].key = 0;
            RLGL.psoCache[i].pso = NULL;
        }
    }
}

//----------------------------------------------------------------------------------
// Helper: Re-establish graphics state after mesh draw PSO might have changed
// Ensures root signature, descriptor heaps, and viewport/scissor are set up
//----------------------------------------------------------------------------------
static void rlEnsureGraphicsStateForMesh(void)
{
    if (!RLGL.commandList) return;
    
    unsigned int shaderId = d3d12_activeShaderId > 0 ? d3d12_activeShaderId : RLGL.State.currentShaderId;
    if (shaderId == 0 || shaderId > RLGL.shaderCount) return;
    
    // Re-establish root signature and descriptor heaps
    ID3D12GraphicsCommandList_SetGraphicsRootSignature(RLGL.commandList, RLGL.rootSignature);
    ID3D12DescriptorHeap *heaps[] = { RLGL.srvHeap };
    ID3D12GraphicsCommandList_SetDescriptorHeaps(RLGL.commandList, 1, heaps);
    
    // Re-bind viewport and scissor (may have been cleared by PSO change)
    D3D12_VIEWPORT vp = { 0, 0, (float)RLGL.State.framebufferWidth, (float)RLGL.State.framebufferHeight, 0.0f, 1.0f };
    ID3D12GraphicsCommandList_RSSetViewports(RLGL.commandList, 1, &vp);
    if (RLGL.State.scissorTestEnabled) {
        D3D12_RECT sr = { RLGL.State.scissorX, RLGL.State.scissorY,
            RLGL.State.scissorX + RLGL.State.scissorWidth,
            RLGL.State.scissorY + RLGL.State.scissorHeight };
        ID3D12GraphicsCommandList_RSSetScissorRects(RLGL.commandList, 1, &sr);
    } else {
        D3D12_RECT sr = { 0, 0, RLGL.State.framebufferWidth, RLGL.State.framebufferHeight };
        ID3D12GraphicsCommandList_RSSetScissorRects(RLGL.commandList, 1, &sr);
    }
}

//----------------------------------------------------------------------------------
// Helper: Ensure current shader has a valid PSO for current render state
//----------------------------------------------------------------------------------
static void rlUpdateD3D12PipelineState(void)
{
    if (!RLGL.pipelineDirty) return;
    unsigned int shaderId = d3d12_activeShaderId > 0 ? d3d12_activeShaderId : RLGL.State.currentShaderId;
    if (shaderId == 0 || shaderId > RLGL.shaderCount) return;
    unsigned int shaderIdx = shaderId - 1;
    if (!RLGL.shaders[shaderIdx].vsBlob || !RLGL.shaders[shaderIdx].psBlob) return;

    // Look up or create PSOs for the current render state.
    // Defer-release old PSOs that were created during shader load with initial (depth=off)
    // state and are no longer needed — they are NOT in the PSO cache.
    {
        ID3D12PipelineState *oldPso = RLGL.shaders[shaderIdx].pso;
        RLGL.shaders[shaderIdx].pso = rlGetOrCreatePso(shaderId,
            RLGL.State.colorBlendEnabled, RLGL.State.currentBlendMode,
            RLGL.State.depthTestEnabled, RLGL.State.depthWriteEnabled,
            RLGL.State.backfaceCullingEnabled, RLGL.State.cullFaceMode,
            RLGL.State.wireMode, RLGL.State.scissorTestEnabled,
            D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE);
        if (oldPso && oldPso != RLGL.shaders[shaderIdx].pso && !rlIsPsoCached(oldPso))
            rlDeferPsoRelease(oldPso);
    }
    {
        ID3D12PipelineState *oldPsoLine = RLGL.shaders[shaderIdx].psoLine;
        RLGL.shaders[shaderIdx].psoLine = rlGetOrCreatePso(shaderId,
            RLGL.State.colorBlendEnabled, RLGL.State.currentBlendMode,
            RLGL.State.depthTestEnabled, RLGL.State.depthWriteEnabled,
            RLGL.State.backfaceCullingEnabled, RLGL.State.cullFaceMode,
            RLGL.State.wireMode, RLGL.State.scissorTestEnabled,
            D3D12_PRIMITIVE_TOPOLOGY_TYPE_LINE);
        if (oldPsoLine && oldPsoLine != RLGL.shaders[shaderIdx].psoLine && !rlIsPsoCached(oldPsoLine))
            rlDeferPsoRelease(oldPsoLine);
    }
    RLGL.pipelineDirty = false;
}

//================================================================================
// Matrix operations (CPU-side, identical to D3D11/OpenGL backend)
//================================================================================
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
    // OpenGL convention: column-major float[16] → raylib row-major Matrix
    Matrix mat = { matf[0],matf[4],matf[8],matf[12],
                   matf[1],matf[5],matf[9],matf[13],
                   matf[2],matf[6],matf[10],matf[14],
                   matf[3],matf[7],matf[11],matf[15] };
    *RLGL.State.currentMatrix = rlMatrixMultiply(mat, *RLGL.State.currentMatrix);
}

void rlFrustum(double left, double right, double bottom, double top, double znear, double zfar)
{
    Matrix matFrustum = { 0 };
    float rl = (float)(right - left);
    float tb = (float)(top - bottom);
    float fn = (float)(zfar - znear);
    matFrustum.m0 = ((float)znear*2.0f)/rl;
    matFrustum.m5 = ((float)znear*2.0f)/tb;
    matFrustum.m8 = ((float)right + (float)left)/rl;
    matFrustum.m9 = ((float)top + (float)bottom)/tb;
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
    matOrtho.m12 = -((float)left + (float)right)/rl;
    matOrtho.m13 = -((float)top + (float)bottom)/tb;
    matOrtho.m14 = -((float)zfar + (float)znear)/fn;
    matOrtho.m15 = 1.0f;
    *RLGL.State.currentMatrix = rlMatrixMultiply(*RLGL.State.currentMatrix, matOrtho);
}

void rlViewport(int x, int y, int width, int height)
{
    RLGL.State.framebufferWidth = width;
    RLGL.State.framebufferHeight = height;

    // Also set viewport immediately on the command list (like D3D11)
    if (RLGL.commandList) {
        D3D12_VIEWPORT vp = { (FLOAT)x, (FLOAT)y, (FLOAT)width, (FLOAT)height, 0.0f, 1.0f };
        ID3D12GraphicsCommandList_RSSetViewports(RLGL.commandList, 1, &vp);
    }
}

void rlSetClipPlanes(double nearPlane, double farPlane)
{
    rlCullDistanceNear = nearPlane;
    rlCullDistanceFar = farPlane;
}

double rlGetCullDistanceNear(void) { return rlCullDistanceNear; }
double rlGetCullDistanceFar(void) { return rlCullDistanceFar; }

//================================================================================
// Vertex-level operations
//================================================================================
void rlBegin(int mode)
{
    if (RLGL.currentBatch->draws[RLGL.currentBatch->drawCounter - 1].mode != mode) {
        if (RLGL.currentBatch->draws[RLGL.currentBatch->drawCounter - 1].vertexCount > 0) {
            if (RLGL.currentBatch->draws[RLGL.currentBatch->drawCounter - 1].mode == RL_LINES)
                RLGL.currentBatch->draws[RLGL.currentBatch->drawCounter - 1].vertexAlignment =
                    ((RLGL.currentBatch->draws[RLGL.currentBatch->drawCounter - 1].vertexCount < 4) ?
                     RLGL.currentBatch->draws[RLGL.currentBatch->drawCounter - 1].vertexCount :
                     RLGL.currentBatch->draws[RLGL.currentBatch->drawCounter - 1].vertexCount % 4);
            else if (RLGL.currentBatch->draws[RLGL.currentBatch->drawCounter - 1].mode == RL_TRIANGLES)
                RLGL.currentBatch->draws[RLGL.currentBatch->drawCounter - 1].vertexAlignment =
                    ((RLGL.currentBatch->draws[RLGL.currentBatch->drawCounter - 1].vertexCount < 4) ?
                     1 : (4 - (RLGL.currentBatch->draws[RLGL.currentBatch->drawCounter - 1].vertexCount % 4)));
            else RLGL.currentBatch->draws[RLGL.currentBatch->drawCounter - 1].vertexAlignment = 0;

            if (!rlCheckRenderBatchLimit(RLGL.currentBatch->draws[RLGL.currentBatch->drawCounter - 1].vertexAlignment)) {
                RLGL.State.vertexCounter += RLGL.currentBatch->draws[RLGL.currentBatch->drawCounter - 1].vertexAlignment;
                RLGL.currentBatch->drawCounter++;
            }
        }
        if (RLGL.currentBatch->drawCounter >= RL_DEFAULT_BATCH_DRAWCALLS) rlDrawRenderBatch(RLGL.currentBatch);
        RLGL.currentBatch->draws[RLGL.currentBatch->drawCounter - 1].mode = mode;
        RLGL.currentBatch->draws[RLGL.currentBatch->drawCounter - 1].vertexCount = 0;
        RLGL.currentBatch->draws[RLGL.currentBatch->drawCounter - 1].textureId = RLGL.State.defaultTextureId;
    }
}

void rlEnd(void)
{
    // Advance depth for each primitive to avoid Z-fighting on coplanar geometry
    RLGL.currentBatch->currentDepth += (1.0f/20000.0f);
}

void rlVertex3f(float x, float y, float z)
{
    float tx = x, ty = y, tz = z;
    if (RLGL.State.transformRequired) {
        tx = RLGL.State.transform.m0*x + RLGL.State.transform.m4*y + RLGL.State.transform.m8*z + RLGL.State.transform.m12;
        ty = RLGL.State.transform.m1*x + RLGL.State.transform.m5*y + RLGL.State.transform.m9*z + RLGL.State.transform.m13;
        tz = RLGL.State.transform.m2*x + RLGL.State.transform.m6*y + RLGL.State.transform.m10*z + RLGL.State.transform.m14;
    }

    // Check buffer overflow and auto-flush if needed (matching D3D11 behavior)
    if (RLGL.State.vertexCounter > (RLGL.currentBatch->vertexBuffer[RLGL.currentBatch->currentBuffer].elementCount*4 - 4)) {
        if ((RLGL.currentBatch->draws[RLGL.currentBatch->drawCounter - 1].mode == RL_LINES) &&
            (RLGL.currentBatch->draws[RLGL.currentBatch->drawCounter - 1].vertexCount%2 == 0))
            rlCheckRenderBatchLimit(2 + 1);
        else if ((RLGL.currentBatch->draws[RLGL.currentBatch->drawCounter - 1].mode == RL_TRIANGLES) &&
            (RLGL.currentBatch->draws[RLGL.currentBatch->drawCounter - 1].vertexCount%3 == 0))
            rlCheckRenderBatchLimit(3 + 1);
        else if ((RLGL.currentBatch->draws[RLGL.currentBatch->drawCounter - 1].mode == RL_QUADS) &&
            (RLGL.currentBatch->draws[RLGL.currentBatch->drawCounter - 1].vertexCount%4 == 0))
            rlCheckRenderBatchLimit(4 + 1);
    }

    RLGL.currentBatch->vertexBuffer[RLGL.currentBatch->currentBuffer].vertices[3*RLGL.State.vertexCounter] = tx;
    RLGL.currentBatch->vertexBuffer[RLGL.currentBatch->currentBuffer].vertices[3*RLGL.State.vertexCounter + 1] = ty;
    RLGL.currentBatch->vertexBuffer[RLGL.currentBatch->currentBuffer].vertices[3*RLGL.State.vertexCounter + 2] = tz;

    RLGL.currentBatch->vertexBuffer[RLGL.currentBatch->currentBuffer].texcoords[2*RLGL.State.vertexCounter] = RLGL.State.texcoordx;
    RLGL.currentBatch->vertexBuffer[RLGL.currentBatch->currentBuffer].texcoords[2*RLGL.State.vertexCounter + 1] = RLGL.State.texcoordy;

    RLGL.currentBatch->vertexBuffer[RLGL.currentBatch->currentBuffer].normals[3*RLGL.State.vertexCounter] = RLGL.State.normalx;
    RLGL.currentBatch->vertexBuffer[RLGL.currentBatch->currentBuffer].normals[3*RLGL.State.vertexCounter + 1] = RLGL.State.normaly;
    RLGL.currentBatch->vertexBuffer[RLGL.currentBatch->currentBuffer].normals[3*RLGL.State.vertexCounter + 2] = RLGL.State.normalz;

    RLGL.currentBatch->vertexBuffer[RLGL.currentBatch->currentBuffer].colors[4*RLGL.State.vertexCounter] = RLGL.State.colorr;
    RLGL.currentBatch->vertexBuffer[RLGL.currentBatch->currentBuffer].colors[4*RLGL.State.vertexCounter + 1] = RLGL.State.colorg;
    RLGL.currentBatch->vertexBuffer[RLGL.currentBatch->currentBuffer].colors[4*RLGL.State.vertexCounter + 2] = RLGL.State.colorb;
    RLGL.currentBatch->vertexBuffer[RLGL.currentBatch->currentBuffer].colors[4*RLGL.State.vertexCounter + 3] = RLGL.State.colora;

    RLGL.State.vertexCounter++;
    RLGL.currentBatch->draws[RLGL.currentBatch->drawCounter - 1].vertexCount++;
}

void rlVertex2f(float x, float y) { rlVertex3f(x, y, RLGL.currentBatch->currentDepth); }
void rlVertex2i(int x, int y) { rlVertex3f((float)x, (float)y, RLGL.currentBatch->currentDepth); }
void rlTexCoord2f(float x, float y) { RLGL.State.texcoordx = x; RLGL.State.texcoordy = y; }
void rlNormal3f(float x, float y, float z) {
    float nx = x, ny = y, nz = z;
    if (RLGL.State.transformRequired) {
        nx = RLGL.State.transform.m0*x + RLGL.State.transform.m4*y + RLGL.State.transform.m8*z;
        ny = RLGL.State.transform.m1*x + RLGL.State.transform.m5*y + RLGL.State.transform.m9*z;
        nz = RLGL.State.transform.m2*x + RLGL.State.transform.m6*y + RLGL.State.transform.m10*z;
    }
    float len = sqrtf(nx*nx + ny*ny + nz*nz);
    if (len != 0.0f) { float il = 1.0f/len; nx *= il; ny *= il; nz *= il; }
    RLGL.State.normalx = nx; RLGL.State.normaly = ny; RLGL.State.normalz = nz;
}

void rlColor4ub(unsigned char r, unsigned char g, unsigned char b, unsigned char a) {
    RLGL.State.colorr = r; RLGL.State.colorg = g; RLGL.State.colorb = b; RLGL.State.colora = a;
}
void rlColor4f(float r, float g, float b, float a) {
    rlColor4ub((unsigned char)(r*255), (unsigned char)(g*255), (unsigned char)(b*255), (unsigned char)(a*255));
}
void rlColor3f(float x, float y, float z) { rlColor4ub((unsigned char)(x*255), (unsigned char)(y*255), (unsigned char)(z*255), 255); }
//================================================================================
// Texture binding (mesh-aware with material map register mapping)
//================================================================================
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
                        ((RLGL.currentBatch->draws[RLGL.currentBatch->drawCounter - 1].vertexCount < 4) ?
                         RLGL.currentBatch->draws[RLGL.currentBatch->drawCounter - 1].vertexCount :
                         RLGL.currentBatch->draws[RLGL.currentBatch->drawCounter - 1].vertexCount % 4);
                else if (RLGL.currentBatch->draws[RLGL.currentBatch->drawCounter - 1].mode == RL_TRIANGLES)
                    RLGL.currentBatch->draws[RLGL.currentBatch->drawCounter - 1].vertexAlignment =
                        ((RLGL.currentBatch->draws[RLGL.currentBatch->drawCounter - 1].vertexCount < 4) ?
                         1 : (4 - (RLGL.currentBatch->draws[RLGL.currentBatch->drawCounter - 1].vertexCount % 4)));
                else RLGL.currentBatch->draws[RLGL.currentBatch->drawCounter - 1].vertexAlignment = 0;

                if (!rlCheckRenderBatchLimit(RLGL.currentBatch->draws[RLGL.currentBatch->drawCounter - 1].vertexAlignment)) {
                    RLGL.State.vertexCounter += RLGL.currentBatch->draws[RLGL.currentBatch->drawCounter - 1].vertexAlignment;
                    RLGL.currentBatch->drawCounter++;
                }
            }
            if (RLGL.currentBatch->drawCounter >= RL_DEFAULT_BATCH_DRAWCALLS) rlDrawRenderBatch(RLGL.currentBatch);
            RLGL.currentBatch->draws[RLGL.currentBatch->drawCounter - 1].textureId = id;
            RLGL.currentBatch->draws[RLGL.currentBatch->drawCounter - 1].vertexCount = 0;
        }
    }
}

void rlActiveTextureSlot(int slot) { d3d12_activeTextureSlot = slot; }

void rlEnableTexture(unsigned int id) {
    if (id > 0 && id <= RLGL.textureCount && RLGL.commandList) {
        // Determine the actual D3D12 register slot using per-shader mapping
        int targetSlot = d3d12_activeTextureSlot;  // Default: identity mapping
        if (d3d12_activeShaderId > 0 && d3d12_activeShaderId <= RLGL.shaderCount) {
            unsigned int shaderIdx = d3d12_activeShaderId - 1;
            if (d3d12_activeTextureSlot >= 0 && d3d12_activeTextureSlot < RL_MAX_MATERIAL_MAPS) {
                targetSlot = RLGL.shaders[shaderIdx].texMaterialMapToRegister[d3d12_activeTextureSlot];
            }
        }
        // In D3D12, we use the descriptor table at the SRV root param.
        // For mesh rendering, we bind the texture SRV directly via the descriptor table.
        // The SRV heap already has all textures; we set the descriptor table to point to the right one.
        // Note: For multi-texture, we'd need to copy descriptors to a contiguous range.
        ID3D12GraphicsCommandList_SetGraphicsRootDescriptorTable(RLGL.commandList, RL_D3D12_ROOT_SRV_TABLE_IDX,
            RLGL.textures[id - 1].srvGpuHandle);
        RLGL.State.activeTextureId[0] = id;
    }
}

void rlDisableTexture(void) {
    if (RLGL.commandList && RLGL.State.defaultTextureId > 0 && RLGL.State.defaultTextureId <= RLGL.textureCount) {
        ID3D12GraphicsCommandList_SetGraphicsRootDescriptorTable(RLGL.commandList, RL_D3D12_ROOT_SRV_TABLE_IDX,
            RLGL.textures[RLGL.State.defaultTextureId - 1].srvGpuHandle);
    }
    RLGL.State.activeTextureId[0] = RLGL.State.defaultTextureId;
}

void rlEnableTextureCubemap(unsigned int id) { rlEnableTexture(id); }
void rlDisableTextureCubemap(void) { rlDisableTexture(); }
void rlTextureParameters(unsigned int id, int param, int value) { (void)id; (void)param; (void)value; }
void rlCubemapParameters(unsigned int id, int param, int value) { (void)id; (void)param; (void)value; }

//================================================================================
// Shader enable/disable (mesh-aware with cbuffer re-binding)
//================================================================================
void rlEnableShader(unsigned int id) {
    if (id > 0 && id <= RLGL.shaderCount && RLGL.commandList) {
        // Use d3d12_lastEnabledShader (not d3d12_activeShaderId) for the change check:
        // rlDisableShader resets d3d12_activeShaderId to 0 between every DrawMesh call,
        // which would make shaderChanged always true and defeat the optimization.
        // d3d12_lastEnabledShader persists across rlDisableShader calls.
        bool shaderChanged = (d3d12_lastEnabledShader != id);
        d3d12_lastEnabledShader = id;
        d3d12_activeShaderId = id;  // Track which shader is actually bound on GPU
        unsigned int idx = id - 1;

        // PSO lookup: always resolve the PSO to guarantee correctness.  The hash
        // table lookup is O(1) and the real perf savings come from d3d12_baseStateSet
        // (skipping 6+ API calls) and d3d12_lastBoundPSO (skipping SetPipelineState).
        // Skip the line-topology PSO — only the batch path needs it and handles it via
        // rlUpdateD3D12PipelineState() independently.
        {
            ID3D12PipelineState *oldPso = RLGL.shaders[idx].pso;
            RLGL.shaders[idx].pso = rlGetOrCreatePso(id,
                RLGL.State.colorBlendEnabled, RLGL.State.currentBlendMode,
                RLGL.State.depthTestEnabled, RLGL.State.depthWriteEnabled,
                RLGL.State.backfaceCullingEnabled, RLGL.State.cullFaceMode,
                RLGL.State.wireMode, RLGL.State.scissorTestEnabled,
                D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE);
            // Defer-release old PSO if it was the initial depth-off one (not in cache)
            if (oldPso && oldPso != RLGL.shaders[idx].pso && !rlIsPsoCached(oldPso))
                rlDeferPsoRelease(oldPso);
            RLGL.pipelineDirty = false;
        }

        // Bind PSO only when it actually changed (avoids redundant driver call for 10K+ objects)
        if (RLGL.shaders[idx].pso && RLGL.shaders[idx].pso != d3d12_lastBoundPSO) {
            ID3D12GraphicsCommandList_SetPipelineState(RLGL.commandList, RLGL.shaders[idx].pso);
            d3d12_lastBoundPSO = RLGL.shaders[idx].pso;
        }

        // Base D3D12 state: root signature, descriptor heaps, viewport, scissor,
        // primitive topology, and render target.  These are mandatory after a command
        // list Reset (which clears ALL state) but PERSIST across draw calls on the
        // same command list.  Setting them once and tracking via d3d12_baseStateSet
        // eliminates ~6 redundant driver calls per object in 10K+ object scenes.
        if (!d3d12_baseStateSet) {
            ID3D12GraphicsCommandList_SetGraphicsRootSignature(RLGL.commandList, RLGL.rootSignature);
            ID3D12DescriptorHeap *heaps[] = { RLGL.srvHeap };
            ID3D12GraphicsCommandList_SetDescriptorHeaps(RLGL.commandList, 1, heaps);

            {
                D3D12_VIEWPORT vp = { 0, 0, (float)RLGL.State.framebufferWidth, (float)RLGL.State.framebufferHeight, 0.0f, 1.0f };
                ID3D12GraphicsCommandList_RSSetViewports(RLGL.commandList, 1, &vp);
                if (RLGL.State.scissorTestEnabled) {
                    D3D12_RECT sr = { RLGL.State.scissorX, RLGL.State.scissorY,
                        RLGL.State.scissorX + RLGL.State.scissorWidth,
                        RLGL.State.scissorY + RLGL.State.scissorHeight };
                    ID3D12GraphicsCommandList_RSSetScissorRects(RLGL.commandList, 1, &sr);
                } else {
                    D3D12_RECT sr = { 0, 0, RLGL.State.framebufferWidth, RLGL.State.framebufferHeight };
                    ID3D12GraphicsCommandList_RSSetScissorRects(RLGL.commandList, 1, &sr);
                }
                ID3D12GraphicsCommandList_IASetPrimitiveTopology(RLGL.commandList, D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            }

            // Bind the correct render target + depth stencil
            if (RLGL.activeFramebuffer > 0 && RLGL.activeFramebuffer <= RLGL.framebufferCount) {
                rlD3D12Framebuffer *fb = &RLGL.framebuffers[RLGL.activeFramebuffer - 1];
                if (fb->colorCount > 0 && fb->colorTexIds[0] > 0 && fb->colorTexIds[0] <= RLGL.textureCount) {
                    D3D12_CPU_DESCRIPTOR_HANDLE rtvH = RLGL.textures[fb->colorTexIds[0] - 1].rtvHandle;
                    D3D12_CPU_DESCRIPTOR_HANDLE *dsvPtr = NULL;
                    D3D12_CPU_DESCRIPTOR_HANDLE dsvH = { 0 };
                    if (fb->depthTexId > 0 && fb->depthTexId <= RLGL.textureCount) {
                        dsvH = RLGL.textures[fb->depthTexId - 1].dsvHandle;
                        dsvPtr = &dsvH;
                    }
                    ID3D12GraphicsCommandList_OMSetRenderTargets(RLGL.commandList, 1, &rtvH, FALSE, dsvPtr);
                }
            } else if (RLGL.swapChain) {
                UINT backIdx = d3d12_cachedBackBufferIndex;  // Use cached index (updated in rlClearScreenBuffers)
                if (RLGL.renderTargets[backIdx]) {
                    ID3D12GraphicsCommandList_OMSetRenderTargets(RLGL.commandList, 1,
                        &RLGL.renderTargetHandles[backIdx], FALSE,
                        RLGL.depthStencilBuffer ? &RLGL.dsvBackbuffer : NULL);
                }
            }

            d3d12_baseStateSet = true;
        }

        // Always mark ALL constant buffers dirty.  In D3D12, cbuffer data lives in
        // a per-frame ring buffer — addresses from a previous command list are invalid
        // after Reset().  Unlike D3D11 (which has persistent cbuffer objects that stay
        // bound across draw calls), D3D12 root CBV bindings must be re-established
        // before every draw.  Re-suballocating 8 small cbuffers (≤256 B each) per
        // object is cheap compared to the cost of a missed binding causing wrong data.
        {
            rlD3D12Shader *s = &RLGL.shaders[idx];
            for (int i = 0; i < s->cbufferCount; i++) {
                s->cbuffers[i].dirty = true;
            }
        }
    }
}

void rlDisableShader(void) {
    d3d12_activeShaderId = 0;
    // NOTE: We intentionally do NOT restore the default shader PSO here.
    // The batch path handles its own PSO via rlUpdateD3D12PipelineState(),
    // and the next rlEnableShader will set the correct mesh PSO.
    // Avoiding the redundant SetPipelineState saves 10K+ driver calls in
    // scenes with many mesh objects.
}

//================================================================================
// Framebuffer state
//================================================================================
void rlEnableFramebuffer(unsigned int id) {
    if (RLGL.activeFramebuffer != id) {
        RLGL.activeFramebuffer = id;
        d3d12_baseStateSet = false;  // Force RT re-bind on next draw
    }
}
void rlDisableFramebuffer(void) {
    if (RLGL.activeFramebuffer != 0) {
        RLGL.activeFramebuffer = 0;
        d3d12_baseStateSet = false;  // Force RT re-bind on next draw
    }
}
unsigned int rlGetActiveFramebuffer(void) { return RLGL.activeFramebuffer; }
void rlActiveDrawBuffers(int count) { (void)count; }
void rlBlitFramebuffer(int srcX, int srcY, int srcWidth, int srcHeight,
    int dstX, int dstY, int dstWidth, int dstHeight, int bufferMask) {
    (void)srcX; (void)srcY; (void)srcWidth; (void)srcHeight;
    (void)dstX; (void)dstY; (void)dstWidth; (void)dstHeight; (void)bufferMask;
}
void rlBindFramebuffer(unsigned int target, unsigned int framebuffer) {
    (void)target;
    unsigned int newFB = (framebuffer == 0) ? 0 : framebuffer;
    if (RLGL.activeFramebuffer != newFB) {
        RLGL.activeFramebuffer = newFB;
        d3d12_baseStateSet = false;  // Force RT re-bind on next draw
    }
}

//================================================================================
// Render state toggles
//================================================================================
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
    // Save scissor rect so rlDrawRenderBatch can re-apply it after command list Reset
    RLGL.State.scissorX = x;
    RLGL.State.scissorY = y;
    RLGL.State.scissorWidth = width;
    RLGL.State.scissorHeight = height;
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
    if (!RLGL.commandList) return;
    // Determine which RTV/DSV to clear
    D3D12_CPU_DESCRIPTOR_HANDLE rtvHandle = { 0 };
    D3D12_CPU_DESCRIPTOR_HANDLE dsvHandle = { 0 };
    bool hasRTV = false, hasDSV = false;

    if (RLGL.activeFramebuffer > 0 && RLGL.activeFramebuffer <= RLGL.framebufferCount) {
        rlD3D12Framebuffer *fb = &RLGL.framebuffers[RLGL.activeFramebuffer - 1];
        if (fb->colorCount > 0 && fb->colorTexIds[0] > 0 && fb->colorTexIds[0] <= RLGL.textureCount) {
            rtvHandle = RLGL.textures[fb->colorTexIds[0] - 1].rtvHandle;
            hasRTV = true;
        }
        if (fb->depthTexId > 0 && fb->depthTexId <= RLGL.textureCount) {
            dsvHandle = RLGL.textures[fb->depthTexId - 1].dsvHandle;
            hasDSV = true;
        }
    } else {
        // Use swap chain backbuffer RTV/DSV
        if (RLGL.swapChain) {
            UINT backIdx = IDXGISwapChain3_GetCurrentBackBufferIndex(RLGL.swapChain);
            d3d12_cachedBackBufferIndex = backIdx;  // Cache once per frame for per-draw hot paths
            if (RLGL.renderTargets[backIdx]) {
                // Transition from PRESENT to RENDER_TARGET so we can clear and draw into it
                rlTransitionResource(RLGL.renderTargets[backIdx],
                    D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_RENDER_TARGET);
                rtvHandle = RLGL.renderTargetHandles[backIdx];
                hasRTV = true;
                // Bind RTV (and optionally DSV) for the clear and subsequent draw calls
                D3D12_CPU_DESCRIPTOR_HANDLE dsvPtr = RLGL.dsvBackbuffer;
                ID3D12GraphicsCommandList_OMSetRenderTargets(RLGL.commandList, 1, &rtvHandle, FALSE,
                    RLGL.depthStencilBuffer ? &dsvPtr : NULL);
            }
        }
        dsvHandle = RLGL.dsvBackbuffer;
        hasDSV = (RLGL.depthStencilBuffer != NULL);
    }

    if (hasRTV) ID3D12GraphicsCommandList_ClearRenderTargetView(RLGL.commandList, rtvHandle, RLGL.State.clearColor, 0, NULL);
    // D32_FLOAT has no stencil component — using D3D12_CLEAR_FLAG_STENCIL is undefined behaviour.
    if (hasDSV) ID3D12GraphicsCommandList_ClearDepthStencilView(RLGL.commandList, dsvHandle, D3D12_CLEAR_FLAG_DEPTH, 1.0f, 0, 0, NULL);
}

void rlSwapScreenBuffer(void) {
    if (!RLGL.commandList || !RLGL.commandQueue) return;

    // Transition the swap chain back-buffer from RENDER_TARGET → PRESENT before submitting
    if (RLGL.activeFramebuffer == 0 && RLGL.swapChain) {
        UINT backIdx = IDXGISwapChain3_GetCurrentBackBufferIndex(RLGL.swapChain);
        if (RLGL.renderTargets[backIdx]) {
            rlTransitionResource(RLGL.renderTargets[backIdx],
                D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_PRESENT);
        }
    }

    // Close and execute the command list built up during this frame
    HRESULT hr = ID3D12GraphicsCommandList_Close(RLGL.commandList);
    if (SUCCEEDED(hr)) {
        ID3D12CommandList *lists[] = { (ID3D12CommandList *)RLGL.commandList };
        ID3D12CommandQueue_ExecuteCommandLists(RLGL.commandQueue, 1, lists);
    }

    // Present
    if (RLGL.swapChain) IDXGISwapChain3_Present(RLGL.swapChain, 0, 0);

    // Signal the GPU using the global monotonically-increasing fence value,
    // then record which fence value this frame was submitted at.
    UINT64 signalValue = ++RLGL.fenceValue;
    rlD3D12FrameResources *frameJustSubmitted = &RLGL.frames[RLGL.currentFrame];
    frameJustSubmitted->fenceValue = signalValue;
    ID3D12CommandQueue_Signal(RLGL.commandQueue, RLGL.fence, signalValue);

    // Advance to the next frame slot
    RLGL.currentFrame = (RLGL.currentFrame + 1) % RL_D3D12_NUM_FRAMES;
    rlD3D12FrameResources *nextFrame = &RLGL.frames[RLGL.currentFrame];

    // Wait if the GPU is still using the next frame's resources
    if (ID3D12Fence_GetCompletedValue(RLGL.fence) < nextFrame->fenceValue) {
        ID3D12Fence_SetEventOnCompletion(RLGL.fence, nextFrame->fenceValue, RLGL.fenceEvent);
        WaitForSingleObject(RLGL.fenceEvent, INFINITE);
    }

    // Release deferred PSOs whose fence value has been reached
    rlFlushDeferredPsoReleases();

    // Release pending upload buffers that the GPU has now consumed
    rlReleasePendingUploads(nextFrame);

    // Reset sub-allocation offsets for the new frame
    nextFrame->batchCBSlot = 0;
    nextFrame->batchVBOffset = 0;
    nextFrame->cbRingOffset = 0;

    // Reset VB/IB tracking for the new frame
    d3d12_hasVBBound = false;
    d3d12_hasIBBound = false;

    // Reset the command allocator and list for the new frame
    ID3D12CommandAllocator_Reset(nextFrame->commandAllocator);
    ID3D12GraphicsCommandList_Reset(RLGL.commandList, nextFrame->commandAllocator, NULL);
    d3d12_baseStateSet = false;
    d3d12_lastBoundPSO = NULL;
    d3d12_lastEnabledShader = 0;
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
//================================================================================
// Backend initialization
//================================================================================
void rlglInit(int width, int height, bool headless)
{
    (void)headless;

    // Create DXGI factory
    UINT dxgiFlags = 0;
#if defined(_DEBUG) || defined(RL_D3D12_ENABLE_DEBUG_LAYER)
    {
        ID3D12Debug *debugController = NULL;
        if (SUCCEEDED(D3D12GetDebugInterface(&IID_ID3D12Debug, (void **)&debugController))) {
            ID3D12Debug_EnableDebugLayer(debugController);
            dxgiFlags |= DXGI_CREATE_FACTORY_DEBUG;
            ID3D12Debug_Release(debugController);
            TRACELOG(RL_LOG_INFO, "D3D12: Debug layer enabled — validation errors will appear in debug output");
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
    if (FAILED(hr)) { TRACELOG(RL_LOG_ERROR, "D3D12: Failed to create command queue"); return; }

    // Create descriptor heaps
    {
        D3D12_DESCRIPTOR_HEAP_DESC heapDesc = { 0 };
        heapDesc.NumDescriptors = RL_D3D12_MAX_SRV_DESC;
        heapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        heapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
        ID3D12Device_CreateDescriptorHeap(RLGL.device, &heapDesc, &IID_ID3D12DescriptorHeap, (void **)&RLGL.srvHeap);

        heapDesc.NumDescriptors = 64;
        heapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
        heapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
        ID3D12Device_CreateDescriptorHeap(RLGL.device, &heapDesc, &IID_ID3D12DescriptorHeap, (void **)&RLGL.rtvHeap);

        heapDesc.NumDescriptors = 64;
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

        UINT cbMatrixSize = ((sizeof(rlD3D12MatrixCB) + 255) & ~255) * RL_D3D12_MAX_BATCH_DRAWS_PER_FRAME;
        UINT cbColorSize = ((sizeof(rlD3D12ColorCB) + 255) & ~255) * RL_D3D12_MAX_BATCH_DRAWS_PER_FRAME;

        frame->cbMatrixUpload = rlCreateUploadBuffer(cbMatrixSize);
        frame->cbColorUpload = rlCreateUploadBuffer(cbColorSize);

        D3D12_RANGE readRange = { 0, 0 };
        if (frame->cbMatrixUpload) ID3D12Resource_Map(frame->cbMatrixUpload, 0, &readRange, &frame->cbMatrixMapped);
        if (frame->cbColorUpload) ID3D12Resource_Map(frame->cbColorUpload, 0, &readRange, &frame->cbColorMapped);

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

        // Per-frame constant buffer ring buffer for mesh shader uniforms
        frame->cbRingBuffer = rlCreateUploadBuffer(RL_D3D12_CB_RING_SIZE);
        if (frame->cbRingBuffer)
            ID3D12Resource_Map(frame->cbRingBuffer, 0, &readRange, &frame->cbRingMapped);
        frame->cbRingOffset = 0;

        frame->fenceValue = 0;
    }

    RLGL.currentFrame = 0;
    RLGL.batchBufferSize = totalVertices;

    // Create command list
    hr = ID3D12Device_CreateCommandList(RLGL.device, 0, D3D12_COMMAND_LIST_TYPE_DIRECT,
        RLGL.frames[0].commandAllocator, NULL, &IID_ID3D12GraphicsCommandList, (void **)&RLGL.commandList);
    if (FAILED(hr)) { TRACELOG(RL_LOG_ERROR, "D3D12: Failed to create command list"); return; }

    // Set descriptor heap
    ID3D12DescriptorHeap *heaps[] = { RLGL.srvHeap };
    ID3D12GraphicsCommandList_SetDescriptorHeaps(RLGL.commandList, 1, heaps);

    // Create swap chain if we have a window
    {
        HWND hwnd = GetActiveWindow();
        if (hwnd && RLGL.factory && RLGL.commandQueue) {
            DXGI_SWAP_CHAIN_DESC1 scDesc = { 0 };
            scDesc.Width = width;
            scDesc.Height = height;
            scDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
            scDesc.SampleDesc.Count = 1;
            scDesc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
            scDesc.BufferCount = RL_D3D12_NUM_FRAMES;
            scDesc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
            scDesc.Flags = 0;

            IDXGISwapChain1 *sc1 = NULL;
            hr = IDXGIFactory4_CreateSwapChainForHwnd(RLGL.factory, (IUnknown *)RLGL.commandQueue,
                hwnd, &scDesc, NULL, NULL, &sc1);
            if (SUCCEEDED(hr) && sc1) {
                hr = IDXGISwapChain1_QueryInterface(sc1, &IID_IDXGISwapChain3, (void **)&RLGL.swapChain);
                IDXGISwapChain1_Release(sc1);
            }
            if (FAILED(hr) || !RLGL.swapChain) {
                TRACELOG(RL_LOG_WARNING, "D3D12: Failed to create swap chain (hr=0x%08X). Continuing without.", hr);
            } else {
                // Disable ALT+ENTER fullscreen toggle
                IDXGIFactory4_MakeWindowAssociation(RLGL.factory, hwnd, DXGI_MWA_NO_ALT_ENTER);

                // Create RTVs for swap chain backbuffers
                D3D12_CPU_DESCRIPTOR_HANDLE rtvStart;
                ID3D12DescriptorHeap_GetCPUDescriptorHandleForHeapStart(RLGL.rtvHeap, &rtvStart);
                RLGL.swapChainRTVBase = RLGL.rtvHeapUsed;

                for (int i = 0; i < RL_D3D12_NUM_FRAMES; i++) {
                    hr = IDXGISwapChain3_GetBuffer(RLGL.swapChain, i, &IID_ID3D12Resource, (void **)&RLGL.renderTargets[i]);
                    if (SUCCEEDED(hr)) {
                        RLGL.renderTargetHandles[i].ptr = rtvStart.ptr + (SIZE_T)((RLGL.rtvHeapUsed++) * RLGL.rtvDescriptorSize);
                        ID3D12Device_CreateRenderTargetView(RLGL.device, RLGL.renderTargets[i], NULL, RLGL.renderTargetHandles[i]);
                    }
                }

                // Create depth-stencil buffer for swap chain
                D3D12_RESOURCE_DESC dsDesc = { 0 };
                dsDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
                dsDesc.Width = width;
                dsDesc.Height = height;
                dsDesc.DepthOrArraySize = 1;
                dsDesc.MipLevels = 1;
                dsDesc.Format = DXGI_FORMAT_D32_FLOAT;
                dsDesc.SampleDesc.Count = 1;
                dsDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
                D3D12_HEAP_PROPERTIES dsHeap = { 0 };
                dsHeap.Type = D3D12_HEAP_TYPE_DEFAULT;
                D3D12_CLEAR_VALUE dsClear = { 0 };
                dsClear.Format = DXGI_FORMAT_D32_FLOAT;
                dsClear.DepthStencil.Depth = 1.0f;
                hr = ID3D12Device_CreateCommittedResource(RLGL.device, &dsHeap, D3D12_HEAP_FLAG_NONE,
                    &dsDesc, D3D12_RESOURCE_STATE_DEPTH_WRITE, &dsClear,
                    &IID_ID3D12Resource, (void **)&RLGL.depthStencilBuffer);
                if (SUCCEEDED(hr)) {
                    D3D12_CPU_DESCRIPTOR_HANDLE dsvStart;
                    ID3D12DescriptorHeap_GetCPUDescriptorHandleForHeapStart(RLGL.dsvHeap, &dsvStart);
                    RLGL.swapChainDSVIndex = RLGL.dsvHeapUsed++;
                    RLGL.dsvBackbuffer.ptr = dsvStart.ptr + (SIZE_T)(RLGL.swapChainDSVIndex * RLGL.dsvDescriptorSize);
                    D3D12_DEPTH_STENCIL_VIEW_DESC dsvDesc = { 0 };
                    dsvDesc.Format = DXGI_FORMAT_D32_FLOAT;
                    dsvDesc.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2D;
                    ID3D12Device_CreateDepthStencilView(RLGL.device, RLGL.depthStencilBuffer, &dsvDesc, RLGL.dsvBackbuffer);
                    TRACELOG(RL_LOG_INFO, "D3D12: Depth buffer created (%dx%d, D32_FLOAT)", width, height);
                } else {
                    TRACELOG(RL_LOG_ERROR, "D3D12: FAILED to create depth buffer (hr=0x%08X) — depth test will not work!", hr);
                }

                TRACELOG(RL_LOG_INFO, "D3D12: Swap chain created (%dx%d, %d buffers)", width, height, RL_D3D12_NUM_FRAMES);
            }
        }
    }

    // Create default white texture
    {
        unsigned char pixels[4] = { 255, 255, 255, 255 };
        RLGL.State.defaultTextureId = rlLoadTexture(pixels, 1, 1, 7, 1);
        if (RLGL.State.defaultTextureId != 0)
            TRACELOG(RL_LOG_INFO, "TEXTURE: [ID %i] Default texture loaded successfully", RLGL.State.defaultTextureId);
    }

    // Create default white vertex buffer for meshes without vertex colors
    {
        UINT whiteSize = RL_D3D12_DEFAULT_WHITE_VBO_VERTS * 4;
        d3d12_defaultWhiteVBO = rlCreateUploadBuffer(whiteSize);
        if (d3d12_defaultWhiteVBO) {
            void *mapped = NULL;
            D3D12_RANGE readRange = { 0, 0 };
            ID3D12Resource_Map(d3d12_defaultWhiteVBO, 0, &readRange, &mapped);
            if (mapped) {
                memset(mapped, 0xFF, whiteSize);  // All 0xFF = white RGBA
                // Don't unmap - keep persistently mapped
            }
            d3d12_defaultWhiteVBOView.BufferLocation = ID3D12Resource_GetGPUVirtualAddress(d3d12_defaultWhiteVBO);
            d3d12_defaultWhiteVBOView.SizeInBytes = whiteSize;
            d3d12_defaultWhiteVBOView.StrideInBytes = 4;  // R8G8B8A8_UNORM
        }
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

    // Default vertex color: opaque white (matches OpenGL backend)
    RLGL.State.colorr = 255;
    RLGL.State.colorg = 255;
    RLGL.State.colorb = 255;
    RLGL.State.colora = 255;

    RLGL.State.framebufferWidth = width;
    RLGL.State.framebufferHeight = height;

    // Execute initial commands
    rlFlushCommandList();

    TRACELOG(RL_LOG_INFO, "RLGL: Direct3D 12 backend initialized successfully");
}

void rlglClose(void)
{
    // Ensure command list is closed and all commands are submitted before cleanup
    if (RLGL.commandList) {
        rlFlushCommandList();  // Close the command list and submit all pending commands
    }
    
    // Wait for GPU to complete ALL in-flight operations multiple times to be safe
    for (int i = 0; i < 3; i++) {
        rlWaitForGpu();
    }

    // Drain all deferred PSO releases (GPU is idle after rlWaitForGpu)
    rlFlushDeferredPsoReleases();

    // Release all pending upload buffers across all frame slots
    for (int i = 0; i < RL_D3D12_NUM_FRAMES; i++) rlReleasePendingUploads(&RLGL.frames[i]);

    // Release cached PSOs and clear shader references
    rlClearPsoCache();

    rlUnloadRenderBatch(RLGL.defaultBatch);
    RLGL.currentBatch = NULL;
    rlUnloadShaderDefault();

    if (RLGL.State.defaultTextureId > 0) rlUnloadTexture(RLGL.State.defaultTextureId);

    // Release default white VBO
    if (d3d12_defaultWhiteVBO) { ID3D12Resource_Release(d3d12_defaultWhiteVBO); d3d12_defaultWhiteVBO = NULL; }

    // Release tracked resources
    for (unsigned int i = 0; i < RLGL.textureCount; i++) {
        if (RLGL.textures[i].resource) ID3D12Resource_Release(RLGL.textures[i].resource);
    }
    for (unsigned int i = 0; i < RLGL.shaderCount; i++) {
        if (RLGL.shaders[i].pso) ID3D12PipelineState_Release(RLGL.shaders[i].pso);
        if (RLGL.shaders[i].psoLine) ID3D12PipelineState_Release(RLGL.shaders[i].psoLine);
        if (RLGL.shaders[i].vsBlob) ID3D10Blob_Release(RLGL.shaders[i].vsBlob);
        if (RLGL.shaders[i].psBlob) ID3D10Blob_Release(RLGL.shaders[i].psBlob);
        // Release per-shader cbuffer resources
        for (int j = 0; j < RLGL.shaders[i].cbufferCount; j++) {
            if (RLGL.shaders[i].cbuffers[j].uploadBuffer) ID3D12Resource_Release(RLGL.shaders[i].cbuffers[j].uploadBuffer);
            if (RLGL.shaders[i].cbuffers[j].cpuData) RL_FREE(RLGL.shaders[i].cbuffers[j].cpuData);
        }
    }
    for (unsigned int i = 0; i < RLGL.bufferCount; i++) {
        if (RLGL.buffers[i].resource) ID3D12Resource_Release(RLGL.buffers[i].resource);
    }

    // Release swap chain resources
    if (RLGL.depthStencilBuffer) { ID3D12Resource_Release(RLGL.depthStencilBuffer); RLGL.depthStencilBuffer = NULL; }
    for (int i = 0; i < RL_D3D12_NUM_FRAMES; i++) {
        if (RLGL.renderTargets[i]) { ID3D12Resource_Release(RLGL.renderTargets[i]); RLGL.renderTargets[i] = NULL; }
    }
    if (RLGL.swapChain) { IDXGISwapChain3_Release(RLGL.swapChain); RLGL.swapChain = NULL; }

    // Release per-frame resources
    for (int i = 0; i < RL_D3D12_NUM_FRAMES; i++) {
        rlD3D12FrameResources *frame = &RLGL.frames[i];
        if (frame->commandAllocator) { ID3D12CommandAllocator_Release(frame->commandAllocator); frame->commandAllocator = NULL; }
        if (frame->cbMatrixUpload) { ID3D12Resource_Release(frame->cbMatrixUpload); frame->cbMatrixUpload = NULL; }
        if (frame->cbColorUpload) { ID3D12Resource_Release(frame->cbColorUpload); frame->cbColorUpload = NULL; }
        frame->cbMatrixMapped = NULL;
        frame->cbColorMapped = NULL;
        for (int j = 0; j < 4; j++) {
            if (frame->batchVertexBuffers[j]) { ID3D12Resource_Release(frame->batchVertexBuffers[j]); frame->batchVertexBuffers[j] = NULL; }
            frame->batchVertexMapped[j] = NULL;
        }
        if (frame->batchIndexBuffer) { ID3D12Resource_Release(frame->batchIndexBuffer); frame->batchIndexBuffer = NULL; }
        frame->batchIndexMapped = NULL;
        if (frame->cbRingBuffer) { ID3D12Resource_Release(frame->cbRingBuffer); frame->cbRingBuffer = NULL; }
        frame->cbRingMapped = NULL;
        frame->cbRingOffset = 0;
    }

    if (RLGL.commandList) { ID3D12GraphicsCommandList_Release(RLGL.commandList); RLGL.commandList = NULL; }
    if (RLGL.rootSignature) { ID3D12RootSignature_Release(RLGL.rootSignature); RLGL.rootSignature = NULL; }
    if (RLGL.fence) { ID3D12Fence_Release(RLGL.fence); RLGL.fence = NULL; }
    if (RLGL.fenceEvent) { CloseHandle(RLGL.fenceEvent); RLGL.fenceEvent = NULL; }
    if (RLGL.srvHeap) { ID3D12DescriptorHeap_Release(RLGL.srvHeap); RLGL.srvHeap = NULL; }
    if (RLGL.rtvHeap) { ID3D12DescriptorHeap_Release(RLGL.rtvHeap); RLGL.rtvHeap = NULL; }
    if (RLGL.dsvHeap) { ID3D12DescriptorHeap_Release(RLGL.dsvHeap); RLGL.dsvHeap = NULL; }
    if (RLGL.commandQueue) { ID3D12CommandQueue_Release(RLGL.commandQueue); RLGL.commandQueue = NULL; }
    if (RLGL.device) { ID3D12Device_Release(RLGL.device); RLGL.device = NULL; }
    if (RLGL.factory) { IDXGIFactory4_Release(RLGL.factory); RLGL.factory = NULL; }

    TRACELOG(RL_LOG_INFO, "RLGL: Direct3D 12 backend closed successfully");
}

void rlLoadExtensions(void *loader)
{
    (void)loader;
    RLGL.ExtSupported.computeShader = true;
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
//================================================================================
// Render batch management
//================================================================================
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
    if (!batch || !RLGL.commandList || !RLGL.device) return;

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

    // ---- Sub-allocation: each rlDrawRenderBatch call within a frame writes
    // to a distinct region of the upload buffers so that the GPU (which only
    // reads at ExecuteCommandLists time) sees the correct data for each draw.
    UINT vbBase = frame->batchVBOffset;            // vertex offset in VBs
    UINT cbSlot = frame->batchCBSlot;              // 256-aligned CB slot
    UINT cbByteOffset = cbSlot * 256;              // byte offset into CB buffer

    // Safety: if we've exhausted sub-allocation slots, skip drawing
    if (cbSlot >= RL_D3D12_MAX_BATCH_DRAWS_PER_FRAME ||
        (vbBase + (UINT)RLGL.State.vertexCounter) > RLGL.batchBufferSize) {
        TRACELOG(RL_LOG_WARNING, "D3D12: Batch sub-allocation overflow, skipping batch draw");
        goto batch_reset;
    }

    // Upload vertex data at sub-allocated VB offset
    UINT vertexSizes[4] = { 3 * sizeof(float), 2 * sizeof(float), 3 * sizeof(float), 4 * sizeof(unsigned char) };
    if (frame->batchVertexMapped[0])
        memcpy((char *)frame->batchVertexMapped[0] + (size_t)vbBase * vertexSizes[0],
            batch->vertexBuffer[batch->currentBuffer].vertices, RLGL.State.vertexCounter * vertexSizes[0]);
    if (frame->batchVertexMapped[1])
        memcpy((char *)frame->batchVertexMapped[1] + (size_t)vbBase * vertexSizes[1],
            batch->vertexBuffer[batch->currentBuffer].texcoords, RLGL.State.vertexCounter * vertexSizes[1]);
    if (frame->batchVertexMapped[2])
        memcpy((char *)frame->batchVertexMapped[2] + (size_t)vbBase * vertexSizes[2],
            batch->vertexBuffer[batch->currentBuffer].normals, RLGL.State.vertexCounter * vertexSizes[2]);
    if (frame->batchVertexMapped[3])
        memcpy((char *)frame->batchVertexMapped[3] + (size_t)vbBase * vertexSizes[3],
            batch->vertexBuffer[batch->currentBuffer].colors, RLGL.State.vertexCounter * vertexSizes[3]);

    // Bind render target (rlClearScreenBuffers already transitioned the swap chain RT to
    // RENDER_TARGET state; for FBOs we bind their color attachment here)
    bool hasRenderTarget = false;
    if (RLGL.activeFramebuffer == 0 && RLGL.swapChain) {
        UINT backIdx = d3d12_cachedBackBufferIndex;  // Use cached index (updated in rlClearScreenBuffers)
        if (RLGL.renderTargets[backIdx]) {
            // RT is already in RENDER_TARGET state (transitioned by rlClearScreenBuffers);
            // just re-bind in case OMSetRenderTargets wasn't called this batch.
            ID3D12GraphicsCommandList_OMSetRenderTargets(RLGL.commandList, 1,
                &RLGL.renderTargetHandles[backIdx], FALSE,
                RLGL.depthStencilBuffer ? &RLGL.dsvBackbuffer : NULL);
            hasRenderTarget = true;
        }
    } else if (RLGL.activeFramebuffer > 0 && RLGL.activeFramebuffer <= RLGL.framebufferCount) {
        rlD3D12Framebuffer *fb = &RLGL.framebuffers[RLGL.activeFramebuffer - 1];
        D3D12_CPU_DESCRIPTOR_HANDLE rtvHandle = { 0 };
        D3D12_CPU_DESCRIPTOR_HANDLE *dsvHandlePtr = NULL;
        D3D12_CPU_DESCRIPTOR_HANDLE dsvHandle = { 0 };
        if (fb->depthTexId > 0 && fb->depthTexId <= RLGL.textureCount) {
            dsvHandle = RLGL.textures[fb->depthTexId - 1].dsvHandle;
            dsvHandlePtr = &dsvHandle;
        }
        if (fb->colorCount > 0 && fb->colorTexIds[0] > 0 && fb->colorTexIds[0] <= RLGL.textureCount) {
            rtvHandle = RLGL.textures[fb->colorTexIds[0] - 1].rtvHandle;
            ID3D12GraphicsCommandList_OMSetRenderTargets(RLGL.commandList, 1, &rtvHandle, FALSE, dsvHandlePtr);
            hasRenderTarget = true;
        }
    }

    // Without a valid render target bound the NVIDIA driver will crash
    // inside SetPipelineState or draw calls (null internal RT descriptor).
    if (!hasRenderTarget) goto batch_reset;

    // Validate critical D3D12 state before recording any commands
    if (!RLGL.rootSignature || !RLGL.srvHeap) return;

    // Set root signature and descriptor heaps
    ID3D12GraphicsCommandList_SetGraphicsRootSignature(RLGL.commandList, RLGL.rootSignature);
    ID3D12DescriptorHeap *heaps[] = { RLGL.srvHeap };
    ID3D12GraphicsCommandList_SetDescriptorHeaps(RLGL.commandList, 1, heaps);

    // Set viewport and scissor — BOTH are mandatory D3D12 state.
    // After a command list Reset all previous state is cleared, so we must
    // always (re-)apply them before any draw call.  The NVIDIA driver crashes
    // at offset 0x458 inside SetPipelineState/Draw when the scissor rect is
    // missing from the command list.
    D3D12_VIEWPORT viewport = { 0, 0, (float)RLGL.State.framebufferWidth, (float)RLGL.State.framebufferHeight, 0.0f, 1.0f };
    ID3D12GraphicsCommandList_RSSetViewports(RLGL.commandList, 1, &viewport);
    if (RLGL.State.scissorTestEnabled) {
        D3D12_RECT scissorRect = {
            RLGL.State.scissorX, RLGL.State.scissorY,
            RLGL.State.scissorX + RLGL.State.scissorWidth,
            RLGL.State.scissorY + RLGL.State.scissorHeight
        };
        ID3D12GraphicsCommandList_RSSetScissorRects(RLGL.commandList, 1, &scissorRect);
    } else {
        D3D12_RECT scissorRect = { 0, 0, RLGL.State.framebufferWidth, RLGL.State.framebufferHeight };
        ID3D12GraphicsCommandList_RSSetScissorRects(RLGL.commandList, 1, &scissorRect);
    }

    // Set pipeline state – ensure a valid PSO is always bound before any draw call.
    // After a command list Reset all state is cleared, so drawing without a PSO
    // causes a NULL-dereference crash inside the GPU driver (nvwgf2umx.dll).
    ID3D12PipelineState *activePSO = NULL;
    ID3D12PipelineState *activePSOLine = NULL;
    if (RLGL.State.currentShaderId > 0 && RLGL.State.currentShaderId <= RLGL.shaderCount) {
        unsigned int shaderIdx = RLGL.State.currentShaderId - 1;
        rlUpdateD3D12PipelineState();
        activePSO = RLGL.shaders[shaderIdx].pso;
        activePSOLine = RLGL.shaders[shaderIdx].psoLine;
    }

    // Fallback: if the current shader has no PSO, try the default shader's PSO
    if (!activePSO && RLGL.State.defaultShaderId > 0 && RLGL.State.defaultShaderId <= RLGL.shaderCount) {
        activePSO = RLGL.shaders[RLGL.State.defaultShaderId - 1].pso;
        activePSOLine = RLGL.shaders[RLGL.State.defaultShaderId - 1].psoLine;
    }

    // If we still have no PSO, we cannot issue any draw calls – bail out to avoid a driver crash
    if (!activePSO) {
        TRACELOG(RL_LOG_WARNING, "D3D12: No valid PSO available, skipping batch draw");
        goto batch_reset;
    }
    ID3D12GraphicsCommandList_SetPipelineState(RLGL.commandList, activePSO);
    d3d12_lastBoundPSO = activePSO;
    d3d12_baseStateSet = true;  // Batch path has set root sig, heaps, viewport, scissor

    // Update and bind batch MVP constant buffer (sub-allocated)
    Matrix matMVP = rlMatrixMultiply(RLGL.State.modelview, RLGL.State.projection);
    rl_float16 matData = rlMatrixToFloatV(matMVP);
    if (frame->cbMatrixMapped) memcpy((char *)frame->cbMatrixMapped + cbByteOffset, matData.v, sizeof(float) * 16);

    float diffuse[4] = { 1.0f, 1.0f, 1.0f, 1.0f };
    if (frame->cbColorMapped) memcpy((char *)frame->cbColorMapped + cbByteOffset, diffuse, sizeof(float) * 4);

    // Bind batch MVP matrix to VS b0 (root param 0)
    if (frame->cbMatrixUpload)
        ID3D12GraphicsCommandList_SetGraphicsRootConstantBufferView(RLGL.commandList,
            rlGetRootParamForCBuffer(0, 0),
            ID3D12Resource_GetGPUVirtualAddress(frame->cbMatrixUpload) + cbByteOffset);
    // Bind batch color to PS b1 (default shader PS colDiffuse register)
    if (frame->cbColorUpload)
        ID3D12GraphicsCommandList_SetGraphicsRootConstantBufferView(RLGL.commandList,
            rlGetRootParamForCBuffer(1, 1),
            ID3D12Resource_GetGPUVirtualAddress(frame->cbColorUpload) + cbByteOffset);

    // Set batch vertex buffers (full buffer; vertex offsets applied via draw params)
    D3D12_VERTEX_BUFFER_VIEW vbViews[4] = { 0 };
    for (int i = 0; i < 4; i++) {
        if (frame->batchVertexBuffers[i]) {
            vbViews[i].BufferLocation = ID3D12Resource_GetGPUVirtualAddress(frame->batchVertexBuffers[i]);
            vbViews[i].SizeInBytes = RLGL.batchBufferSize * vertexSizes[i];
            vbViews[i].StrideInBytes = vertexSizes[i];
        }
    }
    ID3D12GraphicsCommandList_IASetVertexBuffers(RLGL.commandList, 0, 4, vbViews);

    // Set batch index buffer
    if (frame->batchIndexBuffer) {
        D3D12_INDEX_BUFFER_VIEW ibView = { 0 };
        ibView.BufferLocation = ID3D12Resource_GetGPUVirtualAddress(frame->batchIndexBuffer);
        ibView.SizeInBytes = RL_DEFAULT_BATCH_BUFFER_ELEMENTS * 6 * sizeof(unsigned int);
        ibView.Format = DXGI_FORMAT_R32_UINT;
        ID3D12GraphicsCommandList_IASetIndexBuffer(RLGL.commandList, &ibView);
    }

    // Draw each batch draw call (vertex offsets shifted by vbBase for sub-allocation)
    for (int i = 0, vertexOffset = 0; i < batch->drawCounter; i++) {
        // Skip empty draw calls
        if (batch->draws[i].vertexCount <= 0) {
            vertexOffset += batch->draws[i].vertexAlignment;
            continue;
        }

        unsigned int texId = batch->draws[i].textureId;
        if (texId > 0 && texId <= RLGL.textureCount) {
            ID3D12GraphicsCommandList_SetGraphicsRootDescriptorTable(RLGL.commandList, RL_D3D12_ROOT_SRV_TABLE_IDX,
                RLGL.textures[texId - 1].srvGpuHandle);
        }

        if ((batch->draws[i].mode == RL_LINES) || (batch->draws[i].mode == RL_TRIANGLES)) {
            bool isLines = (batch->draws[i].mode == RL_LINES);
            // Switch PSO to match the topology group (line vs triangle)
            ID3D12PipelineState *drawPSO = isLines ? activePSOLine : activePSO;
            if (drawPSO) {
                ID3D12GraphicsCommandList_SetPipelineState(RLGL.commandList, drawPSO);
                d3d12_lastBoundPSO = drawPSO;
            }
            D3D_PRIMITIVE_TOPOLOGY topo = isLines ?
                D3D_PRIMITIVE_TOPOLOGY_LINELIST : D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
            ID3D12GraphicsCommandList_IASetPrimitiveTopology(RLGL.commandList, topo);
            ID3D12GraphicsCommandList_DrawInstanced(RLGL.commandList,
                batch->draws[i].vertexCount, 1, vertexOffset + (int)vbBase, 0);
            // Restore triangle PSO for subsequent draws
            if (isLines && activePSO) {
                ID3D12GraphicsCommandList_SetPipelineState(RLGL.commandList, activePSO);
                d3d12_lastBoundPSO = activePSO;
            }
        } else {
            ID3D12GraphicsCommandList_IASetPrimitiveTopology(RLGL.commandList, D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            ID3D12GraphicsCommandList_DrawIndexedInstanced(RLGL.commandList,
                batch->draws[i].vertexCount / 4 * 6, 1,
                vertexOffset / 4 * 6, (INT)vbBase, 0);
        }
        vertexOffset += (batch->draws[i].vertexCount + batch->draws[i].vertexAlignment);
    }

    // Advance sub-allocation offsets for the next batch draw within this frame
    frame->batchCBSlot = cbSlot + 1;
    frame->batchVBOffset = vbBase + (UINT)RLGL.State.vertexCounter;

    // Reset batch  (RT→PRESENT transition happens once per frame in rlSwapScreenBuffer)
batch_reset:
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
    if (RLGL.currentBatch) rlDrawRenderBatch(RLGL.currentBatch);
    if (batch != NULL) RLGL.currentBatch = batch;
    else RLGL.currentBatch = &RLGL.defaultBatch;
}

void rlDrawRenderBatchActive(void) { rlDrawRenderBatch(RLGL.currentBatch); }

bool rlCheckRenderBatchLimit(int vCount)
{
    bool overflow = false;
    if (!RLGL.currentBatch) return false;
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
//================================================================================
// Texture management
//================================================================================
unsigned int rlLoadTexture(const void *data, int width, int height, int format, int mipmapCount)
{
    if (!RLGL.device || RLGL.textureCount >= RL_D3D12_MAX_TEXTURES) return 0;

    DXGI_FORMAT dxgiFormat = rlGetDXGIFormat(format);
    unsigned int idx = RLGL.textureCount;

    int bpp = 4;
    switch (format) {
        case 1: bpp = 1; break; case 2: bpp = 2; break; case 3: bpp = 2; break;
        case 4: bpp = 4; break; case 5: bpp = 2; break; case 6: bpp = 2; break;
        case 7: bpp = 4; break; case 8: bpp = 4; break; case 9: bpp = 12; break;
        case 10: bpp = 16; break; case 11: bpp = 2; break; case 12: bpp = 8; break;
        case 13: bpp = 8; break; default: bpp = 4; break;
    }

    // Expand 3-channel data
    void *expandedData = NULL;
    const void *uploadData = data;
    if (data != NULL && format == 4) {
        int pixelCount = width * height;
        expandedData = RL_MALLOC(pixelCount * 4);
        const unsigned char *src = (const unsigned char *)data;
        unsigned char *dst = (unsigned char *)expandedData;
        for (int i = 0; i < pixelCount; i++) {
            dst[i*4+0] = src[i*3+0]; dst[i*4+1] = src[i*3+1];
            dst[i*4+2] = src[i*3+2]; dst[i*4+3] = 255;
        }
        uploadData = expandedData;
    } else if (data != NULL && format == 12) {
        int pixelCount = width * height;
        expandedData = RL_MALLOC(pixelCount * 8);
        const unsigned short *src = (const unsigned short *)data;
        unsigned short *dst = (unsigned short *)expandedData;
        for (int i = 0; i < pixelCount; i++) {
            dst[i*4+0] = src[i*3+0]; dst[i*4+1] = src[i*3+1];
            dst[i*4+2] = src[i*3+2]; dst[i*4+3] = 0x3C00;
        }
        uploadData = expandedData;
    }

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
        D3D12_HEAP_FLAG_NONE, &texDesc, D3D12_RESOURCE_STATE_COPY_DEST,
        NULL, &IID_ID3D12Resource, (void **)&RLGL.textures[idx].resource);
    if (FAILED(hr)) { RL_FREE(expandedData); return 0; }

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
                const unsigned char *srcData = (const unsigned char *)uploadData;
                unsigned char *dstData = (unsigned char *)mapped;
                UINT srcPitch = width * bpp;
                for (UINT row = 0; row < numRows; row++) {
                    memcpy(dstData + footprint.Footprint.RowPitch * row, srcData + srcPitch * row, srcPitch);
                }
                ID3D12Resource_Unmap(uploadBuffer, 0, NULL);
            }

            D3D12_TEXTURE_COPY_LOCATION dstLoc = { 0 };
            dstLoc.pResource = RLGL.textures[idx].resource;
            dstLoc.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
            dstLoc.SubresourceIndex = 0;

            D3D12_TEXTURE_COPY_LOCATION srcLoc = { 0 };
            srcLoc.pResource = uploadBuffer;
            srcLoc.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
            srcLoc.PlacedFootprint = footprint;

            ID3D12GraphicsCommandList_CopyTextureRegion(RLGL.commandList, &dstLoc, 0, 0, 0, &srcLoc, NULL);
            rlTransitionResource(RLGL.textures[idx].resource, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
            rlDeferUploadRelease(uploadBuffer);
        }
    } else {
        rlTransitionResource(RLGL.textures[idx].resource, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
    }

    RL_FREE(expandedData);

    // Create SRV
    D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = { 0 };
    srvDesc.Format = dxgiFormat;
    srvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;

    // Component mapping: match OpenGL swizzle masks for < 4-channel formats.
    //   Format 1 (GRAYSCALE, R8):   shader sees (R, R, R, 1)
    //   Format 2 (GRAY_ALPHA, R8G8): shader sees (R, R, R, G)
    // Without this the font texture (GRAY_ALPHA) has alpha stuck at 1.0
    // and draws solid opaque rectangles instead of glyph shapes.
    if (format == 1)
        srvDesc.Shader4ComponentMapping = D3D12_ENCODE_SHADER_4_COMPONENT_MAPPING(0, 0, 0, 5);
    else if (format == 2)
        srvDesc.Shader4ComponentMapping = D3D12_ENCODE_SHADER_4_COMPONENT_MAPPING(0, 0, 0, 1);
    else
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
        D3D12_HEAP_FLAG_NONE, &texDesc, D3D12_RESOURCE_STATE_DEPTH_WRITE,
        &clearValue, &IID_ID3D12Resource, (void **)&RLGL.textures[idx].resource);
    if (FAILED(hr)) return 0;

    D3D12_DEPTH_STENCIL_VIEW_DESC dsvDesc = { 0 };
    dsvDesc.Format = DXGI_FORMAT_D32_FLOAT;
    dsvDesc.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2D;

    D3D12_CPU_DESCRIPTOR_HANDLE dsvStart;
    ID3D12DescriptorHeap_GetCPUDescriptorHandleForHeapStart(RLGL.dsvHeap, &dsvStart);
    RLGL.textures[idx].dsvIndex = RLGL.dsvHeapUsed++;
    RLGL.textures[idx].dsvHandle.ptr = dsvStart.ptr + (SIZE_T)(RLGL.textures[idx].dsvIndex * RLGL.dsvDescriptorSize);
    ID3D12Device_CreateDepthStencilView(RLGL.device, RLGL.textures[idx].resource, &dsvDesc, RLGL.textures[idx].dsvHandle);

    D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = { 0 };
    srvDesc.Format = DXGI_FORMAT_R32_FLOAT;
    srvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srvDesc.Texture2D.MipLevels = 1;

    RLGL.textures[idx].srvIndex = rlAllocSRVDescriptor(&RLGL.textures[idx].srvCpuHandle, &RLGL.textures[idx].srvGpuHandle);
    ID3D12Device_CreateShaderResourceView(RLGL.device, RLGL.textures[idx].resource, &srvDesc, RLGL.textures[idx].srvCpuHandle);

    RLGL.textures[idx].width = width;
    RLGL.textures[idx].height = height;
    RLGL.textures[idx].format = 8;
    RLGL.textures[idx].isDepth = true;
    RLGL.textureCount++;

    unsigned int id = idx + 1;
    TRACELOG(RL_LOG_INFO, "TEXTURE: [ID %i] Depth texture created (%ix%i)", id, width, height);
    return id;
}

unsigned int rlLoadTextureCubemap(const void *data, int size, int format, int mipmapCount)
{
    if (!RLGL.device || !RLGL.commandList || RLGL.textureCount >= RL_D3D12_MAX_TEXTURES) return 0;
    if (mipmapCount < 1) mipmapCount = 1;
    if (mipmapCount > 16) mipmapCount = 16;

    unsigned int idx = RLGL.textureCount;
    DXGI_FORMAT dxgiFormat = rlGetDXGIFormat(format);

    // Source bytes per pixel
    int srcBpp = 4;
    switch (format) {
        case 1: srcBpp = 1; break; case 2: srcBpp = 2; break; case 3: srcBpp = 2; break;
        case 4: srcBpp = 3; break; case 5: srcBpp = 2; break; case 6: srcBpp = 2; break;
        case 7: srcBpp = 4; break; case 8: srcBpp = 4; break; case 9: srcBpp = 12; break;
        case 10: srcBpp = 16; break; case 11: srcBpp = 2; break; case 12: srcBpp = 6; break;
        case 13: srcBpp = 8; break; default: srcBpp = 4; break;
    }
    int dstBpp = srcBpp;
    if (format == 4) dstBpp = 4;
    if (format == 12) dstBpp = 8;

    // Create texture2D array with 6 faces (cubemap)
    D3D12_RESOURCE_DESC texDesc = { 0 };
    texDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    texDesc.Width = size;
    texDesc.Height = size;
    texDesc.DepthOrArraySize = 6;
    texDesc.MipLevels = mipmapCount;
    texDesc.Format = dxgiFormat;
    texDesc.SampleDesc.Count = 1;
    texDesc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    texDesc.Flags = D3D12_RESOURCE_FLAG_NONE;

    D3D12_HEAP_PROPERTIES heapProps = { 0 };
    heapProps.Type = D3D12_HEAP_TYPE_DEFAULT;

    HRESULT hr = ID3D12Device_CreateCommittedResource(RLGL.device, &heapProps,
        D3D12_HEAP_FLAG_NONE, &texDesc, D3D12_RESOURCE_STATE_COPY_DEST,
        NULL, &IID_ID3D12Resource, (void **)&RLGL.textures[idx].resource);
    if (FAILED(hr)) {
        TRACELOG(RL_LOG_WARNING, "D3D12: Failed to create cubemap texture (hr=0x%08X)", hr);
        return 0;
    }

    // Upload per-face data
    if (data) {
        const unsigned char *srcPtr = (const unsigned char *)data;
        int mipSize = size;
        for (int mip = 0; mip < mipmapCount; mip++) {
            int srcFaceBytes = mipSize * mipSize * srcBpp;

            for (int face = 0; face < 6; face++) {
                UINT subresource = face * mipmapCount + mip;

                // Get footprint for this subresource
                D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint = { 0 };
                UINT numRows = 0;
                UINT64 rowSize = 0, totalSize = 0;
                ID3D12Device_GetCopyableFootprints(RLGL.device, &texDesc, subresource, 1, 0, &footprint, &numRows, &rowSize, &totalSize);

                ID3D12Resource *uploadBuf = rlCreateUploadBuffer((UINT)totalSize);
                if (!uploadBuf) { srcPtr += srcFaceBytes; continue; }

                void *mapped = NULL;
                D3D12_RANGE readRange = { 0, 0 };
                ID3D12Resource_Map(uploadBuf, 0, &readRange, &mapped);
                if (mapped) {
                    // Copy with potential format expansion
                    int pixelCount = mipSize * mipSize;
                    if (format == 4) {
                        // R8G8B8 -> R8G8B8A8
                        unsigned char *dst = (unsigned char *)mapped;
                        for (int row = 0; row < mipSize; row++) {
                            for (int col = 0; col < mipSize; col++) {
                                int sp = (row * mipSize + col) * 3;
                                int dp = row * footprint.Footprint.RowPitch + col * 4;
                                dst[dp+0] = srcPtr[sp+0]; dst[dp+1] = srcPtr[sp+1];
                                dst[dp+2] = srcPtr[sp+2]; dst[dp+3] = 255;
                            }
                        }
                    } else if (format == 12) {
                        // R16G16B16 -> R16G16B16A16
                        unsigned char *dst = (unsigned char *)mapped;
                        const unsigned short *src16 = (const unsigned short *)srcPtr;
                        for (int row = 0; row < mipSize; row++) {
                            unsigned short *dstRow = (unsigned short *)(dst + row * footprint.Footprint.RowPitch);
                            for (int col = 0; col < mipSize; col++) {
                                int sp = (row * mipSize + col) * 3;
                                dstRow[col*4+0] = src16[sp+0]; dstRow[col*4+1] = src16[sp+1];
                                dstRow[col*4+2] = src16[sp+2]; dstRow[col*4+3] = 0x3C00;
                            }
                        }
                    } else {
                        // Direct copy row by row
                        UINT srcRowPitch = mipSize * dstBpp;
                        unsigned char *dst = (unsigned char *)mapped;
                        for (UINT row = 0; row < numRows; row++) {
                            memcpy(dst + row * footprint.Footprint.RowPitch, srcPtr + row * srcRowPitch, srcRowPitch);
                        }
                    }
                    ID3D12Resource_Unmap(uploadBuf, 0, NULL);
                }

                D3D12_TEXTURE_COPY_LOCATION dstLoc = { 0 };
                dstLoc.pResource = RLGL.textures[idx].resource;
                dstLoc.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
                dstLoc.SubresourceIndex = subresource;

                D3D12_TEXTURE_COPY_LOCATION srcLoc = { 0 };
                srcLoc.pResource = uploadBuf;
                srcLoc.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
                srcLoc.PlacedFootprint = footprint;

                ID3D12GraphicsCommandList_CopyTextureRegion(RLGL.commandList, &dstLoc, 0, 0, 0, &srcLoc, NULL);

                rlDeferUploadRelease(uploadBuf);

                srcPtr += srcFaceBytes;
            }
            mipSize /= 2;
            if (mipSize < 1) mipSize = 1;
        }
    }

    // Transition to shader resource (no flush - commands execute at frame submit time)
    rlTransitionResource(RLGL.textures[idx].resource, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);

    // Create SRV as cubemap
    D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = { 0 };
    srvDesc.Format = dxgiFormat;
    srvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURECUBE;
    srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srvDesc.TextureCube.MipLevels = mipmapCount;

    RLGL.textures[idx].srvIndex = rlAllocSRVDescriptor(&RLGL.textures[idx].srvCpuHandle, &RLGL.textures[idx].srvGpuHandle);
    if (RLGL.textures[idx].srvIndex < 0) {
        ID3D12Resource_Release(RLGL.textures[idx].resource);
        RLGL.textures[idx].resource = NULL;
        return 0;
    }
    ID3D12Device_CreateShaderResourceView(RLGL.device, RLGL.textures[idx].resource, &srvDesc, RLGL.textures[idx].srvCpuHandle);

    RLGL.textures[idx].width = size;
    RLGL.textures[idx].height = size;
    RLGL.textures[idx].isCubemap = true;
    RLGL.textureCount++;

    TRACELOG(RL_LOG_INFO, "TEXTURE: [ID %i] Cubemap texture loaded successfully (%ix%i, 6 faces)", idx + 1, size, size);
    return idx + 1;
}

void rlUpdateTexture(unsigned int id, int offsetX, int offsetY, int width, int height, int format, const void *data)
{
    if (id == 0 || id > RLGL.textureCount || !RLGL.commandList || !data) return;
    unsigned int idx = id - 1;

    int bpp = 4;
    switch (format) {
        case 1: bpp = 1; break; case 2: bpp = 2; break; case 3: bpp = 2; break;
        case 4: bpp = 4; break; case 5: bpp = 2; break; case 6: bpp = 2; break;
        case 7: bpp = 4; break; case 8: bpp = 4; break; case 9: bpp = 12; break;
        case 10: bpp = 16; break; case 11: bpp = 2; break; case 12: bpp = 8; break;
        case 13: bpp = 8; break; default: bpp = 4; break;
    }

    void *expandedData = NULL;
    const void *uploadData = data;
    if (format == 4) {
        int pixelCount = width * height;
        expandedData = RL_MALLOC(pixelCount * 4);
        const unsigned char *src = (const unsigned char *)data;
        unsigned char *dst = (unsigned char *)expandedData;
        for (int i = 0; i < pixelCount; i++) {
            dst[i*4+0] = src[i*3+0]; dst[i*4+1] = src[i*3+1];
            dst[i*4+2] = src[i*3+2]; dst[i*4+3] = 255;
        }
        uploadData = expandedData;
    } else if (format == 12) {
        int pixelCount = width * height;
        expandedData = RL_MALLOC(pixelCount * 8);
        const unsigned short *src = (const unsigned short *)data;
        unsigned short *dst = (unsigned short *)expandedData;
        for (int i = 0; i < pixelCount; i++) {
            dst[i*4+0] = src[i*3+0]; dst[i*4+1] = src[i*3+1];
            dst[i*4+2] = src[i*3+2]; dst[i*4+3] = 0x3C00;
        }
        uploadData = expandedData;
    }

    UINT uploadSize = width * height * bpp;

    // Use GetCopyableFootprints for correct row pitch alignment (D3D12_TEXTURE_DATA_PITCH_ALIGNMENT)
    D3D12_RESOURCE_DESC updateTexDesc = { 0 };
    updateTexDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    updateTexDesc.Width = width;
    updateTexDesc.Height = height;
    updateTexDesc.DepthOrArraySize = 1;
    updateTexDesc.MipLevels = 1;
    updateTexDesc.Format = RLGL.textures[idx].dxgiFormat;
    updateTexDesc.SampleDesc.Count = 1;
    updateTexDesc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;

    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint = { 0 };
    UINT numRows = 0;
    UINT64 rowSize = 0, totalSize = 0;
    ID3D12Device_GetCopyableFootprints(RLGL.device, &updateTexDesc, 0, 1, 0, &footprint, &numRows, &rowSize, &totalSize);

    ID3D12Resource *uploadBuffer = rlCreateUploadBuffer((UINT)totalSize);
    if (!uploadBuffer) { RL_FREE(expandedData); return; }

    void *mapped = NULL;
    D3D12_RANGE readRange = { 0, 0 };
    ID3D12Resource_Map(uploadBuffer, 0, &readRange, &mapped);
    if (mapped) {
        const unsigned char *srcData = (const unsigned char *)uploadData;
        unsigned char *dstData = (unsigned char *)mapped;
        UINT srcPitch = width * bpp;
        for (UINT row = 0; row < numRows; row++) {
            memcpy(dstData + footprint.Footprint.RowPitch * row, srcData + srcPitch * row, srcPitch);
        }
        ID3D12Resource_Unmap(uploadBuffer, 0, NULL);
    }

    rlTransitionResource(RLGL.textures[idx].resource, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_DEST);

    D3D12_TEXTURE_COPY_LOCATION dstLoc = { 0 };
    dstLoc.pResource = RLGL.textures[idx].resource;
    dstLoc.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    dstLoc.SubresourceIndex = 0;

    D3D12_TEXTURE_COPY_LOCATION srcLoc = { 0 };
    srcLoc.pResource = uploadBuffer;
    srcLoc.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    srcLoc.PlacedFootprint = footprint;

    D3D12_BOX srcBox = { 0, 0, 0, (UINT)width, (UINT)height, 1 };
    ID3D12GraphicsCommandList_CopyTextureRegion(RLGL.commandList, &dstLoc, offsetX, offsetY, 0, &srcLoc, &srcBox);

    rlTransitionResource(RLGL.textures[idx].resource, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
    rlDeferUploadRelease(uploadBuffer);
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
        case 1: return "GRAYSCALE"; case 2: return "GRAY_ALPHA"; case 3: return "R5G6B5";
        case 4: return "R8G8B8"; case 5: return "R5G5B5A1"; case 6: return "R4G4B4A4";
        case 7: return "R8G8B8A8"; case 8: return "R32"; case 9: return "R32G32B32";
        case 10: return "R32G32B32A32"; default: return "UNKNOWN";
    }
}

void rlUnloadTexture(unsigned int id) {
    if (id == 0 || id > RLGL.textureCount) return;
    unsigned int idx = id - 1;
    if (RLGL.textures[idx].resource) { ID3D12Resource_Release(RLGL.textures[idx].resource); RLGL.textures[idx].resource = NULL; }
}

void rlGenTextureMipmaps(unsigned int id, int width, int height, int format, int *mipmaps) {
    (void)id; (void)format;
    if (mipmaps) {
        int max = (width > height) ? width : height;
        *mipmaps = 1 + (int)floor(log((double)max) / log(2.0));
    }
}

void *rlReadTexturePixels(unsigned int id, int width, int height, int format) {
    if (id == 0 || id > RLGL.textureCount || !RLGL.commandList || !RLGL.device) return NULL;
    unsigned int idx = id - 1;

    D3D12_RESOURCE_DESC texDesc = { 0 };
    ID3D12Resource_GetDesc(RLGL.textures[idx].resource, &texDesc);

    UINT64 totalSize = 0;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint = { 0 };
    UINT numRows = 0;
    UINT64 rowSize = 0;
    ID3D12Device_GetCopyableFootprints(RLGL.device, &texDesc, 0, 1, 0, &footprint, &numRows, &rowSize, &totalSize);

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

    rlTransitionResource(RLGL.textures[idx].resource, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_SOURCE);

    D3D12_TEXTURE_COPY_LOCATION dstLoc = { 0 };
    dstLoc.pResource = readback;
    dstLoc.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    dstLoc.PlacedFootprint = footprint;

    D3D12_TEXTURE_COPY_LOCATION srcLoc = { 0 };
    srcLoc.pResource = RLGL.textures[idx].resource;
    srcLoc.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    srcLoc.SubresourceIndex = 0;

    ID3D12GraphicsCommandList_CopyTextureRegion(RLGL.commandList, &dstLoc, 0, 0, 0, &srcLoc, NULL);
    rlTransitionResource(RLGL.textures[idx].resource, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
    rlFlushCommandList();

    int dataSize = rlGetPixelDataSize(width, height, format);
    void *pixels = RL_MALLOC(dataSize);

    void *mapped = NULL;
    D3D12_RANGE readRange = { 0, totalSize };
    if (SUCCEEDED(ID3D12Resource_Map(readback, 0, &readRange, &mapped))) {
        int bpp = dataSize / (width * height);
        for (int row = 0; row < height; row++) {
            memcpy((unsigned char *)pixels + row * width * bpp,
                   (unsigned char *)mapped + row * footprint.Footprint.RowPitch, width * bpp);
        }
        D3D12_RANGE writeRange = { 0, 0 };
        ID3D12Resource_Unmap(readback, 0, &writeRange);
    }

    ID3D12Resource_Release(readback);
    return pixels;
}

//================================================================================
// Framebuffer management
//================================================================================
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
//================================================================================
// Vertex buffer management
//================================================================================
unsigned int rlLoadVertexArray(void) { return 1; }
void rlUnloadVertexArray(unsigned int vaoId) { (void)vaoId; }
bool rlEnableVertexArray(unsigned int vaoId) { (void)vaoId; return false; }  // D3D12 has no VAOs; return false so DrawMesh falls through to manual VBO/IB binding
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
            ID3D12Resource *upload = rlCreateUploadBuffer(size);
            if (upload) {
                void *mapped = NULL;
                D3D12_RANGE readRange = { 0, 0 };
                ID3D12Resource_Map(upload, 0, &readRange, &mapped);
                if (mapped) { memcpy(mapped, buffer, size); ID3D12Resource_Unmap(upload, 0, NULL); }
                rlTransitionResource(RLGL.buffers[idx].resource, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
                ID3D12GraphicsCommandList_CopyBufferRegion(RLGL.commandList, RLGL.buffers[idx].resource, 0, upload, 0, size);
                rlTransitionResource(RLGL.buffers[idx].resource, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER);
                rlDeferUploadRelease(upload);
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
                rlDeferUploadRelease(upload);
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
    // Defer actual binding - rlSetVertexAttribute will bind with correct slot and stride
    d3d12_pendingVBO = id;
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
    d3d12_lastIBView = ibView;
    d3d12_hasIBBound = true;
}

void rlDisableVertexBufferElement(void) { }
void rlEnableVertexAttribute(unsigned int index) { (void)index; }
void rlDisableVertexAttribute(unsigned int index) { (void)index; }

void rlSetVertexAttribute(unsigned int index, int compSize, int type, bool normalized, int stride, int offset) {
    (void)normalized;
    // Deferred VBO binding: consume d3d12_pendingVBO and bind with correct input slot and stride
    if (d3d12_pendingVBO > 0 && d3d12_pendingVBO <= RLGL.bufferCount && RLGL.commandList) {
        unsigned int bufIdx = d3d12_pendingVBO - 1;
        int elemSize = (type == RL_UNSIGNED_BYTE) ? 1 : (int)sizeof(float);
        UINT actualStride = (stride > 0) ? (UINT)stride : (UINT)(compSize * elemSize);
        D3D12_VERTEX_BUFFER_VIEW vbView = { 0 };
        vbView.BufferLocation = ID3D12Resource_GetGPUVirtualAddress(RLGL.buffers[bufIdx].resource);
        vbView.SizeInBytes = RLGL.buffers[bufIdx].size;
        vbView.StrideInBytes = actualStride;
        ID3D12GraphicsCommandList_IASetVertexBuffers(RLGL.commandList, index, 1, &vbView);
        if (index < 4) d3d12_lastVBViews[index] = vbView;
        d3d12_hasVBBound = true;
        d3d12_pendingVBO = 0;
    }
}

void rlSetVertexAttributeDivisor(unsigned int index, int divisor) { (void)index; (void)divisor; }

void rlSetVertexAttributeDefault(int locIndex, const void *value, int attribType, int count) {
    // Bind the default white vertex buffer to provide (1,1,1,1) data
    // Primarily used for the COLOR attribute (locIndex=3) when meshes don't have vertex colors
    (void)value; (void)attribType; (void)count;
    if (locIndex < 0 || !RLGL.commandList || !d3d12_defaultWhiteVBO) return;
    ID3D12GraphicsCommandList_IASetVertexBuffers(RLGL.commandList, locIndex, 1, &d3d12_defaultWhiteVBOView);
    if (locIndex < 4) d3d12_lastVBViews[locIndex] = d3d12_defaultWhiteVBOView;
    d3d12_hasVBBound = true;
}
//================================================================================
// Shader management (with reflection for mesh rendering)
//================================================================================
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
        if (errors) { TRACELOG(RL_LOG_WARNING, "SHADER: Compile error: %s", (char *)ID3D10Blob_GetBufferPointer(errors)); ID3D10Blob_Release(errors); }
        return 0;
    }
    if (errors) ID3D10Blob_Release(errors);

    unsigned int id = RLGL.shaderCount + 1;
    if (type == RL_VERTEX_SHADER) RLGL.shaders[RLGL.shaderCount].vsBlob = blob;
    else RLGL.shaders[RLGL.shaderCount].psBlob = blob;

    return id;
}

//--------------------------------------------------------------------------------
// Shader reflection: discover constant buffers, uniform variables, texture bindings
// Uses ID3D11ShaderReflection via D3DReflect (compatible with SM 5.0 bytecode)
//--------------------------------------------------------------------------------
static void rlReflectShaderStage(unsigned int shaderIdx, ID3DBlob *blob, int stage)
{
    if (!blob) return;
    rlD3D12Shader *s = &RLGL.shaders[shaderIdx];

    ID3D11ShaderReflection *reflect = NULL;
    HRESULT hr = D3DReflect(ID3D10Blob_GetBufferPointer(blob), ID3D10Blob_GetBufferSize(blob),
                            &IID_ID3D11ShaderReflection, (void **)&reflect);
    if (FAILED(hr) || !reflect) return;

    D3D11_SHADER_DESC shaderDesc;
    reflect->lpVtbl->GetDesc(reflect, &shaderDesc);

    // Enumerate constant buffers
    for (UINT i = 0; i < shaderDesc.ConstantBuffers && s->cbufferCount < RL_D3D12_MAX_CBUFFERS; i++) {
        ID3D11ShaderReflectionConstantBuffer *cbReflect =
            reflect->lpVtbl->GetConstantBufferByIndex(reflect, i);
        D3D11_SHADER_BUFFER_DESC cbDesc;
        hr = cbReflect->lpVtbl->GetDesc(cbReflect, &cbDesc);
        if (FAILED(hr)) continue;
        if (cbDesc.Type != D3D_CT_CBUFFER) continue;

        D3D11_SHADER_INPUT_BIND_DESC bindDesc;
        hr = reflect->lpVtbl->GetResourceBindingDescByName(reflect, cbDesc.Name, &bindDesc);
        int regSlot = SUCCEEDED(hr) ? (int)bindDesc.BindPoint : (int)i;

        int cbIdx = s->cbufferCount;
        s->cbuffers[cbIdx].stage = stage;
        s->cbuffers[cbIdx].registerSlot = regSlot;
        s->cbuffers[cbIdx].byteSize = (int)cbDesc.Size;

        // Create upload heap buffer (persistently mapped) for this cbuffer
        UINT alignedSize = (cbDesc.Size + 255) & ~255;  // 256-byte aligned for D3D12
        s->cbuffers[cbIdx].uploadBuffer = rlCreateUploadBuffer(alignedSize);
        if (s->cbuffers[cbIdx].uploadBuffer) {
            D3D12_RANGE readRange = { 0, 0 };
            ID3D12Resource_Map(s->cbuffers[cbIdx].uploadBuffer, 0, &readRange, &s->cbuffers[cbIdx].mappedData);
        }
        s->cbuffers[cbIdx].cpuData = (unsigned char *)RL_CALLOC(1, cbDesc.Size);
        s->cbufferCount++;

        // Enumerate variables within this constant buffer
        for (UINT j = 0; j < cbDesc.Variables && s->uniformCount < RL_D3D12_MAX_UNIFORM_VARS; j++) {
            ID3D11ShaderReflectionVariable *varReflect =
                cbReflect->lpVtbl->GetVariableByIndex(cbReflect, j);
            D3D11_SHADER_VARIABLE_DESC varDesc;
            hr = varReflect->lpVtbl->GetDesc(varReflect, &varDesc);
            if (FAILED(hr)) continue;

            int uIdx = s->uniformCount;
            strncpy(s->uniforms[uIdx].name, varDesc.Name, sizeof(s->uniforms[uIdx].name) - 1);
            s->uniforms[uIdx].name[sizeof(s->uniforms[uIdx].name) - 1] = '\0';
            s->uniforms[uIdx].cbufferIndex = cbIdx;
            s->uniforms[uIdx].byteOffset = (int)varDesc.StartOffset;
            s->uniforms[uIdx].byteSize = (int)varDesc.Size;
            s->uniformCount++;

            // Expand struct arrays for member access (e.g. "lights[0].enabled")
            ID3D11ShaderReflectionType *varType = varReflect->lpVtbl->GetType(varReflect);
            if (varType) {
                D3D11_SHADER_TYPE_DESC typeDesc;
                hr = varType->lpVtbl->GetDesc(varType, &typeDesc);
                if (SUCCEEDED(hr) && typeDesc.Class == D3D_SVC_STRUCT && typeDesc.Elements > 0) {
                    UINT structStride = varDesc.Size / typeDesc.Elements;
                    for (UINT elem = 0; elem < typeDesc.Elements; elem++) {
                        for (UINT m = 0; m < typeDesc.Members && s->uniformCount < RL_D3D12_MAX_UNIFORM_VARS; m++) {
                            ID3D11ShaderReflectionType *memberType = varType->lpVtbl->GetMemberTypeByIndex(varType, m);
                            const char *memberName = varType->lpVtbl->GetMemberTypeName(varType, m);
                            if (!memberName || !memberType) continue;

                            D3D11_SHADER_TYPE_DESC memberTypeDesc;
                            hr = memberType->lpVtbl->GetDesc(memberType, &memberTypeDesc);
                            if (FAILED(hr)) continue;

                            int mIdx = s->uniformCount;
                            snprintf(s->uniforms[mIdx].name, sizeof(s->uniforms[mIdx].name),
                                     "%s[%u].%s", varDesc.Name, elem, memberName);
                            s->uniforms[mIdx].cbufferIndex = cbIdx;
                            s->uniforms[mIdx].byteOffset = (int)(varDesc.StartOffset + elem * structStride + memberTypeDesc.Offset);
                            int memberSize = 4;
                            if (memberTypeDesc.Class == D3D_SVC_SCALAR) memberSize = 4;
                            else if (memberTypeDesc.Class == D3D_SVC_VECTOR) memberSize = memberTypeDesc.Columns * 4;
                            else if (memberTypeDesc.Class == D3D_SVC_MATRIX_COLUMNS || memberTypeDesc.Class == D3D_SVC_MATRIX_ROWS)
                                memberSize = memberTypeDesc.Rows * memberTypeDesc.Columns * 4;
                            s->uniforms[mIdx].byteSize = memberSize;
                            s->uniformCount++;
                        }
                    }
                }
            }
        }
    }

    // Reflect texture/SRV bindings (pixel shader stage only)
    if (stage == 1) {
        for (UINT i = 0; i < shaderDesc.BoundResources && s->textureBindingCount < RL_D3D12_MAX_TEXTURE_BINDINGS; i++) {
            D3D11_SHADER_INPUT_BIND_DESC bindDesc;
            hr = reflect->lpVtbl->GetResourceBindingDesc(reflect, i, &bindDesc);
            if (FAILED(hr)) continue;
            if (bindDesc.Type == D3D_SIT_TEXTURE) {
                int tbIdx = s->textureBindingCount;
                strncpy(s->textureBindings[tbIdx].name, bindDesc.Name, sizeof(s->textureBindings[tbIdx].name) - 1);
                s->textureBindings[tbIdx].name[sizeof(s->textureBindings[tbIdx].name) - 1] = '\0';
                s->textureBindings[tbIdx].registerSlot = (int)bindDesc.BindPoint;
                s->textureBindingCount++;
                TRACELOG(RL_LOG_DEBUG, "SHADER: [stage %d] Texture binding '%s' at register(t%d)", stage, bindDesc.Name, (int)bindDesc.BindPoint);
            }
        }
    }

    reflect->lpVtbl->Release(reflect);
}

static void rlReflectShaderUniforms(unsigned int shaderIdx)
{
    rlD3D12Shader *s = &RLGL.shaders[shaderIdx];
    s->cbufferCount = 0;
    s->uniformCount = 0;
    s->textureBindingCount = 0;

    // Initialize material-map-to-register mapping to identity
    for (int i = 0; i < RL_MAX_MATERIAL_MAPS; i++) s->texMaterialMapToRegister[i] = i;

    // Reflect VS (stage=0) and PS (stage=1)
    rlReflectShaderStage(shaderIdx, s->vsBlob, 0);
    rlReflectShaderStage(shaderIdx, s->psBlob, 1);
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

    // Create PSOs (triangle and line topology)
    RLGL.shaders[idx].pso = rlCreatePipelineState(vsBlob, psBlob,
        RLGL.State.colorBlendEnabled, RLGL.State.currentBlendMode,
        RLGL.State.depthTestEnabled, RLGL.State.depthWriteEnabled,
        RLGL.State.backfaceCullingEnabled, RLGL.State.cullFaceMode,
        RLGL.State.wireMode, RLGL.State.scissorTestEnabled,
        D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE);
    RLGL.shaders[idx].psoLine = rlCreatePipelineState(vsBlob, psBlob,
        RLGL.State.colorBlendEnabled, RLGL.State.currentBlendMode,
        RLGL.State.depthTestEnabled, RLGL.State.depthWriteEnabled,
        RLGL.State.backfaceCullingEnabled, RLGL.State.cullFaceMode,
        RLGL.State.wireMode, RLGL.State.scissorTestEnabled,
        D3D12_PRIMITIVE_TOPOLOGY_TYPE_LINE);

    RLGL.shaders[idx].cbufferCount = 0;
    RLGL.shaders[idx].uniformCount = 0;
    RLGL.shaderCount++;

    // Reflect shader to discover cbuffers, uniforms, and texture bindings
    rlReflectShaderUniforms(idx);

    unsigned int id = idx + 1;
    TRACELOG(RL_LOG_INFO, "SHADER: [ID %i] D3D12 shader loaded (uniforms: %d, cbuffers: %d, texBindings: %d)",
             id, RLGL.shaders[idx].uniformCount, RLGL.shaders[idx].cbufferCount, RLGL.shaders[idx].textureBindingCount);
    return id;
}

unsigned int rlLoadShaderProgram(unsigned int vShaderId, unsigned int fShaderId) {
    (void)vShaderId; (void)fShaderId;
    return 0;
}

void rlUnloadShaderProgram(unsigned int id) {
    if (id == 0 || id > RLGL.shaderCount) return;
    unsigned int idx = id - 1;
    // Invalidate cached PSOs for this shader; defer any non-cached PSOs
    rlInvalidatePsoCacheForShader(id);
    if (RLGL.shaders[idx].pso && !rlIsPsoCached(RLGL.shaders[idx].pso)) {
        rlDeferPsoRelease(RLGL.shaders[idx].pso);
    }
    RLGL.shaders[idx].pso = NULL;
    if (RLGL.shaders[idx].psoLine && !rlIsPsoCached(RLGL.shaders[idx].psoLine)) {
        rlDeferPsoRelease(RLGL.shaders[idx].psoLine);
    }
    RLGL.shaders[idx].psoLine = NULL;
    if (RLGL.shaders[idx].vsBlob) { ID3D10Blob_Release(RLGL.shaders[idx].vsBlob); RLGL.shaders[idx].vsBlob = NULL; }
    if (RLGL.shaders[idx].psBlob) { ID3D10Blob_Release(RLGL.shaders[idx].psBlob); RLGL.shaders[idx].psBlob = NULL; }
    for (int i = 0; i < RLGL.shaders[idx].cbufferCount; i++) {
        if (RLGL.shaders[idx].cbuffers[i].uploadBuffer) { ID3D12Resource_Release(RLGL.shaders[idx].cbuffers[i].uploadBuffer); RLGL.shaders[idx].cbuffers[i].uploadBuffer = NULL; }
        if (RLGL.shaders[idx].cbuffers[i].cpuData) { RL_FREE(RLGL.shaders[idx].cbuffers[i].cpuData); RLGL.shaders[idx].cbuffers[i].cpuData = NULL; }
    }
}

//================================================================================
// Uniform location and value functions
//================================================================================
int rlGetLocationUniform(unsigned int shaderId, const char *uniformName) {
    if (shaderId == 0 || shaderId > RLGL.shaderCount || !uniformName) return -1;
    unsigned int idx = shaderId - 1;
    // Search cbuffer uniforms
    for (int i = 0; i < RLGL.shaders[idx].uniformCount; i++) {
        if (strcmp(RLGL.shaders[idx].uniforms[i].name, uniformName) == 0) return i;
    }
    // Search texture/SRV bindings (return special encoded value)
    for (int i = 0; i < RLGL.shaders[idx].textureBindingCount; i++) {
        if (strcmp(RLGL.shaders[idx].textureBindings[i].name, uniformName) == 0)
            return RL_D3D12_TEX_LOC_OFFSET + i;
    }
    return -1;
}

int rlGetLocationAttrib(unsigned int shaderId, const char *attribName) {
    (void)shaderId;
    if (!attribName) return -1;
    if (strcmp(attribName, "vertexPosition") == 0) return 0;
    if (strcmp(attribName, "vertexTexCoord") == 0) return 1;
    if (strcmp(attribName, "vertexNormal") == 0) return 2;
    if (strcmp(attribName, "vertexColor") == 0) return 3;
    if (strcmp(attribName, "vertexTangent") == 0) return -1;
    if (strcmp(attribName, "vertexTexCoord2") == 0) return -1;
    return -1;
}

void rlSetUniform(int locIndex, const void *value, int uniformType, int count) {
    if (locIndex < 0 || !value) return;
    unsigned int activeShader = d3d12_activeShaderId;
    if (activeShader == 0 || activeShader > RLGL.shaderCount) return;
    unsigned int shaderIdx = activeShader - 1;
    rlD3D12Shader *s = &RLGL.shaders[shaderIdx];

    // Handle texture binding locations
    if (locIndex >= RL_D3D12_TEX_LOC_OFFSET) {
        if (uniformType != RL_SHADER_UNIFORM_INT) return;
        int texBindIdx = locIndex - RL_D3D12_TEX_LOC_OFFSET;
        if (texBindIdx < 0 || texBindIdx >= s->textureBindingCount) return;
        int registerSlot = s->textureBindings[texBindIdx].registerSlot;
        int materialMapIndex = *(const int *)value;
        if (materialMapIndex >= 0 && materialMapIndex < RL_MAX_MATERIAL_MAPS) {
            s->texMaterialMapToRegister[materialMapIndex] = registerSlot;
            TRACELOG(RL_LOG_DEBUG, "SHADER: [ID %i] Texture '%s' mapped: material map %d -> register(t%d)",
                     (int)activeShader, s->textureBindings[texBindIdx].name, materialMapIndex, registerSlot);
        }
        return;
    }

    if (locIndex >= s->uniformCount) return;

    rlD3D12UniformVar *u = &s->uniforms[locIndex];
    rlD3D12CBufferInfo *cb = &s->cbuffers[u->cbufferIndex];

    int dataSize = 0;
    switch (uniformType) {
        case RL_SHADER_UNIFORM_FLOAT:    dataSize = sizeof(float) * count; break;
        case RL_SHADER_UNIFORM_VEC2:     dataSize = 2 * sizeof(float) * count; break;
        case RL_SHADER_UNIFORM_VEC3:     dataSize = 3 * sizeof(float) * count; break;
        case RL_SHADER_UNIFORM_VEC4:     dataSize = 4 * sizeof(float) * count; break;
        case RL_SHADER_UNIFORM_INT:      dataSize = sizeof(int) * count; break;
        case RL_SHADER_UNIFORM_IVEC2:    dataSize = 2 * sizeof(int) * count; break;
        case RL_SHADER_UNIFORM_IVEC3:    dataSize = 3 * sizeof(int) * count; break;
        case RL_SHADER_UNIFORM_IVEC4:    dataSize = 4 * sizeof(int) * count; break;
        case RL_SHADER_UNIFORM_UINT:     dataSize = sizeof(unsigned int) * count; break;
        case RL_SHADER_UNIFORM_SAMPLER2D: return;
        default: return;
    }
    if (dataSize > u->byteSize) dataSize = u->byteSize;

    // Update CPU-side copy and mark cbuffer dirty for deferred binding.
    // The actual ring buffer suballocation + GPU bind happens in rlFlushDirtyCBuffers()
    // which is called just before each draw call.  This eliminates redundant suballocations
    // that previously occurred when rlEnableShader re-bound ALL cbuffers and then individual
    // rlSetUniform* calls immediately re-bound the same cbuffers with updated data.
    if (cb->cpuData) {
        memcpy(cb->cpuData + u->byteOffset, value, dataSize);
        cb->dirty = true;
    }
}

void rlSetUniformMatrix(int locIndex, Matrix mat) {
    if (locIndex < 0) return;
    unsigned int activeShader = d3d12_activeShaderId;
    if (activeShader == 0 || activeShader > RLGL.shaderCount) return;
    unsigned int shaderIdx = activeShader - 1;
    rlD3D12Shader *s = &RLGL.shaders[shaderIdx];
    if (locIndex >= s->uniformCount) return;

    rlD3D12UniformVar *u = &s->uniforms[locIndex];
    rlD3D12CBufferInfo *cb = &s->cbuffers[u->cbufferIndex];

    rl_float16 matData = rlMatrixToFloatV(mat);
    int copySize = (u->byteSize < (int)sizeof(float)*16) ? u->byteSize : (int)sizeof(float)*16;
    if (cb->cpuData) {
        memcpy(cb->cpuData + u->byteOffset, matData.v, copySize);
        cb->dirty = true;
    }
}

void rlSetUniformMatrices(int locIndex, const Matrix *mat, int count) {
    if (locIndex < 0 || !mat || count <= 0) return;
    unsigned int activeShader = d3d12_activeShaderId;
    if (activeShader == 0 || activeShader > RLGL.shaderCount) return;
    unsigned int shaderIdx = activeShader - 1;
    rlD3D12Shader *s = &RLGL.shaders[shaderIdx];
    if (locIndex >= s->uniformCount) return;

    rlD3D12UniformVar *u = &s->uniforms[locIndex];
    rlD3D12CBufferInfo *cb = &s->cbuffers[u->cbufferIndex];

    if (cb->cpuData) {
        for (int i = 0; i < count && (u->byteOffset + (i + 1) * 64) <= cb->byteSize; i++) {
            rl_float16 matData = rlMatrixToFloatV(mat[i]);
            memcpy(cb->cpuData + u->byteOffset + i * 64, matData.v, 64);
        }
        cb->dirty = true;
    }
}

void rlSetUniformSampler(int locIndex, unsigned int textureId) { (void)locIndex; (void)textureId; }

void rlSetShader(unsigned int id, int *locs)
{
    if (RLGL.State.currentShaderId != id) {
        rlDrawRenderBatch(RLGL.currentBatch);
        RLGL.State.currentShaderId = id;
        RLGL.State.currentShaderLocs = locs;
    }
}
//================================================================================
// Draw functions
// NOTE: All draw calls must set IASetPrimitiveTopology before drawing.
// D3D12 resets ALL state on command list Reset(), so the topology is
// D3D_PRIMITIVE_TOPOLOGY_UNDEFINED by default — causing NVIDIA driver crashes.
//================================================================================
// Helper: ensure the currently bound PSO matches the latest render state.
// In D3D11, rlUpdateD3D11States() is called before EVERY draw.  In D3D12 the
// depth/cull/blend state is baked into the PSO, so we must re-select the PSO
// whenever the state has changed since rlEnableShader was last called.
static void rlEnsurePipelineState(void)
{
    if (RLGL.pipelineDirty) {
        rlUpdateD3D12PipelineState();
        unsigned int sid = d3d12_activeShaderId;
        if (sid > 0 && sid <= RLGL.shaderCount && RLGL.shaders[sid - 1].pso &&
            RLGL.shaders[sid - 1].pso != d3d12_lastBoundPSO) {
            ID3D12GraphicsCommandList_SetPipelineState(RLGL.commandList, RLGL.shaders[sid - 1].pso);
            d3d12_lastBoundPSO = RLGL.shaders[sid - 1].pso;
        }
    }
}

void rlDrawVertexArray(int offset, int count)
{
    if (!RLGL.commandList) return;
    
    rlEnsurePipelineState();  // Apply pending state changes (depth, cull, blend) — matches D3D11
    rlFlushDirtyCBuffers();   // Bind any pending uniform updates before drawing

    // SAFETY: Always rebind RT+DSV before every draw to guarantee the depth buffer is bound.
    // In D3D12, OMSetRenderTargets must be called to attach the DSV — without it, depth
    // test is silently disabled even though the PSO has DepthEnable=TRUE.
    if (RLGL.activeFramebuffer == 0 && RLGL.swapChain && RLGL.depthStencilBuffer) {
        UINT backIdx = d3d12_cachedBackBufferIndex;  // Use cached index (updated in rlClearScreenBuffers)
        if (RLGL.renderTargets[backIdx])
            ID3D12GraphicsCommandList_OMSetRenderTargets(RLGL.commandList, 1,
                &RLGL.renderTargetHandles[backIdx], FALSE, &RLGL.dsvBackbuffer);
    } else if (RLGL.activeFramebuffer > 0 && RLGL.activeFramebuffer <= RLGL.framebufferCount) {
        rlD3D12Framebuffer *fb = &RLGL.framebuffers[RLGL.activeFramebuffer - 1];
        if (fb->colorCount > 0 && fb->colorTexIds[0] > 0 && fb->colorTexIds[0] <= RLGL.textureCount) {
            D3D12_CPU_DESCRIPTOR_HANDLE rtvH = RLGL.textures[fb->colorTexIds[0] - 1].rtvHandle;
            D3D12_CPU_DESCRIPTOR_HANDLE *dsvPtr = NULL;
            D3D12_CPU_DESCRIPTOR_HANDLE dsvH = { 0 };
            if (fb->depthTexId > 0 && fb->depthTexId <= RLGL.textureCount) {
                dsvH = RLGL.textures[fb->depthTexId - 1].dsvHandle;
                dsvPtr = &dsvH;
            }
            ID3D12GraphicsCommandList_OMSetRenderTargets(RLGL.commandList, 1, &rtvH, FALSE, dsvPtr);
        }
    }

    ID3D12GraphicsCommandList_IASetPrimitiveTopology(RLGL.commandList, D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    ID3D12GraphicsCommandList_DrawInstanced(RLGL.commandList, count, 1, offset, 0);
}

void rlDrawVertexArrayElements(int offset, int count, const void *buffer)
{
    (void)buffer;
    if (!RLGL.commandList) return;
    
    rlEnsurePipelineState();  // Apply pending state changes (depth, cull, blend) — matches D3D11
    rlFlushDirtyCBuffers();   // Bind any pending uniform updates before drawing

    // SAFETY: Always rebind RT+DSV before every draw to guarantee the depth buffer is bound.
    if (RLGL.activeFramebuffer == 0 && RLGL.swapChain && RLGL.depthStencilBuffer) {
        UINT backIdx = d3d12_cachedBackBufferIndex;  // Use cached index (updated in rlClearScreenBuffers)
        if (RLGL.renderTargets[backIdx])
            ID3D12GraphicsCommandList_OMSetRenderTargets(RLGL.commandList, 1,
                &RLGL.renderTargetHandles[backIdx], FALSE, &RLGL.dsvBackbuffer);
    } else if (RLGL.activeFramebuffer > 0 && RLGL.activeFramebuffer <= RLGL.framebufferCount) {
        rlD3D12Framebuffer *fb = &RLGL.framebuffers[RLGL.activeFramebuffer - 1];
        if (fb->colorCount > 0 && fb->colorTexIds[0] > 0 && fb->colorTexIds[0] <= RLGL.textureCount) {
            D3D12_CPU_DESCRIPTOR_HANDLE rtvH = RLGL.textures[fb->colorTexIds[0] - 1].rtvHandle;
            D3D12_CPU_DESCRIPTOR_HANDLE *dsvPtr = NULL;
            D3D12_CPU_DESCRIPTOR_HANDLE dsvH = { 0 };
            if (fb->depthTexId > 0 && fb->depthTexId <= RLGL.textureCount) {
                dsvH = RLGL.textures[fb->depthTexId - 1].dsvHandle;
                dsvPtr = &dsvH;
            }
            ID3D12GraphicsCommandList_OMSetRenderTargets(RLGL.commandList, 1, &rtvH, FALSE, dsvPtr);
        }
    }

    // One-shot diagnostic: log depth state on first indexed draw
    if (!d3d12_depthDiagLogged) {
        d3d12_depthDiagLogged = true;
        TRACELOG(LOG_WARNING, "D3D12 DIAG: rlDrawVertexArrayElements first call: "
            "shader=%u depthTest=%d depthWrite=%d depthBuf=%p activeFB=%u pso=%p count=%d",
            d3d12_activeShaderId,
            (int)RLGL.State.depthTestEnabled,
            (int)RLGL.State.depthWriteEnabled,
            (void *)RLGL.depthStencilBuffer,
            RLGL.activeFramebuffer,
            (void *)d3d12_lastBoundPSO,
            count);
    }

    ID3D12GraphicsCommandList_IASetPrimitiveTopology(RLGL.commandList, D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    ID3D12GraphicsCommandList_DrawIndexedInstanced(RLGL.commandList, count, 1, offset, 0, 0);
}

void rlDrawVertexArrayInstanced(int offset, int count, int instances)
{
    if (!RLGL.commandList) return;
    
    rlEnsurePipelineState();  // Apply pending state changes (depth, cull, blend) — matches D3D11
    rlFlushDirtyCBuffers();   // Bind any pending uniform updates before drawing
    ID3D12GraphicsCommandList_IASetPrimitiveTopology(RLGL.commandList, D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    ID3D12GraphicsCommandList_DrawInstanced(RLGL.commandList, count, instances, offset, 0);
}

void rlDrawVertexArrayElementsInstanced(int offset, int count, const void *buffer, int instances)
{
    (void)buffer;
    if (!RLGL.commandList) return;
    
    rlEnsurePipelineState();  // Apply pending state changes (depth, cull, blend) — matches D3D11
    rlFlushDirtyCBuffers();   // Bind any pending uniform updates before drawing
    ID3D12GraphicsCommandList_IASetPrimitiveTopology(RLGL.commandList, D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    ID3D12GraphicsCommandList_DrawIndexedInstanced(RLGL.commandList, count, instances, offset, 0, 0);
}

//================================================================================
// Compute shader / SSBO stubs (not supported in D3D12 backend yet)
//================================================================================
unsigned int rlLoadComputeShaderProgram(unsigned int shaderId) { (void)shaderId; return 0; }
void rlComputeShaderDispatch(unsigned int groupX, unsigned int groupY, unsigned int groupZ) { (void)groupX; (void)groupY; (void)groupZ; }
unsigned int rlLoadShaderBuffer(unsigned int size, const void *data, int usageHint) { (void)size; (void)data; (void)usageHint; return 0; }
void rlUnloadShaderBuffer(unsigned int ssboId) { (void)ssboId; }
void rlUpdateShaderBuffer(unsigned int id, const void *data, unsigned int dataSize, unsigned int offset) { (void)id; (void)data; (void)dataSize; (void)offset; }
void rlReadShaderBuffer(unsigned int id, void *dest, unsigned int count, unsigned int offset) { (void)id; (void)dest; (void)count; (void)offset; }
void rlBindShaderBuffer(unsigned int id, unsigned int index) { (void)id; (void)index; }
unsigned int rlGetShaderBufferSize(unsigned int id) { (void)id; return 0; }
void rlCopyShaderBuffer(unsigned int destId, unsigned int srcId, unsigned int destOffset, unsigned int srcOffset, unsigned int count) { (void)destId; (void)srcId; (void)destOffset; (void)srcOffset; (void)count; }
void rlBindImageTexture(unsigned int id, unsigned int index, int format, bool readonly) { (void)id; (void)index; (void)format; (void)readonly; }

//================================================================================
// Default shader load/unload
//================================================================================
static void rlLoadShaderDefault(void)
{
    unsigned int shaderId = rlLoadShaderCode(defaultVShaderHLSL, defaultPShaderHLSL);
    RLGL.State.defaultShaderId = shaderId;

    if (shaderId > 0) {
        RLGL.State.defaultShaderLocs = (int *)RL_CALLOC(RL_MAX_SHADER_LOCATIONS, sizeof(int));
        for (int i = 0; i < RL_MAX_SHADER_LOCATIONS; i++) RLGL.State.defaultShaderLocs[i] = -1;

        RLGL.State.defaultShaderLocs[RL_SHADER_LOC_VERTEX_POSITION] = rlGetLocationAttrib(shaderId, "vertexPosition");
        RLGL.State.defaultShaderLocs[RL_SHADER_LOC_VERTEX_TEXCOORD01] = rlGetLocationAttrib(shaderId, "vertexTexCoord");
        RLGL.State.defaultShaderLocs[RL_SHADER_LOC_VERTEX_COLOR] = rlGetLocationAttrib(shaderId, "vertexColor");
        RLGL.State.defaultShaderLocs[RL_SHADER_LOC_VERTEX_NORMAL] = rlGetLocationAttrib(shaderId, "vertexNormal");

        RLGL.State.defaultShaderLocs[RL_SHADER_LOC_MATRIX_MVP] = rlGetLocationUniform(shaderId, "mvp");
        RLGL.State.defaultShaderLocs[RL_SHADER_LOC_COLOR_DIFFUSE] = rlGetLocationUniform(shaderId, "colDiffuse");
        RLGL.State.defaultShaderLocs[RL_SHADER_LOC_MAP_DIFFUSE] = rlGetLocationUniform(shaderId, "texture0");

        RLGL.State.currentShaderId = shaderId;
        RLGL.State.currentShaderLocs = RLGL.State.defaultShaderLocs;
    }
}

static void rlUnloadShaderDefault(void)
{
    if (RLGL.State.defaultShaderId > 0) {
        rlUnloadShaderProgram(RLGL.State.defaultShaderId);
        if (RLGL.State.defaultShaderLocs) { RL_FREE(RLGL.State.defaultShaderLocs); RLGL.State.defaultShaderLocs = NULL; }
        RLGL.State.defaultShaderId = 0;
    }
}

//================================================================================
// Matrix getters/setters and math utilities
//================================================================================
Matrix rlGetMatrixModelview(void)
{
    Matrix m = RLGL.State.modelview;
    return m;
}

Matrix rlGetMatrixProjection(void)
{
    Matrix m = RLGL.State.projection;
    return m;
}

Matrix rlGetMatrixTransform(void)
{
    Matrix m = RLGL.State.transform;
    return m;
}

Matrix rlGetMatrixProjectionStereo(int eye)
{
    return RLGL.State.projectionStereo[eye];
}

Matrix rlGetMatrixViewOffsetStereo(int eye)
{
    return RLGL.State.viewOffsetStereo[eye];
}

void rlSetMatrixProjection(Matrix projection)
{
    RLGL.State.projection = projection;
}

void rlSetMatrixModelview(Matrix view)
{
    RLGL.State.modelview = view;
}

void rlSetMatrixProjectionStereo(Matrix right, Matrix left)
{
    RLGL.State.projectionStereo[0] = right;
    RLGL.State.projectionStereo[1] = left;
}

void rlSetMatrixViewOffsetStereo(Matrix right, Matrix left)
{
    RLGL.State.viewOffsetStereo[0] = right;
    RLGL.State.viewOffsetStereo[1] = left;
}

//================================================================================
// Math utility functions
//================================================================================
rl_float16 rlMatrixToFloatV(Matrix mat)
{
    rl_float16 result = { 0 };
    result.v[0]  = mat.m0; result.v[1]  = mat.m1; result.v[2]  = mat.m2; result.v[3]  = mat.m3;
    result.v[4]  = mat.m4; result.v[5]  = mat.m5; result.v[6]  = mat.m6; result.v[7]  = mat.m7;
    result.v[8]  = mat.m8; result.v[9]  = mat.m9; result.v[10] = mat.m10; result.v[11] = mat.m11;
    result.v[12] = mat.m12; result.v[13] = mat.m13; result.v[14] = mat.m14; result.v[15] = mat.m15;
    return result;
}

Matrix rlMatrixIdentity(void)
{
    Matrix result = {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f
    };
    return result;
}

Matrix rlMatrixMultiply(Matrix left, Matrix right)
{
    Matrix result = { 0 };
    result.m0 = left.m0*right.m0 + left.m1*right.m4 + left.m2*right.m8 + left.m3*right.m12;
    result.m1 = left.m0*right.m1 + left.m1*right.m5 + left.m2*right.m9 + left.m3*right.m13;
    result.m2 = left.m0*right.m2 + left.m1*right.m6 + left.m2*right.m10 + left.m3*right.m14;
    result.m3 = left.m0*right.m3 + left.m1*right.m7 + left.m2*right.m11 + left.m3*right.m15;
    result.m4 = left.m4*right.m0 + left.m5*right.m4 + left.m6*right.m8 + left.m7*right.m12;
    result.m5 = left.m4*right.m1 + left.m5*right.m5 + left.m6*right.m9 + left.m7*right.m13;
    result.m6 = left.m4*right.m2 + left.m5*right.m6 + left.m6*right.m10 + left.m7*right.m14;
    result.m7 = left.m4*right.m3 + left.m5*right.m7 + left.m6*right.m11 + left.m7*right.m15;
    result.m8 = left.m8*right.m0 + left.m9*right.m4 + left.m10*right.m8 + left.m11*right.m12;
    result.m9 = left.m8*right.m1 + left.m9*right.m5 + left.m10*right.m9 + left.m11*right.m13;
    result.m10 = left.m8*right.m2 + left.m9*right.m6 + left.m10*right.m10 + left.m11*right.m14;
    result.m11 = left.m8*right.m3 + left.m9*right.m7 + left.m10*right.m11 + left.m11*right.m15;
    result.m12 = left.m12*right.m0 + left.m13*right.m4 + left.m14*right.m8 + left.m15*right.m12;
    result.m13 = left.m12*right.m1 + left.m13*right.m5 + left.m14*right.m9 + left.m15*right.m13;
    result.m14 = left.m12*right.m2 + left.m13*right.m6 + left.m14*right.m10 + left.m15*right.m14;
    result.m15 = left.m12*right.m3 + left.m13*right.m7 + left.m14*right.m11 + left.m15*right.m15;
    return result;
}

Matrix rlMatrixTranspose(Matrix mat)
{
    Matrix result = { 0 };
    result.m0 = mat.m0; result.m1 = mat.m4; result.m2 = mat.m8;  result.m3 = mat.m12;
    result.m4 = mat.m1; result.m5 = mat.m5; result.m6 = mat.m9;  result.m7 = mat.m13;
    result.m8 = mat.m2; result.m9 = mat.m6; result.m10 = mat.m10; result.m11 = mat.m14;
    result.m12 = mat.m3; result.m13 = mat.m7; result.m14 = mat.m11; result.m15 = mat.m15;
    return result;
}

Matrix rlMatrixInvert(Matrix mat)
{
    Matrix result = { 0 };
    float a00 = mat.m0, a01 = mat.m1, a02 = mat.m2, a03 = mat.m3;
    float a10 = mat.m4, a11 = mat.m5, a12 = mat.m6, a13 = mat.m7;
    float a20 = mat.m8, a21 = mat.m9, a22 = mat.m10, a23 = mat.m11;
    float a30 = mat.m12, a31 = mat.m13, a32 = mat.m14, a33 = mat.m15;
    float b00 = a00*a11 - a01*a10;
    float b01 = a00*a12 - a02*a10;
    float b02 = a00*a13 - a03*a10;
    float b03 = a01*a12 - a02*a11;
    float b04 = a01*a13 - a03*a11;
    float b05 = a02*a13 - a03*a12;
    float b06 = a20*a31 - a21*a30;
    float b07 = a20*a32 - a22*a30;
    float b08 = a20*a33 - a23*a30;
    float b09 = a21*a32 - a22*a31;
    float b10 = a21*a33 - a23*a31;
    float b11 = a22*a33 - a23*a32;
    float invDet = 1.0f/(b00*b11 - b01*b10 + b02*b09 + b03*b08 - b04*b07 + b05*b06);
    result.m0 = (a11*b11 - a12*b10 + a13*b09)*invDet;
    result.m1 = (-a01*b11 + a02*b10 - a03*b09)*invDet;
    result.m2 = (a31*b05 - a32*b04 + a33*b03)*invDet;
    result.m3 = (-a21*b05 + a22*b04 - a23*b03)*invDet;
    result.m4 = (-a10*b11 + a12*b08 - a13*b07)*invDet;
    result.m5 = (a00*b11 - a02*b08 + a03*b07)*invDet;
    result.m6 = (-a30*b05 + a32*b02 - a33*b01)*invDet;
    result.m7 = (a20*b05 - a22*b02 + a23*b01)*invDet;
    result.m8 = (a10*b10 - a11*b08 + a13*b06)*invDet;
    result.m9 = (-a00*b10 + a01*b08 - a03*b06)*invDet;
    result.m10 = (a30*b04 - a31*b02 + a33*b00)*invDet;
    result.m11 = (-a20*b04 + a21*b02 - a23*b00)*invDet;
    result.m12 = (-a10*b09 + a11*b07 - a12*b06)*invDet;
    result.m13 = (a00*b09 - a01*b07 + a02*b06)*invDet;
    result.m14 = (-a30*b03 + a31*b01 - a32*b00)*invDet;
    result.m15 = (a20*b03 - a21*b01 + a22*b00)*invDet;
    return result;
}

int rlGetPixelDataSize(int width, int height, int format)
{
    int bpp = 0;
    switch (format) {
        case RL_PIXELFORMAT_UNCOMPRESSED_GRAYSCALE: bpp = 8; break;
        case RL_PIXELFORMAT_UNCOMPRESSED_GRAY_ALPHA:
        case RL_PIXELFORMAT_UNCOMPRESSED_R5G6B5:
        case RL_PIXELFORMAT_UNCOMPRESSED_R5G5B5A1:
        case RL_PIXELFORMAT_UNCOMPRESSED_R4G4B4A4: bpp = 16; break;
        case RL_PIXELFORMAT_UNCOMPRESSED_R8G8B8: bpp = 24; break;
        case RL_PIXELFORMAT_UNCOMPRESSED_R8G8B8A8: bpp = 32; break;
        case RL_PIXELFORMAT_UNCOMPRESSED_R32: bpp = 32; break;
        case RL_PIXELFORMAT_UNCOMPRESSED_R32G32B32: bpp = 32*3; break;
        case RL_PIXELFORMAT_UNCOMPRESSED_R32G32B32A32: bpp = 32*4; break;
        case RL_PIXELFORMAT_UNCOMPRESSED_R16: bpp = 16; break;
        case RL_PIXELFORMAT_UNCOMPRESSED_R16G16B16: bpp = 16*3; break;
        case RL_PIXELFORMAT_UNCOMPRESSED_R16G16B16A16: bpp = 16*4; break;
        case RL_PIXELFORMAT_COMPRESSED_DXT1_RGB:
        case RL_PIXELFORMAT_COMPRESSED_DXT1_RGBA:
        case RL_PIXELFORMAT_COMPRESSED_ETC1_RGB:
        case RL_PIXELFORMAT_COMPRESSED_ETC2_RGB:
        case RL_PIXELFORMAT_COMPRESSED_PVRT_RGB:
        case RL_PIXELFORMAT_COMPRESSED_PVRT_RGBA: bpp = 4; break;
        case RL_PIXELFORMAT_COMPRESSED_DXT3_RGBA:
        case RL_PIXELFORMAT_COMPRESSED_DXT5_RGBA:
        case RL_PIXELFORMAT_COMPRESSED_ETC2_EAC_RGBA:
        case RL_PIXELFORMAT_COMPRESSED_ASTC_4x4_RGBA: bpp = 8; break;
        case RL_PIXELFORMAT_COMPRESSED_ASTC_8x8_RGBA: bpp = 2; break;
        default: break;
    }
    int dataSize = (width*height*bpp + 7)/8;
    if (width < 4 && height < 4) {
        if ((format >= RL_PIXELFORMAT_COMPRESSED_DXT1_RGB) && (format <= RL_PIXELFORMAT_COMPRESSED_ASTC_8x8_RGBA))
            dataSize = (bpp < 8) ? 8 : 16;
    }
    return dataSize;
}

//================================================================================
// Screen pixel readback
//================================================================================
unsigned char *rlReadScreenPixels(int width, int height)
{
    unsigned char *data = (unsigned char *)RL_CALLOC(width * height * 4, sizeof(unsigned char));
    if (!data || !RLGL.device || !RLGL.commandQueue) return data;

    // Determine the bytes-per-row aligned to D3D12_TEXTURE_DATA_PITCH_ALIGNMENT
    UINT rowPitch = (UINT)(width * 4);
    UINT alignedRowPitch = (rowPitch + (D3D12_TEXTURE_DATA_PITCH_ALIGNMENT - 1)) & ~(D3D12_TEXTURE_DATA_PITCH_ALIGNMENT - 1);
    UINT64 totalSize = (UINT64)alignedRowPitch * (UINT64)height;

    // Create a readback buffer
    D3D12_HEAP_PROPERTIES readbackHeapProps = { 0 };
    readbackHeapProps.Type = D3D12_HEAP_TYPE_READBACK;

    D3D12_RESOURCE_DESC readbackDesc = { 0 };
    readbackDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    readbackDesc.Width = totalSize;
    readbackDesc.Height = 1;
    readbackDesc.DepthOrArraySize = 1;
    readbackDesc.MipLevels = 1;
    readbackDesc.Format = DXGI_FORMAT_UNKNOWN;
    readbackDesc.SampleDesc.Count = 1;
    readbackDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;

    ID3D12Resource *readbackBuffer = NULL;
    HRESULT hr = ID3D12Device_CreateCommittedResource(RLGL.device, &readbackHeapProps,
        D3D12_HEAP_FLAG_NONE, &readbackDesc, D3D12_RESOURCE_STATE_COPY_DEST,
        NULL, &IID_ID3D12Resource, (void **)&readbackBuffer);
    if (FAILED(hr)) return data;

    // Get current back buffer
    UINT bbIdx = IDXGISwapChain3_GetCurrentBackBufferIndex(RLGL.swapChain);
    ID3D12Resource *backBuffer = RLGL.renderTargets[bbIdx];

    // Reset command allocator and list for a one-shot copy
    rlD3D12FrameResources *frame = &RLGL.frames[RLGL.currentFrame];
    ID3D12CommandAllocator_Reset(frame->commandAllocator);
    ID3D12GraphicsCommandList_Reset(RLGL.commandList, frame->commandAllocator, NULL);

    // Transition back buffer: PRESENT -> COPY_SOURCE
    D3D12_RESOURCE_BARRIER barrierToCopy = { 0 };
    barrierToCopy.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrierToCopy.Transition.pResource = backBuffer;
    barrierToCopy.Transition.StateBefore = D3D12_RESOURCE_STATE_PRESENT;
    barrierToCopy.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
    barrierToCopy.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    ID3D12GraphicsCommandList_ResourceBarrier(RLGL.commandList, 1, &barrierToCopy);

    // Describe copy destination (buffer footprint)
    D3D12_TEXTURE_COPY_LOCATION dst = { 0 };
    dst.pResource = readbackBuffer;
    dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    dst.PlacedFootprint.Offset = 0;
    dst.PlacedFootprint.Footprint.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    dst.PlacedFootprint.Footprint.Width = (UINT)width;
    dst.PlacedFootprint.Footprint.Height = (UINT)height;
    dst.PlacedFootprint.Footprint.Depth = 1;
    dst.PlacedFootprint.Footprint.RowPitch = alignedRowPitch;

    D3D12_TEXTURE_COPY_LOCATION src = { 0 };
    src.pResource = backBuffer;
    src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    src.SubresourceIndex = 0;

    D3D12_BOX srcBox = { 0, 0, 0, (UINT)width, (UINT)height, 1 };
    ID3D12GraphicsCommandList_CopyTextureRegion(RLGL.commandList, &dst, 0, 0, 0, &src, &srcBox);

    // Transition back buffer: COPY_SOURCE -> PRESENT
    D3D12_RESOURCE_BARRIER barrierToPresent = { 0 };
    barrierToPresent.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrierToPresent.Transition.pResource = backBuffer;
    barrierToPresent.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_SOURCE;
    barrierToPresent.Transition.StateAfter = D3D12_RESOURCE_STATE_PRESENT;
    barrierToPresent.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    ID3D12GraphicsCommandList_ResourceBarrier(RLGL.commandList, 1, &barrierToPresent);

    ID3D12GraphicsCommandList_Close(RLGL.commandList);

    // Execute and wait for GPU
    ID3D12CommandList *cmdLists[] = { (ID3D12CommandList *)RLGL.commandList };
    ID3D12CommandQueue_ExecuteCommandLists(RLGL.commandQueue, 1, cmdLists);

    // Signal and wait using the global monotonic fence value
    UINT64 fenceVal = ++RLGL.fenceValue;
    frame->fenceValue = fenceVal;
    ID3D12CommandQueue_Signal(RLGL.commandQueue, RLGL.fence, fenceVal);
    if (ID3D12Fence_GetCompletedValue(RLGL.fence) < fenceVal) {
        ID3D12Fence_SetEventOnCompletion(RLGL.fence, fenceVal, RLGL.fenceEvent);
        WaitForSingleObject(RLGL.fenceEvent, INFINITE);
    }

    // Map readback buffer and copy row by row (handle pitch alignment)
    void *mapped = NULL;
    D3D12_RANGE readRange = { 0, totalSize };
    hr = ID3D12Resource_Map(readbackBuffer, 0, &readRange, &mapped);
    if (SUCCEEDED(hr)) {
        const unsigned char *src_ptr = (const unsigned char *)mapped;
        for (int y = 0; y < height; y++) {
            // D3D12 back buffer is top-down, same as raylib's expected output
            memcpy(data + y * width * 4, src_ptr + y * alignedRowPitch, width * 4);
        }
        D3D12_RANGE writeRange = { 0, 0 };
        ID3D12Resource_Unmap(readbackBuffer, 0, &writeRange);
    }

    ID3D12Resource_Release(readbackBuffer);

    // Re-open the command list so the next frame's draw calls (rlClearScreenBuffers etc.) work normally
    ID3D12CommandAllocator_Reset(frame->commandAllocator);
    ID3D12GraphicsCommandList_Reset(RLGL.commandList, frame->commandAllocator, NULL);

    return data;
}

//================================================================================
// Headless render viewport stubs
//================================================================================
void rlSetHeadlessRenderViewport(int width, int height) { (void)width; (void)height; }
void rlUnsetHeadlessRenderViewport(void) { }

//================================================================================
// Quick draw helpers
//================================================================================
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
        // Left face
        rlNormal3f(-1,0,0); rlVertex3f(-1,1,1); rlNormal3f(-1,0,0); rlVertex3f(-1,1,-1); rlNormal3f(-1,0,0); rlVertex3f(-1,-1,-1);
        rlNormal3f(-1,0,0); rlVertex3f(-1,-1,-1); rlNormal3f(-1,0,0); rlVertex3f(-1,-1,1); rlNormal3f(-1,0,0); rlVertex3f(-1,1,1);
        // Right face
        rlNormal3f(1,0,0); rlVertex3f(1,1,1); rlNormal3f(1,0,0); rlVertex3f(1,-1,-1); rlNormal3f(1,0,0); rlVertex3f(1,1,-1);
        rlNormal3f(1,0,0); rlVertex3f(1,-1,-1); rlNormal3f(1,0,0); rlVertex3f(1,1,1); rlNormal3f(1,0,0); rlVertex3f(1,-1,1);
        // Bottom face
        rlNormal3f(0,-1,0); rlVertex3f(-1,-1,-1); rlNormal3f(0,-1,0); rlVertex3f(1,-1,-1); rlNormal3f(0,-1,0); rlVertex3f(1,-1,1);
        rlNormal3f(0,-1,0); rlVertex3f(1,-1,1); rlNormal3f(0,-1,0); rlVertex3f(-1,-1,1); rlNormal3f(0,-1,0); rlVertex3f(-1,-1,-1);
        // Top face
        rlNormal3f(0,1,0); rlVertex3f(-1,1,-1); rlNormal3f(0,1,0); rlVertex3f(1,1,1); rlNormal3f(0,1,0); rlVertex3f(1,1,-1);
        rlNormal3f(0,1,0); rlVertex3f(1,1,1); rlNormal3f(0,1,0); rlVertex3f(-1,1,-1); rlNormal3f(0,1,0); rlVertex3f(-1,1,1);
    rlEnd();
    rlDrawRenderBatchActive();
}

#endif // GRAPHICS_API_DIRECT3D12