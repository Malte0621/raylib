/**********************************************************************************************
*
*   rl_backend_d3d11 - Direct3D 11 rendering backend for rlgl
*
*   DESCRIPTION:
*       Complete Direct3D 11 implementation of the rlgl rendering API.
*       Provides all functions required by rlgl.h when GRAPHICS_API_DIRECT3D11 is defined.
*
*   COMPILATION:
*       Only compiled when GRAPHICS_API_DIRECT3D11 is defined (Windows only).
*       Requires: d3d11.lib, d3dcompiler.lib, dxgi.lib
*
*   LICENSE: zlib/libpng (same as raylib)
*
**********************************************************************************************/

#if defined(GRAPHICS_API_DIRECT3D11)

#include "rlgl.h"
#include "rl_backend.h"

// Required Windows/D3D11 headers
#ifndef WIN32_LEAN_AND_MEAN
    #define WIN32_LEAN_AND_MEAN
#endif

// Ensure COM C-interface macros are available (ID3D11Device_CreateBuffer, etc.)
// These macros expand to vtable calls and are only defined when CINTERFACE/COBJMACROS are set.
#ifndef CINTERFACE
    #define CINTERFACE
#endif
#ifndef COBJMACROS
    #define COBJMACROS
#endif

#include <windows.h>
#include <d3d11.h>
#include <d3d11shader.h>
#include <d3dcompiler.h>
#include <dxgi.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "d3dcompiler.lib")
#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "dxguid.lib")

// rlgl.h is included by the translation unit that defines RLGL_IMPLEMENTATION
// We reference types from it. Ensure defines are available.
#ifndef PI
    #define PI 3.14159265358979323846f
#endif
#ifndef DEG2RAD
    #define DEG2RAD (PI/180.0f)
#endif

// Support TRACELOG macros
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
#define RL_D3D11_MAX_TEXTURES     4096
#define RL_D3D11_MAX_SHADERS      256
#define RL_D3D11_MAX_BUFFERS      4096
#define RL_D3D11_MAX_FRAMEBUFFERS 256
#define RL_D3D11_MAX_CBUFFERS     16    // Max constant buffers per shader
#define RL_D3D11_MAX_UNIFORM_VARS 64    // Max tracked uniform variables per shader

//----------------------------------------------------------------------------------
// Types - Resource tracking (maps unsigned int IDs to D3D11 objects)
//----------------------------------------------------------------------------------
typedef struct {
    ID3D11Texture2D *texture;
    ID3D11ShaderResourceView *srv;
    ID3D11SamplerState *sampler;
    int width, height, format;
    bool isCubemap;
    bool isDepth;
    ID3D11DepthStencilView *dsv;          // Only for depth textures
    ID3D11RenderTargetView *rtv;          // Only for render targets
} rlD3D11Texture;

// Per-constant-buffer info (tracked per shader via reflection)
typedef struct {
    int stage;               // 0 = VS, 1 = PS
    int registerSlot;        // register(bN) slot
    int byteSize;            // Total cbuffer size (16-byte aligned)
    ID3D11Buffer *buffer;    // D3D11 GPU constant buffer
    unsigned char *cpuData;  // CPU-side copy for partial updates
} rlD3D11CBufferInfo;

// Per-uniform variable info (tracked per shader via reflection)
typedef struct {
    char name[128];
    int cbufferIndex;        // Index into shader's cbuffers[] array
    int byteOffset;          // Offset within the constant buffer
    int byteSize;            // Size of this variable in bytes
} rlD3D11UniformVar;

// Per-shader texture binding info (from reflection)
typedef struct {
    char name[128];
    int registerSlot;        // register(tN) slot in HLSL
} rlD3D11TextureBindingInfo;

#define RL_D3D11_MAX_TEXTURE_BINDINGS 16
#define RL_D3D11_TEX_LOC_OFFSET 10000   // rlGetLocationUniform returns 10000+idx for texture bindings
#define RL_MAX_MATERIAL_MAPS 12          // Max material maps tracked for register mapping

typedef struct {
    ID3D11VertexShader *vertexShader;
    ID3D11PixelShader *pixelShader;
    ID3D11InputLayout *inputLayout;
    ID3DBlob *vsBlob;
    ID3DBlob *psBlob;
    // Constant buffer tracking (populated by shader reflection)
    rlD3D11CBufferInfo cbuffers[RL_D3D11_MAX_CBUFFERS];
    int cbufferCount;
    // Uniform variable tracking (populated by shader reflection)
    rlD3D11UniformVar uniforms[RL_D3D11_MAX_UNIFORM_VARS];
    int uniformCount;
    // Texture binding tracking (populated by shader reflection)
    rlD3D11TextureBindingInfo textureBindings[RL_D3D11_MAX_TEXTURE_BINDINGS];
    int textureBindingCount;
    // Material-map-index → D3D11 register slot mapping
    // Set via SetShaderValue(shader, texLoc, &materialMapIndex, INT)
    // Default: identity (slot i → register i)
    int texMaterialMapToRegister[RL_MAX_MATERIAL_MAPS];
} rlD3D11Shader;

typedef struct {
    ID3D11Buffer *buffer;
    int size;
    bool dynamic;
    bool isIndex;
} rlD3D11Buffer;

typedef struct {
    ID3D11RenderTargetView *rtv[8];
    ID3D11DepthStencilView *dsv;
    int colorCount;
    bool complete;
} rlD3D11Framebuffer;

//----------------------------------------------------------------------------------
// Constant buffer structures (must be 16-byte aligned)
//----------------------------------------------------------------------------------
typedef struct RL_ALIGN(16) {
    float mvp[16];
} rlD3D11MatrixCB;

typedef struct RL_ALIGN(16) {
    float colDiffuse[4];
} rlD3D11ColorCB;

//----------------------------------------------------------------------------------
// Internal rlgl state for D3D11
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
        int cullFaceMode;     // 0=back, 1=front
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

    // D3D11 device objects
    ID3D11Device *device;
    ID3D11DeviceContext *context;
    D3D_FEATURE_LEVEL featureLevel;

    // Default constant buffers
    ID3D11Buffer *cbMatrix;
    ID3D11Buffer *cbColor;

    // Default sampler
    ID3D11SamplerState *defaultSampler;

    // State objects (cached)
    ID3D11BlendState *currentBlendState;
    ID3D11DepthStencilState *currentDSState;
    ID3D11RasterizerState *currentRSState;
    bool stateDirty;

    // Resource tracking arrays
    rlD3D11Texture textures[RL_D3D11_MAX_TEXTURES];
    unsigned int textureCount;

    rlD3D11Shader shaders[RL_D3D11_MAX_SHADERS];
    unsigned int shaderCount;

    rlD3D11Buffer buffers[RL_D3D11_MAX_BUFFERS];
    unsigned int bufferCount;

    rlD3D11Framebuffer framebuffers[RL_D3D11_MAX_FRAMEBUFFERS];
    unsigned int framebufferCount;

    // Current active framebuffer (0 = default/backbuffer)
    unsigned int activeFramebuffer;

    // Swap chain (default back buffer presentation)
    IDXGISwapChain *swapChain;
    ID3D11RenderTargetView *backbufferRTV;
    ID3D11Texture2D *depthStencilTexture;
    ID3D11DepthStencilView *depthStencilView;

    // Batch vertex/index buffers on GPU
    ID3D11Buffer *batchVertexBuffers[4];    // pos, texcoord, normal, color
    ID3D11Buffer *batchIndexBuffer;
    int batchBufferSize;

} rlglData;

//----------------------------------------------------------------------------------
// Forward declarations for internal helpers
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
static void rlUpdateD3D11States(void);
static DXGI_FORMAT rlGetDXGIFormat(int format);
static void rlReflectShaderUniforms(unsigned int shaderIdx);

// External: raylib provides this to get the native window handle (HWND)
extern void *GetWindowHandle(void);

//----------------------------------------------------------------------------------
// Global state
//----------------------------------------------------------------------------------
static double rlCullDistanceNear = RL_CULL_DISTANCE_NEAR;
static double rlCullDistanceFar = RL_CULL_DISTANCE_FAR;
static rlglData RLGL = { 0 };

// D3D11-specific rendering state (mesh path)
static unsigned int d3d11_activeShaderId = 0;  // Shader currently bound on GPU (set by rlEnableShader, analogous to glUseProgram)
static int d3d11_activeTextureSlot = 0;     // Current active texture slot for rlEnableTexture
static unsigned int d3d11_pendingVBO = 0;   // Pending VBO ID for deferred binding (set by rlEnableVertexBuffer, consumed by rlSetVertexAttribute)

// Default white vertex buffer for meshes without vertex colors
static ID3D11Buffer *d3d11_defaultWhiteVBO = NULL;
#define RL_D3D11_DEFAULT_WHITE_VBO_VERTS 65536  // 256KB total (4 bytes per vertex)

//----------------------------------------------------------------------------------
// Default HLSL Shaders (embedded as strings)
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
// Helper: Create D3D11 buffer
//----------------------------------------------------------------------------------
static ID3D11Buffer *rlCreateD3D11Buffer(const void *data, UINT byteWidth, UINT bindFlags, bool dynamic)
{
    D3D11_BUFFER_DESC desc = { 0 };
    desc.ByteWidth = byteWidth;
    desc.Usage = dynamic ? D3D11_USAGE_DYNAMIC : D3D11_USAGE_DEFAULT;
    desc.BindFlags = bindFlags;
    desc.CPUAccessFlags = dynamic ? D3D11_CPU_ACCESS_WRITE : 0;

    D3D11_SUBRESOURCE_DATA initData = { 0 };
    initData.pSysMem = data;

    ID3D11Buffer *buffer = NULL;
    HRESULT hr = ID3D11Device_CreateBuffer(RLGL.device, &desc, data ? &initData : NULL, &buffer);
    if (FAILED(hr)) return NULL;
    return buffer;
}

//----------------------------------------------------------------------------------
// Helper: Update D3D11 states (blend, depth-stencil, rasterizer)
//----------------------------------------------------------------------------------
static void rlUpdateD3D11States(void)
{
    if (!RLGL.stateDirty || !RLGL.device) return;

    // Release old states
    if (RLGL.currentBlendState) { ID3D11BlendState_Release(RLGL.currentBlendState); RLGL.currentBlendState = NULL; }
    if (RLGL.currentDSState) { ID3D11DepthStencilState_Release(RLGL.currentDSState); RLGL.currentDSState = NULL; }
    if (RLGL.currentRSState) { ID3D11RasterizerState_Release(RLGL.currentRSState); RLGL.currentRSState = NULL; }

    // Blend state
    {
        D3D11_BLEND_DESC bd = { 0 };
        bd.RenderTarget[0].BlendEnable = RLGL.State.colorBlendEnabled;
        bd.RenderTarget[0].RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL;
        if (RLGL.State.colorBlendEnabled) {
            switch (RLGL.State.currentBlendMode) {
                case 0: // RL_BLEND_ALPHA
                    bd.RenderTarget[0].SrcBlend = D3D11_BLEND_SRC_ALPHA;
                    bd.RenderTarget[0].DestBlend = D3D11_BLEND_INV_SRC_ALPHA;
                    bd.RenderTarget[0].BlendOp = D3D11_BLEND_OP_ADD;
                    bd.RenderTarget[0].SrcBlendAlpha = D3D11_BLEND_ONE;
                    bd.RenderTarget[0].DestBlendAlpha = D3D11_BLEND_INV_SRC_ALPHA;
                    bd.RenderTarget[0].BlendOpAlpha = D3D11_BLEND_OP_ADD;
                    break;
                case 1: // RL_BLEND_ADDITIVE
                    bd.RenderTarget[0].SrcBlend = D3D11_BLEND_SRC_ALPHA;
                    bd.RenderTarget[0].DestBlend = D3D11_BLEND_ONE;
                    bd.RenderTarget[0].BlendOp = D3D11_BLEND_OP_ADD;
                    bd.RenderTarget[0].SrcBlendAlpha = D3D11_BLEND_ONE;
                    bd.RenderTarget[0].DestBlendAlpha = D3D11_BLEND_ONE;
                    bd.RenderTarget[0].BlendOpAlpha = D3D11_BLEND_OP_ADD;
                    break;
                case 2: // RL_BLEND_MULTIPLIED
                    bd.RenderTarget[0].SrcBlend = D3D11_BLEND_DEST_COLOR;
                    bd.RenderTarget[0].DestBlend = D3D11_BLEND_INV_SRC_ALPHA;
                    bd.RenderTarget[0].BlendOp = D3D11_BLEND_OP_ADD;
                    bd.RenderTarget[0].SrcBlendAlpha = D3D11_BLEND_ONE;
                    bd.RenderTarget[0].DestBlendAlpha = D3D11_BLEND_INV_SRC_ALPHA;
                    bd.RenderTarget[0].BlendOpAlpha = D3D11_BLEND_OP_ADD;
                    break;
                default: // Alpha blend as fallback
                    bd.RenderTarget[0].SrcBlend = D3D11_BLEND_SRC_ALPHA;
                    bd.RenderTarget[0].DestBlend = D3D11_BLEND_INV_SRC_ALPHA;
                    bd.RenderTarget[0].BlendOp = D3D11_BLEND_OP_ADD;
                    bd.RenderTarget[0].SrcBlendAlpha = D3D11_BLEND_ONE;
                    bd.RenderTarget[0].DestBlendAlpha = D3D11_BLEND_INV_SRC_ALPHA;
                    bd.RenderTarget[0].BlendOpAlpha = D3D11_BLEND_OP_ADD;
                    break;
            }
        }
        ID3D11Device_CreateBlendState(RLGL.device, &bd, &RLGL.currentBlendState);
        float blendFactor[4] = { 0, 0, 0, 0 };
        ID3D11DeviceContext_OMSetBlendState(RLGL.context, RLGL.currentBlendState, blendFactor, 0xFFFFFFFF);
    }

    // Depth stencil state
    {
        D3D11_DEPTH_STENCIL_DESC dsd = { 0 };
        dsd.DepthEnable = RLGL.State.depthTestEnabled;
        dsd.DepthWriteMask = RLGL.State.depthWriteEnabled ? D3D11_DEPTH_WRITE_MASK_ALL : D3D11_DEPTH_WRITE_MASK_ZERO;
        dsd.DepthFunc = D3D11_COMPARISON_LESS_EQUAL;
        dsd.StencilEnable = FALSE;
        ID3D11Device_CreateDepthStencilState(RLGL.device, &dsd, &RLGL.currentDSState);
        ID3D11DeviceContext_OMSetDepthStencilState(RLGL.context, RLGL.currentDSState, 0);
    }

    // Rasterizer state
    {
        D3D11_RASTERIZER_DESC rd = { 0 };
        rd.FillMode = RLGL.State.wireMode ? D3D11_FILL_WIREFRAME : D3D11_FILL_SOLID;
        rd.CullMode = RLGL.State.backfaceCullingEnabled ?
            (RLGL.State.cullFaceMode == 1 ? D3D11_CULL_FRONT : D3D11_CULL_BACK) : D3D11_CULL_NONE;
        rd.FrontCounterClockwise = TRUE;
        rd.DepthClipEnable = TRUE;
        rd.ScissorEnable = RLGL.State.scissorTestEnabled;
        rd.MultisampleEnable = FALSE;
        rd.AntialiasedLineEnable = FALSE;
        ID3D11Device_CreateRasterizerState(RLGL.device, &rd, &RLGL.currentRSState);
        ID3D11DeviceContext_RSSetState(RLGL.context, RLGL.currentRSState);
    }

    RLGL.stateDirty = false;
}

//----------------------------------------------------------------------------------
// Helper: Get DXGI format from rlgl pixel format
//----------------------------------------------------------------------------------
static DXGI_FORMAT rlGetDXGIFormat(int format)
{
    switch (format) {
        case 1:  return DXGI_FORMAT_R8_UNORM;            // GRAYSCALE
        case 2:  return DXGI_FORMAT_R8G8_UNORM;          // GRAY_ALPHA
        case 3:  return DXGI_FORMAT_B5G6R5_UNORM;        // R5G6B5
        case 4:  return DXGI_FORMAT_R8G8B8A8_UNORM;      // R8G8B8 (no 3-channel, use RGBA)
        case 5:  return DXGI_FORMAT_B5G5R5A1_UNORM;      // R5G5B5A1
        case 6:  return DXGI_FORMAT_B4G4R4A4_UNORM;      // R4G4B4A4
        case 7:  return DXGI_FORMAT_R8G8B8A8_UNORM;      // R8G8B8A8
        case 8:  return DXGI_FORMAT_R32_FLOAT;            // R32
        case 9:  return DXGI_FORMAT_R32G32B32_FLOAT;      // R32G32B32
        case 10: return DXGI_FORMAT_R32G32B32A32_FLOAT;   // R32G32B32A32
        case 11: return DXGI_FORMAT_R16_FLOAT;            // R16
        case 12: return DXGI_FORMAT_R16G16B16A16_FLOAT;   // R16G16B16 (no 3-ch, use 4)
        case 13: return DXGI_FORMAT_R16G16B16A16_FLOAT;   // R16G16B16A16
        case 14: return DXGI_FORMAT_BC1_UNORM;            // DXT1_RGB
        case 15: return DXGI_FORMAT_BC1_UNORM;            // DXT1_RGBA
        case 16: return DXGI_FORMAT_BC2_UNORM;            // DXT3_RGBA
        case 17: return DXGI_FORMAT_BC3_UNORM;            // DXT5_RGBA
        default: return DXGI_FORMAT_R8G8B8A8_UNORM;
    }
}

//----------------------------------------------------------------------------------
// Matrix operations (CPU-side, identical logic to OpenGL backend)
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
        float inverseLength = 1.0f/sqrtf(lengthSquared);
        x *= inverseLength; y *= inverseLength; z *= inverseLength;
    }
    float sinres = sinf(DEG2RAD*angle);
    float cosres = cosf(DEG2RAD*angle);
    float t = 1.0f - cosres;
    matRotation.m0 = x*x*t+cosres;   matRotation.m1 = y*x*t+z*sinres; matRotation.m2 = z*x*t-y*sinres;
    matRotation.m4 = x*y*t-z*sinres; matRotation.m5 = y*y*t+cosres;   matRotation.m6 = z*y*t+x*sinres;
    matRotation.m8 = x*z*t+y*sinres; matRotation.m9 = y*z*t-x*sinres; matRotation.m10= z*z*t+cosres;
    *RLGL.State.currentMatrix = rlMatrixMultiply(matRotation, *RLGL.State.currentMatrix);
}

void rlScalef(float x, float y, float z)
{
    Matrix matScale = { x,0,0,0, 0,y,0,0, 0,0,z,0, 0,0,0,1 };
    *RLGL.State.currentMatrix = rlMatrixMultiply(matScale, *RLGL.State.currentMatrix);
}

void rlMultMatrixf(const float *matf)
{
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
    matFrustum.m8 = ((float)right+(float)left)/rl;
    matFrustum.m9 = ((float)top+(float)bottom)/tb;
    matFrustum.m10 = -((float)zfar+(float)znear)/fn;
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
    matOrtho.m12 = -((float)left+(float)right)/rl;
    matOrtho.m13 = -((float)top+(float)bottom)/tb;
    matOrtho.m14 = -((float)zfar+(float)znear)/fn;
    matOrtho.m15 = 1.0f;
    *RLGL.State.currentMatrix = rlMatrixMultiply(*RLGL.State.currentMatrix, matOrtho);
}

void rlViewport(int x, int y, int width, int height)
{
    if (!RLGL.context) return;
    D3D11_VIEWPORT vp = { (FLOAT)x, (FLOAT)y, (FLOAT)width, (FLOAT)height, 0.0f, 1.0f };
    ID3D11DeviceContext_RSSetViewports(RLGL.context, 1, &vp);
}

void rlSetClipPlanes(double nearPlane, double farPlane)
{
    rlCullDistanceNear = nearPlane;
    rlCullDistanceFar = farPlane;
}

double rlGetCullDistanceNear(void) { return rlCullDistanceNear; }
double rlGetCullDistanceFar(void) { return rlCullDistanceFar; }

//----------------------------------------------------------------------------------
// Vertex-level operations (CPU-side batch accumulation)
//----------------------------------------------------------------------------------
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

//----------------------------------------------------------------------------------
// Texture state
//----------------------------------------------------------------------------------
void rlSetTexture(unsigned int id)
{
    if (id == 0) {
        if (RLGL.State.vertexCounter >= RLGL.currentBatch->vertexBuffer[RLGL.currentBatch->currentBuffer].elementCount*4)
            rlDrawRenderBatch(RLGL.currentBatch);
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

void rlActiveTextureSlot(int slot) { d3d11_activeTextureSlot = slot; }
void rlEnableTexture(unsigned int id) {
    if (id > 0 && id <= RLGL.textureCount && RLGL.textures[id-1].srv && RLGL.context) {
        // Determine the actual D3D11 register slot using per-shader mapping
        // In OpenGL, the material map index = texture unit, and glUniform1i maps sampler → unit.
        // In D3D11, we map material map index → HLSL register(tN) using texMaterialMapToRegister.
        int targetSlot = d3d11_activeTextureSlot;  // Default: identity mapping
        if (d3d11_activeShaderId > 0 && d3d11_activeShaderId <= RLGL.shaderCount) {
            unsigned int shaderIdx = d3d11_activeShaderId - 1;
            if (d3d11_activeTextureSlot >= 0 && d3d11_activeTextureSlot < RL_MAX_MATERIAL_MAPS) {
                targetSlot = RLGL.shaders[shaderIdx].texMaterialMapToRegister[d3d11_activeTextureSlot];
            }
        }
        ID3D11DeviceContext_PSSetShaderResources(RLGL.context, targetSlot, 1, &RLGL.textures[id-1].srv);
        // Also set sampler for this slot
        ID3D11SamplerState *sampler = RLGL.textures[id-1].sampler ? RLGL.textures[id-1].sampler : RLGL.defaultSampler;
        ID3D11DeviceContext_PSSetSamplers(RLGL.context, targetSlot, 1, &sampler);
    }
}
void rlDisableTexture(void) {
    int targetSlot = d3d11_activeTextureSlot;
    if (d3d11_activeShaderId > 0 && d3d11_activeShaderId <= RLGL.shaderCount) {
        unsigned int shaderIdx = d3d11_activeShaderId - 1;
        if (d3d11_activeTextureSlot >= 0 && d3d11_activeTextureSlot < RL_MAX_MATERIAL_MAPS) {
            targetSlot = RLGL.shaders[shaderIdx].texMaterialMapToRegister[d3d11_activeTextureSlot];
        }
    }
    ID3D11ShaderResourceView *nullSRV = NULL;
    if (RLGL.context) ID3D11DeviceContext_PSSetShaderResources(RLGL.context, targetSlot, 1, &nullSRV);
}
void rlEnableTextureCubemap(unsigned int id) { rlEnableTexture(id); }
void rlDisableTextureCubemap(void) { rlDisableTexture(); }
void rlTextureParameters(unsigned int id, int param, int value) { (void)id; (void)param; (void)value; }
void rlCubemapParameters(unsigned int id, int param, int value) { (void)id; (void)param; (void)value; }

//----------------------------------------------------------------------------------
// Shader state
//----------------------------------------------------------------------------------
void rlEnableShader(unsigned int id) {
    if (id > 0 && id <= RLGL.shaderCount && RLGL.context) {
        d3d11_activeShaderId = id;  // Track which shader is actually bound (for rlSetUniform/rlEnableTexture lookups)
        ID3D11DeviceContext_VSSetShader(RLGL.context, RLGL.shaders[id-1].vertexShader, NULL, 0);
        ID3D11DeviceContext_PSSetShader(RLGL.context, RLGL.shaders[id-1].pixelShader, NULL, 0);
        if (RLGL.shaders[id-1].inputLayout)
            ID3D11DeviceContext_IASetInputLayout(RLGL.context, RLGL.shaders[id-1].inputLayout);

        // Re-bind all constant buffers for this shader so that previously set uniform values
        // (e.g. from SetShaderValue at init time) remain active even if another shader/batch
        // rendering path has overwritten the cbuffer bindings in the meantime.
        rlD3D11Shader *s = &RLGL.shaders[id-1];
        for (int i = 0; i < s->cbufferCount; i++) {
            if (s->cbuffers[i].buffer) {
                if (s->cbuffers[i].stage == 0)
                    ID3D11DeviceContext_VSSetConstantBuffers(RLGL.context, s->cbuffers[i].registerSlot, 1, &s->cbuffers[i].buffer);
                else
                    ID3D11DeviceContext_PSSetConstantBuffers(RLGL.context, s->cbuffers[i].registerSlot, 1, &s->cbuffers[i].buffer);
            }
        }
    }
}
void rlDisableShader(void) {
    d3d11_activeShaderId = 0;
    if (RLGL.context) {
        ID3D11DeviceContext_VSSetShader(RLGL.context, NULL, NULL, 0);
        ID3D11DeviceContext_PSSetShader(RLGL.context, NULL, NULL, 0);
    }
}

//----------------------------------------------------------------------------------
// Framebuffer state
//----------------------------------------------------------------------------------
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
    (void)target; RLGL.activeFramebuffer = framebuffer;
}

//----------------------------------------------------------------------------------
// Render state
//----------------------------------------------------------------------------------
void rlEnableColorBlend(void) { RLGL.State.colorBlendEnabled = true; RLGL.stateDirty = true; }
void rlDisableColorBlend(void) { RLGL.State.colorBlendEnabled = false; RLGL.stateDirty = true; }
void rlEnableDepthTest(void) { RLGL.State.depthTestEnabled = true; RLGL.stateDirty = true; }
void rlDisableDepthTest(void) { RLGL.State.depthTestEnabled = false; RLGL.stateDirty = true; }
void rlEnableDepthMask(void) { RLGL.State.depthWriteEnabled = true; RLGL.stateDirty = true; }
void rlDisableDepthMask(void) { RLGL.State.depthWriteEnabled = false; RLGL.stateDirty = true; }
void rlEnableBackfaceCulling(void) { RLGL.State.backfaceCullingEnabled = true; RLGL.stateDirty = true; }
void rlDisableBackfaceCulling(void) { RLGL.State.backfaceCullingEnabled = false; RLGL.stateDirty = true; }
void rlColorMask(bool r, bool g, bool b, bool a) { (void)r; (void)g; (void)b; (void)a; }
void rlSetCullFace(int mode) { RLGL.State.cullFaceMode = (mode == RL_CULL_FACE_FRONT) ? 1 : 0; RLGL.stateDirty = true; }
void rlEnableScissorTest(void) { RLGL.State.scissorTestEnabled = true; RLGL.stateDirty = true; }
void rlDisableScissorTest(void) { RLGL.State.scissorTestEnabled = false; RLGL.stateDirty = true; }
void rlScissor(int x, int y, int width, int height) {
    if (RLGL.context) {
        D3D11_RECT rect = { x, y, x + width, y + height };
        ID3D11DeviceContext_RSSetScissorRects(RLGL.context, 1, &rect);
    }
}
void rlEnableWireMode(void) { RLGL.State.wireMode = true; RLGL.stateDirty = true; }
void rlEnablePointMode(void) { RLGL.State.wireMode = true; RLGL.stateDirty = true; }
void rlDisableWireMode(void) { RLGL.State.wireMode = false; RLGL.stateDirty = true; }
void rlSetLineWidth(float width) { RLGL.State.lineWidth = width; }
float rlGetLineWidth(void) { return RLGL.State.lineWidth; }
void rlEnableSmoothLines(void) { /* Not directly supported in D3D11 */ }
void rlDisableSmoothLines(void) { }
void rlEnableStereoRender(void) { RLGL.State.stereoRender = true; }
void rlDisableStereoRender(void) { RLGL.State.stereoRender = false; }
bool rlIsStereoRenderEnabled(void) { return RLGL.State.stereoRender; }

void rlClearColor(unsigned char r, unsigned char g, unsigned char b, unsigned char a) {
    RLGL.State.clearColor[0] = (float)r/255.0f;
    RLGL.State.clearColor[1] = (float)g/255.0f;
    RLGL.State.clearColor[2] = (float)b/255.0f;
    RLGL.State.clearColor[3] = (float)a/255.0f;
}

void rlClearScreenBuffers(void) {
    if (!RLGL.context) return;

    // Clear the active render target (backbuffer or FBO)
    ID3D11RenderTargetView *rtv = RLGL.backbufferRTV;
    ID3D11DepthStencilView *dsv = RLGL.depthStencilView;

    // If an FBO is active, use its RTV/DSV instead
    if (RLGL.activeFramebuffer > 0 && RLGL.activeFramebuffer <= RLGL.framebufferCount) {
        rlD3D11Framebuffer *fb = &RLGL.framebuffers[RLGL.activeFramebuffer - 1];
        if (fb->rtv[0]) rtv = fb->rtv[0];
        if (fb->dsv) dsv = fb->dsv;
    }

    if (rtv) ID3D11DeviceContext_ClearRenderTargetView(RLGL.context, rtv, RLGL.State.clearColor);
    if (dsv) ID3D11DeviceContext_ClearDepthStencilView(RLGL.context, dsv, D3D11_CLEAR_DEPTH | D3D11_CLEAR_STENCIL, 1.0f, 0);
}

void rlSwapScreenBuffer(void) {
    if (RLGL.swapChain) IDXGISwapChain_Present(RLGL.swapChain, 0, 0);
}

void rlCheckErrors(void) { /* D3D11 debug layer handles errors via OutputDebugString */ }

void rlSetBlendMode(int mode) {
    if (RLGL.State.currentBlendMode != mode) {
        rlDrawRenderBatch(RLGL.currentBatch);
        RLGL.State.currentBlendMode = mode;
        RLGL.stateDirty = true;
    }
}

void rlSetBlendFactors(int glSrcFactor, int glDstFactor, int glEquation) {
    (void)glSrcFactor; (void)glDstFactor; (void)glEquation;
}

void rlSetBlendFactorsSeparate(int glSrcRGB, int glDstRGB, int glSrcAlpha, int glDstAlpha, int glEqRGB, int glEqAlpha) {
    (void)glSrcRGB; (void)glDstRGB; (void)glSrcAlpha; (void)glDstAlpha; (void)glEqRGB; (void)glEqAlpha;
}

//----------------------------------------------------------------------------------
// rlgl initialization
//----------------------------------------------------------------------------------
void rlglInit(int width, int height, bool headless)
{
    (void)headless;

    // Create D3D11 device and context
    D3D_FEATURE_LEVEL featureLevels[] = { D3D_FEATURE_LEVEL_11_0, D3D_FEATURE_LEVEL_10_1, D3D_FEATURE_LEVEL_10_0 };
    UINT createFlags = 0;
#if defined(_DEBUG)
    createFlags |= D3D11_CREATE_DEVICE_DEBUG;
#endif

    HRESULT hr = D3D11CreateDevice(NULL, D3D_DRIVER_TYPE_HARDWARE, NULL, createFlags,
        featureLevels, 3, D3D11_SDK_VERSION,
        &RLGL.device, &RLGL.featureLevel, &RLGL.context);

    if (FAILED(hr)) {
        // Try WARP (software) driver
        hr = D3D11CreateDevice(NULL, D3D_DRIVER_TYPE_WARP, NULL, createFlags,
            featureLevels, 3, D3D11_SDK_VERSION,
            &RLGL.device, &RLGL.featureLevel, &RLGL.context);
    }

    if (FAILED(hr)) {
        TRACELOG(RL_LOG_ERROR, "D3D11: Failed to create device");
        return;
    }

    TRACELOG(RL_LOG_INFO, "D3D11: Device created successfully (Feature Level: %X)", RLGL.featureLevel);

    // Create swap chain using the native window handle
    if (!headless)
    {
        HWND hwnd = (HWND)GetWindowHandle();
        if (hwnd)
        {
            // Get DXGI factory from the device
            IDXGIDevice *dxgiDevice = NULL;
            IDXGIAdapter *dxgiAdapter = NULL;
            IDXGIFactory *dxgiFactory = NULL;

            hr = ID3D11Device_QueryInterface(RLGL.device, &IID_IDXGIDevice, (void **)&dxgiDevice);
            if (SUCCEEDED(hr)) hr = IDXGIDevice_GetAdapter(dxgiDevice, &dxgiAdapter);
            if (SUCCEEDED(hr)) hr = IDXGIAdapter_GetParent(dxgiAdapter, &IID_IDXGIFactory, (void **)&dxgiFactory);

            if (SUCCEEDED(hr))
            {
                DXGI_SWAP_CHAIN_DESC scd = { 0 };
                scd.BufferCount = 2;
                scd.BufferDesc.Width = width;
                scd.BufferDesc.Height = height;
                scd.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
                scd.BufferDesc.RefreshRate.Numerator = 0;
                scd.BufferDesc.RefreshRate.Denominator = 1;
                scd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
                scd.OutputWindow = hwnd;
                scd.SampleDesc.Count = 1;
                scd.SampleDesc.Quality = 0;
                scd.Windowed = TRUE;
                scd.SwapEffect = DXGI_SWAP_EFFECT_DISCARD;

                hr = IDXGIFactory_CreateSwapChain(dxgiFactory, (IUnknown *)RLGL.device, &scd, &RLGL.swapChain);

                if (SUCCEEDED(hr))
                {
                    TRACELOG(RL_LOG_INFO, "D3D11: Swap chain created successfully (%ix%i)", width, height);

                    // Create render target view from back buffer
                    ID3D11Texture2D *backBuffer = NULL;
                    hr = IDXGISwapChain_GetBuffer(RLGL.swapChain, 0, &IID_ID3D11Texture2D, (void **)&backBuffer);
                    if (SUCCEEDED(hr))
                    {
                        hr = ID3D11Device_CreateRenderTargetView(RLGL.device, (ID3D11Resource *)backBuffer, NULL, &RLGL.backbufferRTV);
                        ID3D11Texture2D_Release(backBuffer);
                    }

                    // Create depth-stencil buffer
                    D3D11_TEXTURE2D_DESC dtd = { 0 };
                    dtd.Width = width;
                    dtd.Height = height;
                    dtd.MipLevels = 1;
                    dtd.ArraySize = 1;
                    dtd.Format = DXGI_FORMAT_D24_UNORM_S8_UINT;
                    dtd.SampleDesc.Count = 1;
                    dtd.Usage = D3D11_USAGE_DEFAULT;
                    dtd.BindFlags = D3D11_BIND_DEPTH_STENCIL;

                    hr = ID3D11Device_CreateTexture2D(RLGL.device, &dtd, NULL, &RLGL.depthStencilTexture);
                    if (SUCCEEDED(hr))
                    {
                        hr = ID3D11Device_CreateDepthStencilView(RLGL.device, (ID3D11Resource *)RLGL.depthStencilTexture, NULL, &RLGL.depthStencilView);
                    }

                    // Bind render targets
                    if (RLGL.backbufferRTV && RLGL.depthStencilView)
                    {
                        ID3D11DeviceContext_OMSetRenderTargets(RLGL.context, 1, &RLGL.backbufferRTV, RLGL.depthStencilView);
                        TRACELOG(RL_LOG_INFO, "D3D11: Render target and depth-stencil created successfully");
                    }
                    else
                    {
                        TRACELOG(RL_LOG_ERROR, "D3D11: Failed to create render target or depth-stencil view");
                    }

                    // Set viewport
                    D3D11_VIEWPORT vp = { 0 };
                    vp.Width = (float)width;
                    vp.Height = (float)height;
                    vp.MinDepth = 0.0f;
                    vp.MaxDepth = 1.0f;
                    ID3D11DeviceContext_RSSetViewports(RLGL.context, 1, &vp);
                }
                else
                {
                    TRACELOG(RL_LOG_ERROR, "D3D11: Failed to create swap chain (HRESULT: 0x%08X)", hr);
                }
            }
            else
            {
                TRACELOG(RL_LOG_ERROR, "D3D11: Failed to get DXGI factory from device");
            }

            if (dxgiFactory) IDXGIFactory_Release(dxgiFactory);
            if (dxgiAdapter) IDXGIAdapter_Release(dxgiAdapter);
            if (dxgiDevice) IDXGIDevice_Release(dxgiDevice);
        }
        else
        {
            TRACELOG(RL_LOG_WARNING, "D3D11: No window handle available, swap chain not created");
        }
    }

    // Create constant buffers
    RLGL.cbMatrix = rlCreateD3D11Buffer(NULL, sizeof(rlD3D11MatrixCB), D3D11_BIND_CONSTANT_BUFFER, true);
    RLGL.cbColor = rlCreateD3D11Buffer(NULL, sizeof(rlD3D11ColorCB), D3D11_BIND_CONSTANT_BUFFER, true);

    // Create default sampler
    {
        D3D11_SAMPLER_DESC sd = { 0 };
        sd.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
        sd.AddressU = D3D11_TEXTURE_ADDRESS_WRAP;
        sd.AddressV = D3D11_TEXTURE_ADDRESS_WRAP;
        sd.AddressW = D3D11_TEXTURE_ADDRESS_WRAP;
        sd.ComparisonFunc = D3D11_COMPARISON_NEVER;
        sd.MaxLOD = D3D11_FLOAT32_MAX;
        ID3D11Device_CreateSamplerState(RLGL.device, &sd, &RLGL.defaultSampler);
        ID3D11DeviceContext_PSSetSamplers(RLGL.context, 0, 1, &RLGL.defaultSampler);
    }

    // Init default white texture (1x1 RGBA)
    {
        unsigned char pixels[4] = { 255, 255, 255, 255 };
        RLGL.State.defaultTextureId = rlLoadTexture(pixels, 1, 1, 7, 1); // 7 = R8G8B8A8
        if (RLGL.State.defaultTextureId != 0)
            TRACELOG(RL_LOG_INFO, "TEXTURE: [ID %i] Default texture loaded successfully", RLGL.State.defaultTextureId);
    }

    // Create default white vertex buffer for meshes without vertex colors
    // This provides (255,255,255,255) = white for every vertex when bound as the COLOR attribute
    {
        unsigned char *whiteData = (unsigned char *)RL_MALLOC(RL_D3D11_DEFAULT_WHITE_VBO_VERTS * 4);
        if (whiteData) {
            memset(whiteData, 0xFF, RL_D3D11_DEFAULT_WHITE_VBO_VERTS * 4);
            d3d11_defaultWhiteVBO = rlCreateD3D11Buffer(whiteData, RL_D3D11_DEFAULT_WHITE_VBO_VERTS * 4, D3D11_BIND_VERTEX_BUFFER, false);
            RL_FREE(whiteData);
            if (d3d11_defaultWhiteVBO) TRACELOG(RL_LOG_INFO, "BUFFER: Default white vertex buffer created (%d vertices)", RL_D3D11_DEFAULT_WHITE_VBO_VERTS);
        }
    }

    // Init default shader
    rlLoadShaderDefault();
    RLGL.State.currentShaderId = RLGL.State.defaultShaderId;
    RLGL.State.currentShaderLocs = RLGL.State.defaultShaderLocs;

    // Init default vertex arrays buffers
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
    RLGL.stateDirty = true;

    RLGL.State.framebufferWidth = width;
    RLGL.State.framebufferHeight = height;

    // Set initial states
    rlUpdateD3D11States();

    TRACELOG(RL_LOG_INFO, "RLGL: Direct3D 11 backend initialized successfully");
}

void rlglClose(void)
{
    rlUnloadRenderBatch(RLGL.defaultBatch);
    rlUnloadShaderDefault();

    if (RLGL.State.defaultTextureId > 0) rlUnloadTexture(RLGL.State.defaultTextureId);

    // Release default white vertex buffer
    if (d3d11_defaultWhiteVBO) { ID3D11Buffer_Release(d3d11_defaultWhiteVBO); d3d11_defaultWhiteVBO = NULL; }

    // Release all tracked resources
    for (unsigned int i = 0; i < RLGL.textureCount; i++) {
        if (RLGL.textures[i].srv) ID3D11ShaderResourceView_Release(RLGL.textures[i].srv);
        if (RLGL.textures[i].texture) ID3D11Texture2D_Release(RLGL.textures[i].texture);
        if (RLGL.textures[i].sampler) ID3D11SamplerState_Release(RLGL.textures[i].sampler);
        if (RLGL.textures[i].dsv) ID3D11DepthStencilView_Release(RLGL.textures[i].dsv);
        if (RLGL.textures[i].rtv) ID3D11RenderTargetView_Release(RLGL.textures[i].rtv);
    }
    for (unsigned int i = 0; i < RLGL.shaderCount; i++) {
        if (RLGL.shaders[i].vertexShader) ID3D11VertexShader_Release(RLGL.shaders[i].vertexShader);
        if (RLGL.shaders[i].pixelShader) ID3D11PixelShader_Release(RLGL.shaders[i].pixelShader);
        if (RLGL.shaders[i].inputLayout) ID3D11InputLayout_Release(RLGL.shaders[i].inputLayout);
        if (RLGL.shaders[i].vsBlob) ID3D10Blob_Release(RLGL.shaders[i].vsBlob);
        if (RLGL.shaders[i].psBlob) ID3D10Blob_Release(RLGL.shaders[i].psBlob);
    }
    for (unsigned int i = 0; i < RLGL.bufferCount; i++) {
        if (RLGL.buffers[i].buffer) ID3D11Buffer_Release(RLGL.buffers[i].buffer);
    }

    if (RLGL.currentBlendState) ID3D11BlendState_Release(RLGL.currentBlendState);
    if (RLGL.currentDSState) ID3D11DepthStencilState_Release(RLGL.currentDSState);
    if (RLGL.currentRSState) ID3D11RasterizerState_Release(RLGL.currentRSState);
    if (RLGL.cbMatrix) ID3D11Buffer_Release(RLGL.cbMatrix);
    if (RLGL.cbColor) ID3D11Buffer_Release(RLGL.cbColor);
    if (RLGL.defaultSampler) ID3D11SamplerState_Release(RLGL.defaultSampler);

    for (int i = 0; i < 4; i++) {
        if (RLGL.batchVertexBuffers[i]) ID3D11Buffer_Release(RLGL.batchVertexBuffers[i]);
    }
    if (RLGL.batchIndexBuffer) ID3D11Buffer_Release(RLGL.batchIndexBuffer);

    // Release swap chain resources
    if (RLGL.depthStencilView) { ID3D11DepthStencilView_Release(RLGL.depthStencilView); RLGL.depthStencilView = NULL; }
    if (RLGL.depthStencilTexture) { ID3D11Texture2D_Release(RLGL.depthStencilTexture); RLGL.depthStencilTexture = NULL; }
    if (RLGL.backbufferRTV) { ID3D11RenderTargetView_Release(RLGL.backbufferRTV); RLGL.backbufferRTV = NULL; }
    if (RLGL.swapChain) { IDXGISwapChain_Release(RLGL.swapChain); RLGL.swapChain = NULL; }

    if (RLGL.context) { ID3D11DeviceContext_Release(RLGL.context); RLGL.context = NULL; }
    if (RLGL.device) { ID3D11Device_Release(RLGL.device); RLGL.device = NULL; }

    TRACELOG(RL_LOG_INFO, "RLGL: Direct3D 11 backend closed successfully");
}

void rlLoadExtensions(void *loader)
{
    (void)loader;
    // D3D11 capabilities are determined at device creation time
    RLGL.ExtSupported.computeShader = (RLGL.featureLevel >= D3D_FEATURE_LEVEL_11_0);
    RLGL.ExtSupported.ssbo = (RLGL.featureLevel >= D3D_FEATURE_LEVEL_11_0);
    RLGL.ExtSupported.maxAnisotropyLevel = 16.0f;
    RLGL.ExtSupported.maxDepthBits = 32;

    TRACELOG(RL_LOG_INFO, "D3D11: Feature level: 0x%X", (unsigned int)RLGL.featureLevel);
}

int rlGetVersion(void) { return RL_DIRECT3D_11; }

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
    batch.vertexBuffer = (rlVertexBuffer *)RL_MALLOC(numBuffers*sizeof(rlVertexBuffer));

    for (int i = 0; i < numBuffers; i++) {
        batch.vertexBuffer[i].elementCount = bufferElements;
        batch.vertexBuffer[i].vertices = (float *)RL_MALLOC(bufferElements*3*4*sizeof(float));
        batch.vertexBuffer[i].texcoords = (float *)RL_MALLOC(bufferElements*2*4*sizeof(float));
        batch.vertexBuffer[i].normals = (float *)RL_MALLOC(bufferElements*3*4*sizeof(float));
        batch.vertexBuffer[i].colors = (unsigned char *)RL_MALLOC(bufferElements*4*4*sizeof(unsigned char));
        batch.vertexBuffer[i].indices = (unsigned int *)RL_MALLOC(bufferElements*6*sizeof(unsigned int));

        for (int j = 0; j < (3*4*bufferElements); j++) batch.vertexBuffer[i].vertices[j] = 0.0f;
        for (int j = 0; j < (2*4*bufferElements); j++) batch.vertexBuffer[i].texcoords[j] = 0.0f;
        for (int j = 0; j < (3*4*bufferElements); j++) batch.vertexBuffer[i].normals[j] = 0.0f;
        for (int j = 0; j < (4*4*bufferElements); j++) batch.vertexBuffer[i].colors[j] = 0;

        int k = 0;
        for (int j = 0; j < (6*bufferElements); j += 6) {
            batch.vertexBuffer[i].indices[j] = 4*k;
            batch.vertexBuffer[i].indices[j+1] = 4*k+1;
            batch.vertexBuffer[i].indices[j+2] = 4*k+2;
            batch.vertexBuffer[i].indices[j+3] = 4*k;
            batch.vertexBuffer[i].indices[j+4] = 4*k+2;
            batch.vertexBuffer[i].indices[j+5] = 4*k+3;
            k++;
        }
        RLGL.State.vertexCounter = 0;
    }

    // Create GPU buffers for batch rendering
    int totalVertices = bufferElements * 4;
    if (RLGL.device) {
        RLGL.batchVertexBuffers[0] = rlCreateD3D11Buffer(NULL, totalVertices*3*sizeof(float), D3D11_BIND_VERTEX_BUFFER, true);
        RLGL.batchVertexBuffers[1] = rlCreateD3D11Buffer(NULL, totalVertices*2*sizeof(float), D3D11_BIND_VERTEX_BUFFER, true);
        RLGL.batchVertexBuffers[2] = rlCreateD3D11Buffer(NULL, totalVertices*3*sizeof(float), D3D11_BIND_VERTEX_BUFFER, true);
        RLGL.batchVertexBuffers[3] = rlCreateD3D11Buffer(NULL, totalVertices*4*sizeof(unsigned char), D3D11_BIND_VERTEX_BUFFER, true);
        RLGL.batchIndexBuffer = rlCreateD3D11Buffer(batch.vertexBuffer[0].indices, bufferElements*6*sizeof(unsigned int), D3D11_BIND_INDEX_BUFFER, false);
        RLGL.batchBufferSize = totalVertices;
    }

    batch.draws = (rlDrawCall *)RL_MALLOC(RL_DEFAULT_BATCH_DRAWCALLS*sizeof(rlDrawCall));
    for (int i = 0; i < RL_DEFAULT_BATCH_DRAWCALLS; i++) {
        batch.draws[i].mode = RL_QUADS;
        batch.draws[i].vertexCount = 0;
        batch.draws[i].vertexAlignment = 0;
        batch.draws[i].textureId = RLGL.State.defaultTextureId;
    }
    batch.bufferCount = numBuffers;
    batch.drawCounter = 1;
    batch.currentDepth = -1.0f;

    TRACELOG(RL_LOG_INFO, "RLGL: D3D11 render batch loaded successfully");
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
    if (!RLGL.context || !RLGL.device) return;
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

    // Update D3D11 render states
    if (RLGL.stateDirty) rlUpdateD3D11States();

    // Upload vertex data to GPU
    D3D11_MAPPED_SUBRESOURCE mapped;

    // Position buffer
    if (SUCCEEDED(ID3D11DeviceContext_Map(RLGL.context, (ID3D11Resource *)RLGL.batchVertexBuffers[0], 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped))) {
        memcpy(mapped.pData, batch->vertexBuffer[batch->currentBuffer].vertices, RLGL.State.vertexCounter*3*sizeof(float));
        ID3D11DeviceContext_Unmap(RLGL.context, (ID3D11Resource *)RLGL.batchVertexBuffers[0], 0);
    }
    // Texcoord buffer
    if (SUCCEEDED(ID3D11DeviceContext_Map(RLGL.context, (ID3D11Resource *)RLGL.batchVertexBuffers[1], 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped))) {
        memcpy(mapped.pData, batch->vertexBuffer[batch->currentBuffer].texcoords, RLGL.State.vertexCounter*2*sizeof(float));
        ID3D11DeviceContext_Unmap(RLGL.context, (ID3D11Resource *)RLGL.batchVertexBuffers[1], 0);
    }
    // Normal buffer
    if (SUCCEEDED(ID3D11DeviceContext_Map(RLGL.context, (ID3D11Resource *)RLGL.batchVertexBuffers[2], 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped))) {
        memcpy(mapped.pData, batch->vertexBuffer[batch->currentBuffer].normals, RLGL.State.vertexCounter*3*sizeof(float));
        ID3D11DeviceContext_Unmap(RLGL.context, (ID3D11Resource *)RLGL.batchVertexBuffers[2], 0);
    }
    // Color buffer
    if (SUCCEEDED(ID3D11DeviceContext_Map(RLGL.context, (ID3D11Resource *)RLGL.batchVertexBuffers[3], 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped))) {
        memcpy(mapped.pData, batch->vertexBuffer[batch->currentBuffer].colors, RLGL.State.vertexCounter*4*sizeof(unsigned char));
        ID3D11DeviceContext_Unmap(RLGL.context, (ID3D11Resource *)RLGL.batchVertexBuffers[3], 0);
    }

    // Set vertex buffers
    UINT strides[4] = { 3*sizeof(float), 2*sizeof(float), 3*sizeof(float), 4*sizeof(unsigned char) };
    UINT offsets[4] = { 0, 0, 0, 0 };
    ID3D11DeviceContext_IASetVertexBuffers(RLGL.context, 0, 4, RLGL.batchVertexBuffers, strides, offsets);
    ID3D11DeviceContext_IASetIndexBuffer(RLGL.context, RLGL.batchIndexBuffer, DXGI_FORMAT_R32_UINT, 0);

    // Set shader
    rlEnableShader(RLGL.State.currentShaderId);

    // Set constant buffers
    ID3D11DeviceContext_VSSetConstantBuffers(RLGL.context, 0, 1, &RLGL.cbMatrix);
    ID3D11DeviceContext_PSSetConstantBuffers(RLGL.context, 1, 1, &RLGL.cbColor);

    // Update MVP matrix
    Matrix matMVP = rlMatrixMultiply(RLGL.State.modelview, RLGL.State.projection);
    rl_float16 matData = rlMatrixToFloatV(matMVP);
    if (SUCCEEDED(ID3D11DeviceContext_Map(RLGL.context, (ID3D11Resource *)RLGL.cbMatrix, 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped))) {
        memcpy(mapped.pData, matData.v, sizeof(float)*16);
        ID3D11DeviceContext_Unmap(RLGL.context, (ID3D11Resource *)RLGL.cbMatrix, 0);
    }

    // Update diffuse color (white default)
    float diffuse[4] = { 1.0f, 1.0f, 1.0f, 1.0f };
    if (SUCCEEDED(ID3D11DeviceContext_Map(RLGL.context, (ID3D11Resource *)RLGL.cbColor, 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped))) {
        memcpy(mapped.pData, diffuse, sizeof(float)*4);
        ID3D11DeviceContext_Unmap(RLGL.context, (ID3D11Resource *)RLGL.cbColor, 0);
    }

    // Set sampler
    ID3D11DeviceContext_PSSetSamplers(RLGL.context, 0, 1, &RLGL.defaultSampler);

    // Draw each batch draw call
    for (int i = 0, vertexOffset = 0; i < batch->drawCounter; i++) {
        // Bind texture
        unsigned int texId = batch->draws[i].textureId;
        if (texId > 0 && texId <= RLGL.textureCount && RLGL.textures[texId-1].srv) {
            ID3D11DeviceContext_PSSetShaderResources(RLGL.context, 0, 1, &RLGL.textures[texId-1].srv);
        }

        if ((batch->draws[i].mode == RL_LINES) || (batch->draws[i].mode == RL_TRIANGLES)) {
            D3D11_PRIMITIVE_TOPOLOGY topo = (batch->draws[i].mode == RL_LINES) ?
                D3D11_PRIMITIVE_TOPOLOGY_LINELIST : D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
            ID3D11DeviceContext_IASetPrimitiveTopology(RLGL.context, topo);
            ID3D11DeviceContext_Draw(RLGL.context, batch->draws[i].vertexCount, vertexOffset);
        } else {
            // QUADS: Draw as indexed triangles
            ID3D11DeviceContext_IASetPrimitiveTopology(RLGL.context, D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            ID3D11DeviceContext_DrawIndexed(RLGL.context, batch->draws[i].vertexCount/4*6, vertexOffset/4*6, 0);
        }
        vertexOffset += (batch->draws[i].vertexCount + batch->draws[i].vertexAlignment);
    }

    // Reset batch
    RLGL.State.vertexCounter = 0;
    batch->currentDepth = -1.0f;
    Matrix matProjection = RLGL.State.projection;
    Matrix matModelView = RLGL.State.modelview;
    RLGL.State.projection = matProjection;
    RLGL.State.modelview = matModelView;

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
    if ((RLGL.State.vertexCounter + vCount) >= (RLGL.currentBatch->vertexBuffer[RLGL.currentBatch->currentBuffer].elementCount*4)) {
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
    if (!RLGL.device || RLGL.textureCount >= RL_D3D11_MAX_TEXTURES) return 0;

    DXGI_FORMAT dxgiFormat = rlGetDXGIFormat(format);
    unsigned int idx = RLGL.textureCount;

    D3D11_TEXTURE2D_DESC desc = { 0 };
    desc.Width = width;
    desc.Height = height;
    desc.MipLevels = mipmapCount;
    desc.ArraySize = 1;
    desc.Format = dxgiFormat;
    desc.SampleDesc.Count = 1;
    desc.Usage = D3D11_USAGE_DEFAULT;
    desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;

    D3D11_SUBRESOURCE_DATA initData = { 0 };
    int bpp = 4; // Default bytes per pixel for RGBA8
    switch (format) {
        case 1: bpp = 1; break; // GRAYSCALE
        case 2: bpp = 2; break; // GRAY_ALPHA
        case 3: bpp = 2; break; // R5G6B5
        case 4: bpp = 4; break; // R8G8B8 -> stored as RGBA (expanded below)
        case 5: bpp = 2; break; // R5G5B5A1
        case 6: bpp = 2; break; // R4G4B4A4
        case 7: bpp = 4; break; // R8G8B8A8
        case 8: bpp = 4; break; // R32
        case 9: bpp = 12; break; // R32G32B32
        case 10: bpp = 16; break; // R32G32B32A32
        case 11: bpp = 2; break; // R16
        case 12: bpp = 8; break; // R16G16B16 -> stored as RGBA16 (expanded below)
        case 13: bpp = 8; break; // R16G16B16A16
        default: bpp = 4; break;
    }

    // D3D11 has no 3-channel texture formats, so we must expand 3-channel source
    // data (RGB) to 4-channel (RGBA) before upload.
    void *expandedData = NULL;
    if (data != NULL && format == 4) {
        // R8G8B8 (3 bytes/pixel) -> R8G8B8A8 (4 bytes/pixel)
        int pixelCount = width * height;
        expandedData = RL_MALLOC(pixelCount * 4);
        const unsigned char *src = (const unsigned char *)data;
        unsigned char *dst = (unsigned char *)expandedData;
        for (int i = 0; i < pixelCount; i++) {
            dst[i*4 + 0] = src[i*3 + 0];
            dst[i*4 + 1] = src[i*3 + 1];
            dst[i*4 + 2] = src[i*3 + 2];
            dst[i*4 + 3] = 255;
        }
        initData.pSysMem = expandedData;
    } else if (data != NULL && format == 12) {
        // R16G16B16 (6 bytes/pixel) -> R16G16B16A16 (8 bytes/pixel)
        int pixelCount = width * height;
        expandedData = RL_MALLOC(pixelCount * 8);
        const unsigned short *src = (const unsigned short *)data;
        unsigned short *dst = (unsigned short *)expandedData;
        for (int i = 0; i < pixelCount; i++) {
            dst[i*4 + 0] = src[i*3 + 0];
            dst[i*4 + 1] = src[i*3 + 1];
            dst[i*4 + 2] = src[i*3 + 2];
            dst[i*4 + 3] = 0x3C00; // 1.0 in float16
        }
        initData.pSysMem = expandedData;
    } else {
        initData.pSysMem = data;
    }
    initData.SysMemPitch = width * bpp;

    HRESULT hr = ID3D11Device_CreateTexture2D(RLGL.device, &desc, data ? &initData : NULL, &RLGL.textures[idx].texture);
    RL_FREE(expandedData); // Safe to call with NULL
    if (FAILED(hr)) return 0;

    // Create SRV
    D3D11_SHADER_RESOURCE_VIEW_DESC srvDesc = { 0 };
    srvDesc.Format = dxgiFormat;
    srvDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
    srvDesc.Texture2D.MipLevels = mipmapCount;
    hr = ID3D11Device_CreateShaderResourceView(RLGL.device, (ID3D11Resource *)RLGL.textures[idx].texture, &srvDesc, &RLGL.textures[idx].srv);
    if (FAILED(hr)) {
        ID3D11Texture2D_Release(RLGL.textures[idx].texture);
        RLGL.textures[idx].texture = NULL;
        return 0;
    }

    RLGL.textures[idx].width = width;
    RLGL.textures[idx].height = height;
    RLGL.textures[idx].format = format;
    RLGL.textureCount++;

    unsigned int id = idx + 1;
    TRACELOG(RL_LOG_INFO, "TEXTURE: [ID %i] Texture loaded successfully (%ix%i)", id, width, height);
    return id;
}

unsigned int rlLoadTextureDepth(int width, int height, bool useRenderBuffer)
{
    (void)useRenderBuffer;
    if (!RLGL.device || RLGL.textureCount >= RL_D3D11_MAX_TEXTURES) return 0;

    unsigned int idx = RLGL.textureCount;

    D3D11_TEXTURE2D_DESC desc = { 0 };
    desc.Width = width;
    desc.Height = height;
    desc.MipLevels = 1;
    desc.ArraySize = 1;
    desc.Format = DXGI_FORMAT_R32_TYPELESS;
    desc.SampleDesc.Count = 1;
    desc.Usage = D3D11_USAGE_DEFAULT;
    desc.BindFlags = D3D11_BIND_DEPTH_STENCIL | D3D11_BIND_SHADER_RESOURCE;

    HRESULT hr = ID3D11Device_CreateTexture2D(RLGL.device, &desc, NULL, &RLGL.textures[idx].texture);
    if (FAILED(hr)) return 0;

    // DSV
    D3D11_DEPTH_STENCIL_VIEW_DESC dsvDesc = { 0 };
    dsvDesc.Format = DXGI_FORMAT_D32_FLOAT;
    dsvDesc.ViewDimension = D3D11_DSV_DIMENSION_TEXTURE2D;
    hr = ID3D11Device_CreateDepthStencilView(RLGL.device, (ID3D11Resource *)RLGL.textures[idx].texture, &dsvDesc, &RLGL.textures[idx].dsv);

    // SRV
    D3D11_SHADER_RESOURCE_VIEW_DESC srvDesc = { 0 };
    srvDesc.Format = DXGI_FORMAT_R32_FLOAT;
    srvDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
    srvDesc.Texture2D.MipLevels = 1;
    ID3D11Device_CreateShaderResourceView(RLGL.device, (ID3D11Resource *)RLGL.textures[idx].texture, &srvDesc, &RLGL.textures[idx].srv);

    RLGL.textures[idx].width = width;
    RLGL.textures[idx].height = height;
    RLGL.textures[idx].isDepth = true;
    RLGL.textureCount++;

    return idx + 1;
}

unsigned int rlLoadTextureCubemap(const void *data, int size, int format, int mipmapCount)
{
    if (!RLGL.device || RLGL.textureCount >= RL_D3D11_MAX_TEXTURES) return 0;
    if (mipmapCount < 1) mipmapCount = 1;
    if (mipmapCount > 16) mipmapCount = 16;

    unsigned int idx = RLGL.textureCount;
    DXGI_FORMAT dxgiFormat = rlGetDXGIFormat(format);

    // Source bytes per pixel (how data arrives from raylib)
    int srcBpp = 4;
    switch (format) {
        case 1: srcBpp = 1; break;   // GRAYSCALE
        case 2: srcBpp = 2; break;   // GRAY_ALPHA
        case 3: srcBpp = 2; break;   // R5G6B5
        case 4: srcBpp = 3; break;   // R8G8B8 (3 bytes, will expand to 4)
        case 5: srcBpp = 2; break;   // R5G5B5A1
        case 6: srcBpp = 2; break;   // R4G4B4A4
        case 7: srcBpp = 4; break;   // R8G8B8A8
        case 8: srcBpp = 4; break;   // R32
        case 9: srcBpp = 12; break;  // R32G32B32
        case 10: srcBpp = 16; break; // R32G32B32A32
        case 11: srcBpp = 2; break;  // R16
        case 12: srcBpp = 6; break;  // R16G16B16 (6 bytes, will expand to 8)
        case 13: srcBpp = 8; break;  // R16G16B16A16
        default: srcBpp = 4; break;
    }
    // D3D11 dest bytes per pixel (after any expansion)
    int dstBpp = srcBpp;
    if (format == 4) dstBpp = 4;   // R8G8B8 -> R8G8B8A8
    if (format == 12) dstBpp = 8;  // R16G16B16 -> R16G16B16A16

    D3D11_TEXTURE2D_DESC desc = { 0 };
    desc.Width = size;
    desc.Height = size;
    desc.MipLevels = mipmapCount;
    desc.ArraySize = 6;
    desc.Format = dxgiFormat;
    desc.SampleDesc.Count = 1;
    desc.Usage = D3D11_USAGE_DEFAULT;
    desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    desc.MiscFlags = D3D11_RESOURCE_MISC_TEXTURECUBE;

    // Prepare per-face subresource data
    // D3D11 subresource index for texture arrays: subresource = mipSlice + arraySlice * mipLevels
    int maxSubs = 6 * mipmapCount;
    D3D11_SUBRESOURCE_DATA initData[6 * 16];
    void *expandedBuffers[6 * 16]; // Track expanded allocations for cleanup
    for (int i = 0; i < maxSubs; i++) expandedBuffers[i] = NULL;

    bool hasData = (data != NULL);
    if (hasData) {
        const unsigned char *srcPtr = (const unsigned char *)data;
        int mipSize = size;
        for (int mip = 0; mip < mipmapCount; mip++) {
            int srcFaceBytes = mipSize * mipSize * srcBpp;

            for (int face = 0; face < 6; face++) {
                int subIdx = face * mipmapCount + mip;

                if (format == 4) {
                    // R8G8B8 -> R8G8B8A8 expansion
                    int pixelCount = mipSize * mipSize;
                    unsigned char *expanded = (unsigned char *)RL_MALLOC(pixelCount * 4);
                    for (int p = 0; p < pixelCount; p++) {
                        expanded[p*4 + 0] = srcPtr[p*3 + 0];
                        expanded[p*4 + 1] = srcPtr[p*3 + 1];
                        expanded[p*4 + 2] = srcPtr[p*3 + 2];
                        expanded[p*4 + 3] = 255;
                    }
                    initData[subIdx].pSysMem = expanded;
                    initData[subIdx].SysMemPitch = mipSize * 4;
                    initData[subIdx].SysMemSlicePitch = 0;
                    expandedBuffers[subIdx] = expanded;
                } else if (format == 12) {
                    // R16G16B16 -> R16G16B16A16 expansion
                    int pixelCount = mipSize * mipSize;
                    unsigned char *expanded = (unsigned char *)RL_MALLOC(pixelCount * 8);
                    const unsigned short *src16 = (const unsigned short *)srcPtr;
                    unsigned short *dst16 = (unsigned short *)expanded;
                    for (int p = 0; p < pixelCount; p++) {
                        dst16[p*4 + 0] = src16[p*3 + 0];
                        dst16[p*4 + 1] = src16[p*3 + 1];
                        dst16[p*4 + 2] = src16[p*3 + 2];
                        dst16[p*4 + 3] = 0x3C00; // 1.0 in half float
                    }
                    initData[subIdx].pSysMem = expanded;
                    initData[subIdx].SysMemPitch = mipSize * 8;
                    initData[subIdx].SysMemSlicePitch = 0;
                    expandedBuffers[subIdx] = expanded;
                } else {
                    initData[subIdx].pSysMem = srcPtr;
                    initData[subIdx].SysMemPitch = (unsigned int)(mipSize * dstBpp);
                    initData[subIdx].SysMemSlicePitch = 0;
                }
                srcPtr += srcFaceBytes;
            }
            mipSize /= 2;
            if (mipSize < 1) mipSize = 1;
        }
    }

    HRESULT hr = ID3D11Device_CreateTexture2D(RLGL.device, &desc, hasData ? initData : NULL, &RLGL.textures[idx].texture);

    // Free all expanded buffers (safe to call after CreateTexture2D has copied the data)
    for (int i = 0; i < maxSubs; i++) { if (expandedBuffers[i]) RL_FREE(expandedBuffers[i]); }

    if (FAILED(hr)) {
        TRACELOG(RL_LOG_WARNING, "TEXTURE: Failed to create cubemap texture (HRESULT: 0x%08X)", hr);
        return 0;
    }

    D3D11_SHADER_RESOURCE_VIEW_DESC srvDesc = { 0 };
    srvDesc.Format = dxgiFormat;
    srvDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURECUBE;
    srvDesc.TextureCube.MipLevels = mipmapCount;
    ID3D11Device_CreateShaderResourceView(RLGL.device, (ID3D11Resource *)RLGL.textures[idx].texture, &srvDesc, &RLGL.textures[idx].srv);

    RLGL.textures[idx].width = size;
    RLGL.textures[idx].height = size;
    RLGL.textures[idx].isCubemap = true;
    RLGL.textureCount++;

    TRACELOG(RL_LOG_INFO, "TEXTURE: [ID %i] Cubemap texture loaded successfully (%ix%i, %d faces)", idx + 1, size, size, 6);
    return idx + 1;
}

void rlUpdateTexture(unsigned int id, int offsetX, int offsetY, int width, int height, int format, const void *data)
{
    if (id == 0 || id > RLGL.textureCount || !RLGL.context || !data) return;
    unsigned int idx = id - 1;
    D3D11_BOX box = { (UINT)offsetX, (UINT)offsetY, 0, (UINT)(offsetX+width), (UINT)(offsetY+height), 1 };
    int bpp = 4;
    switch (format) {
        case 1: bpp = 1; break;   // GRAYSCALE
        case 2: bpp = 2; break;   // GRAY_ALPHA
        case 3: bpp = 2; break;   // R5G6B5
        case 4: bpp = 4; break;   // R8G8B8 -> RGBA (expanded below)
        case 5: bpp = 2; break;   // R5G5B5A1
        case 6: bpp = 2; break;   // R4G4B4A4
        case 7: bpp = 4; break;   // R8G8B8A8
        case 8: bpp = 4; break;   // R32
        case 9: bpp = 12; break;  // R32G32B32
        case 10: bpp = 16; break; // R32G32B32A32
        case 11: bpp = 2; break;  // R16
        case 12: bpp = 8; break;  // R16G16B16 -> RGBA16 (expanded below)
        case 13: bpp = 8; break;  // R16G16B16A16
        default: bpp = 4; break;
    }

    void *expandedData = NULL;
    const void *uploadData = data;
    if (format == 4) {
        int pixelCount = width * height;
        expandedData = RL_MALLOC(pixelCount * 4);
        const unsigned char *src = (const unsigned char *)data;
        unsigned char *dst = (unsigned char *)expandedData;
        for (int i = 0; i < pixelCount; i++) {
            dst[i*4 + 0] = src[i*3 + 0];
            dst[i*4 + 1] = src[i*3 + 1];
            dst[i*4 + 2] = src[i*3 + 2];
            dst[i*4 + 3] = 255;
        }
        uploadData = expandedData;
    } else if (format == 12) {
        int pixelCount = width * height;
        expandedData = RL_MALLOC(pixelCount * 8);
        const unsigned short *src = (const unsigned short *)data;
        unsigned short *dst = (unsigned short *)expandedData;
        for (int i = 0; i < pixelCount; i++) {
            dst[i*4 + 0] = src[i*3 + 0];
            dst[i*4 + 1] = src[i*3 + 1];
            dst[i*4 + 2] = src[i*3 + 2];
            dst[i*4 + 3] = 0x3C00;
        }
        uploadData = expandedData;
    }

    ID3D11DeviceContext_UpdateSubresource(RLGL.context, (ID3D11Resource *)RLGL.textures[idx].texture, 0, &box, uploadData, width*bpp, 0);
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
    if (RLGL.textures[idx].srv) { ID3D11ShaderResourceView_Release(RLGL.textures[idx].srv); RLGL.textures[idx].srv = NULL; }
    if (RLGL.textures[idx].texture) { ID3D11Texture2D_Release(RLGL.textures[idx].texture); RLGL.textures[idx].texture = NULL; }
    if (RLGL.textures[idx].sampler) { ID3D11SamplerState_Release(RLGL.textures[idx].sampler); RLGL.textures[idx].sampler = NULL; }
    if (RLGL.textures[idx].dsv) { ID3D11DepthStencilView_Release(RLGL.textures[idx].dsv); RLGL.textures[idx].dsv = NULL; }
    if (RLGL.textures[idx].rtv) { ID3D11RenderTargetView_Release(RLGL.textures[idx].rtv); RLGL.textures[idx].rtv = NULL; }
}

void rlGenTextureMipmaps(unsigned int id, int width, int height, int format, int *mipmaps) {
    if (id == 0 || id > RLGL.textureCount || !RLGL.context) return;
    unsigned int idx = id - 1;
    if (RLGL.textures[idx].srv) ID3D11DeviceContext_GenerateMips(RLGL.context, RLGL.textures[idx].srv);
    if (mipmaps) {
        int max = (width > height) ? width : height;
        *mipmaps = 1 + (int)floor(log((double)max)/log(2.0));
    }
}

void *rlReadTexturePixels(unsigned int id, int width, int height, int format) {
    if (id == 0 || id > RLGL.textureCount || !RLGL.context || !RLGL.device) return NULL;
    unsigned int idx = id - 1;

    // Create staging texture
    D3D11_TEXTURE2D_DESC desc = { 0 };
    ID3D11Texture2D_GetDesc(RLGL.textures[idx].texture, &desc);
    desc.Usage = D3D11_USAGE_STAGING;
    desc.BindFlags = 0;
    desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;

    ID3D11Texture2D *staging = NULL;
    HRESULT hr = ID3D11Device_CreateTexture2D(RLGL.device, &desc, NULL, &staging);
    if (FAILED(hr)) return NULL;

    ID3D11DeviceContext_CopyResource(RLGL.context, (ID3D11Resource *)staging, (ID3D11Resource *)RLGL.textures[idx].texture);

    D3D11_MAPPED_SUBRESOURCE mapped;
    hr = ID3D11DeviceContext_Map(RLGL.context, (ID3D11Resource *)staging, 0, D3D11_MAP_READ, 0, &mapped);
    if (FAILED(hr)) { ID3D11Texture2D_Release(staging); return NULL; }

    int dataSize = rlGetPixelDataSize(width, height, format);
    void *pixels = RL_MALLOC(dataSize);
    memcpy(pixels, mapped.pData, dataSize);

    ID3D11DeviceContext_Unmap(RLGL.context, (ID3D11Resource *)staging, 0);
    ID3D11Texture2D_Release(staging);

    return pixels;
}

unsigned char *rlReadScreenPixels(int width, int height) {
    // TODO: Implement using backbuffer copy
    unsigned char *data = (unsigned char *)RL_CALLOC(width*height*4, sizeof(unsigned char));
    return data;
}

//----------------------------------------------------------------------------------
// Framebuffer management
//----------------------------------------------------------------------------------
unsigned int rlLoadFramebuffer(void) {
    if (RLGL.framebufferCount >= RL_D3D11_MAX_FRAMEBUFFERS) return 0;
    unsigned int idx = RLGL.framebufferCount;
    memset(&RLGL.framebuffers[idx], 0, sizeof(rlD3D11Framebuffer));
    RLGL.framebufferCount++;
    return idx + 1;
}

void rlFramebufferAttach(unsigned int fboId, unsigned int texId, int attachType, int texType, int mipLevel) {
    (void)texType; (void)mipLevel;
    if (fboId == 0 || fboId > RLGL.framebufferCount) return;
    if (texId == 0 || texId > RLGL.textureCount) return;
    unsigned int fbIdx = fboId - 1;
    unsigned int texIdx = texId - 1;

    if (attachType >= 0 && attachType <= 7) { // Color attachment
        if (RLGL.textures[texIdx].rtv == NULL) {
            D3D11_RENDER_TARGET_VIEW_DESC rtvDesc = { 0 };
            rtvDesc.Format = rlGetDXGIFormat(RLGL.textures[texIdx].format);
            rtvDesc.ViewDimension = D3D11_RTV_DIMENSION_TEXTURE2D;
            ID3D11Device_CreateRenderTargetView(RLGL.device, (ID3D11Resource *)RLGL.textures[texIdx].texture, &rtvDesc, &RLGL.textures[texIdx].rtv);
        }
        RLGL.framebuffers[fbIdx].rtv[attachType] = RLGL.textures[texIdx].rtv;
        if (attachType >= RLGL.framebuffers[fbIdx].colorCount) RLGL.framebuffers[fbIdx].colorCount = attachType + 1;
    } else if (attachType == 100) { // Depth
        RLGL.framebuffers[fbIdx].dsv = RLGL.textures[texIdx].dsv;
    }
}

bool rlFramebufferComplete(unsigned int id) {
    if (id == 0 || id > RLGL.framebufferCount) return false;
    unsigned int idx = id - 1;
    RLGL.framebuffers[idx].complete = (RLGL.framebuffers[idx].colorCount > 0 || RLGL.framebuffers[idx].dsv != NULL);
    return RLGL.framebuffers[idx].complete;
}

void rlUnloadFramebuffer(unsigned int id) {
    if (id == 0 || id > RLGL.framebufferCount) return;
    unsigned int idx = id - 1;
    memset(&RLGL.framebuffers[idx], 0, sizeof(rlD3D11Framebuffer));
}

//----------------------------------------------------------------------------------
// Vertex buffer management
//----------------------------------------------------------------------------------
unsigned int rlLoadVertexArray(void) { return 1; } // VAO concept mapped simply
void rlUnloadVertexArray(unsigned int vaoId) { (void)vaoId; }
bool rlEnableVertexArray(unsigned int vaoId) { (void)vaoId; return false; } // Return false to force per-VBO binding path in DrawMesh
void rlDisableVertexArray(void) { }

unsigned int rlLoadVertexBuffer(const void *buffer, int size, bool dynamic) {
    if (!RLGL.device || RLGL.bufferCount >= RL_D3D11_MAX_BUFFERS) return 0;
    unsigned int idx = RLGL.bufferCount;
    RLGL.buffers[idx].buffer = rlCreateD3D11Buffer(buffer, size, D3D11_BIND_VERTEX_BUFFER, dynamic);
    RLGL.buffers[idx].size = size;
    RLGL.buffers[idx].dynamic = dynamic;
    RLGL.buffers[idx].isIndex = false;
    if (!RLGL.buffers[idx].buffer) return 0;
    RLGL.bufferCount++;
    return idx + 1;
}

unsigned int rlLoadVertexBufferElement(const void *buffer, int size, bool dynamic) {
    if (!RLGL.device || RLGL.bufferCount >= RL_D3D11_MAX_BUFFERS) return 0;
    unsigned int idx = RLGL.bufferCount;
    RLGL.buffers[idx].buffer = rlCreateD3D11Buffer(buffer, size, D3D11_BIND_INDEX_BUFFER, dynamic);
    RLGL.buffers[idx].size = size;
    RLGL.buffers[idx].dynamic = dynamic;
    RLGL.buffers[idx].isIndex = true;
    if (!RLGL.buffers[idx].buffer) return 0;
    RLGL.bufferCount++;
    return idx + 1;
}

void rlUpdateVertexBuffer(unsigned int id, const void *data, int dataSize, int offset) {
    if (id == 0 || id > RLGL.bufferCount || !RLGL.context) return;
    unsigned int idx = id - 1;
    if (RLGL.buffers[idx].dynamic) {
        D3D11_MAPPED_SUBRESOURCE mapped;
        if (SUCCEEDED(ID3D11DeviceContext_Map(RLGL.context, (ID3D11Resource *)RLGL.buffers[idx].buffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped))) {
            memcpy((char *)mapped.pData + offset, data, dataSize);
            ID3D11DeviceContext_Unmap(RLGL.context, (ID3D11Resource *)RLGL.buffers[idx].buffer, 0);
        }
    } else {
        D3D11_BOX box = { (UINT)offset, 0, 0, (UINT)(offset + dataSize), 1, 1 };
        ID3D11DeviceContext_UpdateSubresource(RLGL.context, (ID3D11Resource *)RLGL.buffers[idx].buffer, 0, &box, data, dataSize, 0);
    }
}

void rlUpdateVertexBufferElements(unsigned int id, const void *data, int dataSize, int offset) {
    rlUpdateVertexBuffer(id, data, dataSize, offset);
}

void rlUnloadVertexBuffer(unsigned int vboId) {
    if (vboId == 0 || vboId > RLGL.bufferCount) return;
    unsigned int idx = vboId - 1;
    if (RLGL.buffers[idx].buffer) { ID3D11Buffer_Release(RLGL.buffers[idx].buffer); RLGL.buffers[idx].buffer = NULL; }
}

void rlEnableVertexBuffer(unsigned int id) {
    // Defer actual binding — rlSetVertexAttribute will do the IASetVertexBuffers call
    // with the correct input slot and stride
    d3d11_pendingVBO = id;
}
void rlDisableVertexBuffer(void) { d3d11_pendingVBO = 0; }
void rlEnableVertexBufferElement(unsigned int id) {
    if (id == 0 || id > RLGL.bufferCount || !RLGL.context) return;
    unsigned int idx = id - 1;
    ID3D11DeviceContext_IASetIndexBuffer(RLGL.context, RLGL.buffers[idx].buffer, DXGI_FORMAT_R16_UINT, 0);
}
void rlDisableVertexBufferElement(void) { }
void rlEnableVertexAttribute(unsigned int index) { (void)index; }
void rlDisableVertexAttribute(unsigned int index) { (void)index; }
void rlSetVertexAttribute(unsigned int index, int compSize, int type, bool normalized, int stride, int offset) {
    (void)normalized;
    // In D3D11, input layout handles the format. Here we perform the deferred
    // IASetVertexBuffers call using the pending VBO and the known attribute info.
    if (d3d11_pendingVBO > 0 && d3d11_pendingVBO <= RLGL.bufferCount && RLGL.context) {
        unsigned int bufIdx = d3d11_pendingVBO - 1;
        int elemSize = (type == RL_UNSIGNED_BYTE) ? 1 : (int)sizeof(float);
        UINT actualStride = (stride > 0) ? (UINT)stride : (UINT)(compSize * elemSize);
        UINT bufOffset = (UINT)offset;
        ID3D11DeviceContext_IASetVertexBuffers(RLGL.context, index, 1, &RLGL.buffers[bufIdx].buffer, &actualStride, &bufOffset);
        d3d11_pendingVBO = 0;
    }
}
void rlSetVertexAttributeDivisor(unsigned int index, int divisor) { (void)index; (void)divisor; }
void rlSetVertexAttributeDefault(int locIndex, const void *value, int attribType, int count) {
    // In OpenGL, glVertexAttrib4f sets a constant default attribute value when the attribute array is disabled.
    // In D3D11, there's no equivalent. Instead, we bind a default white vertex buffer to provide (1,1,1,1) data.
    // This is primarily used for the COLOR attribute (locIndex=3) when meshes don't have vertex colors.
    (void)value; (void)attribType; (void)count;
    if (locIndex < 0 || !RLGL.context || !d3d11_defaultWhiteVBO) return;
    UINT stride = 4;  // R8G8B8A8_UNORM = 4 bytes per vertex
    UINT offset = 0;
    ID3D11DeviceContext_IASetVertexBuffers(RLGL.context, locIndex, 1, &d3d11_defaultWhiteVBO, &stride, &offset);
}

void rlDrawVertexArray(int offset, int count) {
    if (!RLGL.context) return;
    if (RLGL.stateDirty) rlUpdateD3D11States();  // Apply pending state changes (depth, cull, blend)
    ID3D11DeviceContext_IASetPrimitiveTopology(RLGL.context, D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    ID3D11DeviceContext_Draw(RLGL.context, count, offset);
}

void rlDrawVertexArrayElements(int offset, int count, const void *buffer) {
    (void)buffer;
    if (!RLGL.context) return;
    if (RLGL.stateDirty) rlUpdateD3D11States();  // Apply pending state changes (depth, cull, blend)
    ID3D11DeviceContext_IASetPrimitiveTopology(RLGL.context, D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    ID3D11DeviceContext_DrawIndexed(RLGL.context, count, offset, 0);
}

void rlDrawVertexArrayInstanced(int offset, int count, int instances) {
    (void)offset;
    if (!RLGL.context) return;
    if (RLGL.stateDirty) rlUpdateD3D11States();  // Apply pending state changes (depth, cull, blend)
    ID3D11DeviceContext_IASetPrimitiveTopology(RLGL.context, D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    ID3D11DeviceContext_DrawInstanced(RLGL.context, count, instances, 0, 0);
}

void rlDrawVertexArrayElementsInstanced(int offset, int count, const void *buffer, int instances) {
    (void)buffer;
    if (!RLGL.context) return;
    if (RLGL.stateDirty) rlUpdateD3D11States();  // Apply pending state changes (depth, cull, blend)
    ID3D11DeviceContext_IASetPrimitiveTopology(RLGL.context, D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    ID3D11DeviceContext_DrawIndexedInstanced(RLGL.context, count, instances, offset, 0, 0);
}

//----------------------------------------------------------------------------------
// Shader management
//----------------------------------------------------------------------------------
unsigned int rlCompileShader(const char *shaderCode, int type)
{
    if (!RLGL.device || RLGL.shaderCount >= RL_D3D11_MAX_SHADERS) return 0;

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

    // Store blob temporarily - actual shader creation happens in rlLoadShaderProgram
    // We use a simple scheme: odd IDs for vertex shaders, even for pixel
    unsigned int id = RLGL.shaderCount + 1;

    if (type == RL_VERTEX_SHADER) {
        // Just allocate the shader entry and store the blob
        if (RLGL.shaderCount < RL_D3D11_MAX_SHADERS) {
            RLGL.shaders[RLGL.shaderCount].vsBlob = blob;
            ID3D11Device_CreateVertexShader(RLGL.device, ID3D10Blob_GetBufferPointer(blob),
                ID3D10Blob_GetBufferSize(blob), NULL, &RLGL.shaders[RLGL.shaderCount].vertexShader);
        }
    } else {
        if (RLGL.shaderCount < RL_D3D11_MAX_SHADERS) {
            RLGL.shaders[RLGL.shaderCount].psBlob = blob;
            ID3D11Device_CreatePixelShader(RLGL.device, ID3D10Blob_GetBufferPointer(blob),
                ID3D10Blob_GetBufferSize(blob), NULL, &RLGL.shaders[RLGL.shaderCount].pixelShader);
        }
    }

    return id;
}

unsigned int rlLoadShaderCode(const char *vsCode, const char *fsCode)
{
    if (!vsCode) vsCode = defaultVShaderHLSL;
    if (!fsCode) fsCode = defaultPShaderHLSL;

    if (RLGL.shaderCount >= RL_D3D11_MAX_SHADERS) return 0;
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

    // Create shaders
    ID3D11Device_CreateVertexShader(RLGL.device, ID3D10Blob_GetBufferPointer(vsBlob), ID3D10Blob_GetBufferSize(vsBlob), NULL, &RLGL.shaders[idx].vertexShader);
    ID3D11Device_CreatePixelShader(RLGL.device, ID3D10Blob_GetBufferPointer(psBlob), ID3D10Blob_GetBufferSize(psBlob), NULL, &RLGL.shaders[idx].pixelShader);

    // Create input layout
    D3D11_INPUT_ELEMENT_DESC layout[] = {
        { "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT,    0, 0,                            D3D11_INPUT_PER_VERTEX_DATA, 0 },
        { "TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT,       1, 0,                            D3D11_INPUT_PER_VERTEX_DATA, 0 },
        { "NORMAL",   0, DXGI_FORMAT_R32G32B32_FLOAT,    2, 0,                            D3D11_INPUT_PER_VERTEX_DATA, 0 },
        { "COLOR",    0, DXGI_FORMAT_R8G8B8A8_UNORM,     3, 0,                            D3D11_INPUT_PER_VERTEX_DATA, 0 },
    };
    ID3D11Device_CreateInputLayout(RLGL.device, layout, 4,
        ID3D10Blob_GetBufferPointer(vsBlob), ID3D10Blob_GetBufferSize(vsBlob), &RLGL.shaders[idx].inputLayout);

    RLGL.shaders[idx].vsBlob = vsBlob;
    RLGL.shaders[idx].psBlob = psBlob;
    RLGL.shaders[idx].cbufferCount = 0;
    RLGL.shaders[idx].uniformCount = 0;
    RLGL.shaderCount++;

    // Reflect shader to discover constant buffers and uniform variables
    rlReflectShaderUniforms(idx);

    unsigned int id = idx + 1;
    TRACELOG(RL_LOG_INFO, "SHADER: [ID %i] D3D11 shader loaded successfully (uniforms: %d, cbuffers: %d)", id, RLGL.shaders[idx].uniformCount, RLGL.shaders[idx].cbufferCount);
    return id;
}

unsigned int rlLoadShaderProgram(unsigned int vShaderId, unsigned int fShaderId) {
    (void)vShaderId; (void)fShaderId;
    return 0; // Not used in D3D11 path - rlLoadShaderCode handles everything
}

void rlUnloadShaderProgram(unsigned int id) {
    if (id == 0 || id > RLGL.shaderCount) return;
    unsigned int idx = id - 1;
    if (RLGL.shaders[idx].vertexShader) { ID3D11VertexShader_Release(RLGL.shaders[idx].vertexShader); RLGL.shaders[idx].vertexShader = NULL; }
    if (RLGL.shaders[idx].pixelShader) { ID3D11PixelShader_Release(RLGL.shaders[idx].pixelShader); RLGL.shaders[idx].pixelShader = NULL; }
    if (RLGL.shaders[idx].inputLayout) { ID3D11InputLayout_Release(RLGL.shaders[idx].inputLayout); RLGL.shaders[idx].inputLayout = NULL; }
    if (RLGL.shaders[idx].vsBlob) { ID3D10Blob_Release(RLGL.shaders[idx].vsBlob); RLGL.shaders[idx].vsBlob = NULL; }
    if (RLGL.shaders[idx].psBlob) { ID3D10Blob_Release(RLGL.shaders[idx].psBlob); RLGL.shaders[idx].psBlob = NULL; }
    // Clean up constant buffers and CPU data
    for (int i = 0; i < RLGL.shaders[idx].cbufferCount; i++) {
        if (RLGL.shaders[idx].cbuffers[i].buffer) { ID3D11Buffer_Release(RLGL.shaders[idx].cbuffers[i].buffer); RLGL.shaders[idx].cbuffers[i].buffer = NULL; }
        if (RLGL.shaders[idx].cbuffers[i].cpuData) { RL_FREE(RLGL.shaders[idx].cbuffers[i].cpuData); RLGL.shaders[idx].cbuffers[i].cpuData = NULL; }
    }
    RLGL.shaders[idx].cbufferCount = 0;
    RLGL.shaders[idx].uniformCount = 0;
}

int rlGetLocationUniform(unsigned int shaderId, const char *uniformName) {
    if (shaderId == 0 || shaderId > RLGL.shaderCount || !uniformName) return -1;
    unsigned int idx = shaderId - 1;
    // First check cbuffer uniforms
    for (int i = 0; i < RLGL.shaders[idx].uniformCount; i++) {
        if (strcmp(RLGL.shaders[idx].uniforms[i].name, uniformName) == 0)
            return i;
    }
    // Then check texture/SRV bindings (return special encoded value)
    for (int i = 0; i < RLGL.shaders[idx].textureBindingCount; i++) {
        if (strcmp(RLGL.shaders[idx].textureBindings[i].name, uniformName) == 0)
            return RL_D3D11_TEX_LOC_OFFSET + i;
    }
    return -1; // Not found (may be unused by this shader)
}

int rlGetLocationAttrib(unsigned int shaderId, const char *attribName) {
    (void)shaderId;
    if (!attribName) return -1;
    // Map known raylib attribute names to D3D11 input layout slot indices
    if (strcmp(attribName, "vertexPosition") == 0) return 0;
    if (strcmp(attribName, "vertexTexCoord") == 0) return 1;
    if (strcmp(attribName, "vertexNormal") == 0) return 2;
    if (strcmp(attribName, "vertexColor") == 0) return 3;
    if (strcmp(attribName, "vertexTangent") == 0) return -1;      // Not in our 4-slot input layout
    if (strcmp(attribName, "vertexTexCoord2") == 0) return -1;    // Not in our 4-slot input layout
    return -1;
}

void rlSetUniform(int locIndex, const void *value, int uniformType, int count) {
    if (locIndex < 0 || !value || !RLGL.context) return;
    if (d3d11_activeShaderId == 0 || d3d11_activeShaderId > RLGL.shaderCount) return;
    unsigned int shaderIdx = d3d11_activeShaderId - 1;
    rlD3D11Shader *s = &RLGL.shaders[shaderIdx];

    // Handle texture binding locations (locIndex >= RL_D3D11_TEX_LOC_OFFSET)
    // In OpenGL, SetShaderValue(shader, texLoc, &mapIndex, INT) sets which texture unit the sampler reads from.
    // In D3D11, we store the material-map-index → register mapping so rlEnableTexture can bind to the right slot.
    if (locIndex >= RL_D3D11_TEX_LOC_OFFSET) {
        if (uniformType != RL_SHADER_UNIFORM_INT) return;
        int texBindIdx = locIndex - RL_D3D11_TEX_LOC_OFFSET;
        if (texBindIdx < 0 || texBindIdx >= s->textureBindingCount) return;
        int registerSlot = s->textureBindings[texBindIdx].registerSlot;
        int materialMapIndex = *(const int *)value;
        if (materialMapIndex >= 0 && materialMapIndex < RL_MAX_MATERIAL_MAPS) {
            s->texMaterialMapToRegister[materialMapIndex] = registerSlot;
            TRACELOG(RL_LOG_DEBUG, "SHADER: [ID %i] Texture '%s' mapped: material map %d -> register(t%d)",
                     (int)d3d11_activeShaderId, s->textureBindings[texBindIdx].name, materialMapIndex, registerSlot);
        }
        return;
    }

    if (locIndex >= s->uniformCount) return;

    rlD3D11UniformVar *u = &s->uniforms[locIndex];
    rlD3D11CBufferInfo *cb = &s->cbuffers[u->cbufferIndex];

    // Compute data size based on uniform type
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
        case RL_SHADER_UNIFORM_SAMPLER2D: return; // Textures handled separately in D3D11
        default: return;
    }
    if (dataSize > u->byteSize) dataSize = u->byteSize; // Clamp to variable size

    // Update CPU-side copy
    if (cb->cpuData) memcpy(cb->cpuData + u->byteOffset, value, dataSize);

    // Upload entire cbuffer to GPU and bind
    if (cb->buffer) {
        D3D11_MAPPED_SUBRESOURCE mapped;
        if (SUCCEEDED(ID3D11DeviceContext_Map(RLGL.context, (ID3D11Resource *)cb->buffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped))) {
            memcpy(mapped.pData, cb->cpuData, cb->byteSize);
            ID3D11DeviceContext_Unmap(RLGL.context, (ID3D11Resource *)cb->buffer, 0);
        }
        if (cb->stage == 0) ID3D11DeviceContext_VSSetConstantBuffers(RLGL.context, cb->registerSlot, 1, &cb->buffer);
        else                ID3D11DeviceContext_PSSetConstantBuffers(RLGL.context, cb->registerSlot, 1, &cb->buffer);
    }
}

void rlSetUniformMatrix(int locIndex, Matrix mat) {
    if (locIndex < 0 || !RLGL.context) return;
    if (d3d11_activeShaderId == 0 || d3d11_activeShaderId > RLGL.shaderCount) return;
    unsigned int shaderIdx = d3d11_activeShaderId - 1;
    rlD3D11Shader *s = &RLGL.shaders[shaderIdx];
    if (locIndex >= s->uniformCount) return;

    rlD3D11UniformVar *u = &s->uniforms[locIndex];
    rlD3D11CBufferInfo *cb = &s->cbuffers[u->cbufferIndex];

    // Convert Matrix to float[16] and update CPU-side copy
    rl_float16 matData = rlMatrixToFloatV(mat);
    int copySize = (u->byteSize < (int)sizeof(float)*16) ? u->byteSize : (int)sizeof(float)*16;
    if (cb->cpuData) memcpy(cb->cpuData + u->byteOffset, matData.v, copySize);

    // Upload and bind
    if (cb->buffer) {
        D3D11_MAPPED_SUBRESOURCE mapped;
        if (SUCCEEDED(ID3D11DeviceContext_Map(RLGL.context, (ID3D11Resource *)cb->buffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped))) {
            memcpy(mapped.pData, cb->cpuData, cb->byteSize);
            ID3D11DeviceContext_Unmap(RLGL.context, (ID3D11Resource *)cb->buffer, 0);
        }
        if (cb->stage == 0) ID3D11DeviceContext_VSSetConstantBuffers(RLGL.context, cb->registerSlot, 1, &cb->buffer);
        else                ID3D11DeviceContext_PSSetConstantBuffers(RLGL.context, cb->registerSlot, 1, &cb->buffer);
    }
}

void rlSetUniformMatrices(int locIndex, const Matrix *mat, int count) {
    // For bone matrices: update as an array of float4x4
    if (locIndex < 0 || !mat || !RLGL.context || count <= 0) return;
    if (d3d11_activeShaderId == 0 || d3d11_activeShaderId > RLGL.shaderCount) return;
    unsigned int shaderIdx = d3d11_activeShaderId - 1;
    rlD3D11Shader *s = &RLGL.shaders[shaderIdx];
    if (locIndex >= s->uniformCount) return;

    rlD3D11UniformVar *u = &s->uniforms[locIndex];
    rlD3D11CBufferInfo *cb = &s->cbuffers[u->cbufferIndex];

    int totalSize = count * (int)sizeof(float) * 16;
    if (totalSize > u->byteSize) totalSize = u->byteSize;

    // Convert matrices and copy to CPU buffer
    if (cb->cpuData) {
        for (int i = 0; i < count && (u->byteOffset + (i + 1) * 64) <= cb->byteSize; i++) {
            rl_float16 matData = rlMatrixToFloatV(mat[i]);
            memcpy(cb->cpuData + u->byteOffset + i * 64, matData.v, 64);
        }
    }

    if (cb->buffer) {
        D3D11_MAPPED_SUBRESOURCE mapped;
        if (SUCCEEDED(ID3D11DeviceContext_Map(RLGL.context, (ID3D11Resource *)cb->buffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped))) {
            memcpy(mapped.pData, cb->cpuData, cb->byteSize);
            ID3D11DeviceContext_Unmap(RLGL.context, (ID3D11Resource *)cb->buffer, 0);
        }
        if (cb->stage == 0) ID3D11DeviceContext_VSSetConstantBuffers(RLGL.context, cb->registerSlot, 1, &cb->buffer);
        else                ID3D11DeviceContext_PSSetConstantBuffers(RLGL.context, cb->registerSlot, 1, &cb->buffer);
    }
}

void rlSetUniformSampler(int locIndex, unsigned int textureId) {
    // In D3D11, textures are bound via rlEnableTexture/rlActiveTextureSlot, not via uniforms
    (void)locIndex; (void)textureId;
}

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
    TRACELOG(RL_LOG_WARNING, "SHADER: Compute shaders require D3D11 CS5.0, use GRAPHICS_API_OPENGL_43 for full support");
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
        // Left/Right/Bottom/Top faces (abbreviated for space, same pattern)
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
// Shader reflection: extract constant buffer and uniform variable info
//----------------------------------------------------------------------------------
static void rlReflectShaderStage(unsigned int shaderIdx, ID3DBlob *blob, int stage)
{
    if (!blob) return;
    rlD3D11Shader *s = &RLGL.shaders[shaderIdx];

    ID3D11ShaderReflection *reflect = NULL;
    HRESULT hr = D3DReflect(ID3D10Blob_GetBufferPointer(blob), ID3D10Blob_GetBufferSize(blob),
                            &IID_ID3D11ShaderReflection, (void **)&reflect);
    if (FAILED(hr) || !reflect) return;

    D3D11_SHADER_DESC shaderDesc;
    reflect->lpVtbl->GetDesc(reflect, &shaderDesc);

    for (UINT i = 0; i < shaderDesc.ConstantBuffers && s->cbufferCount < RL_D3D11_MAX_CBUFFERS; i++) {
        ID3D11ShaderReflectionConstantBuffer *cbReflect =
            reflect->lpVtbl->GetConstantBufferByIndex(reflect, i);
        D3D11_SHADER_BUFFER_DESC cbDesc;
        hr = cbReflect->lpVtbl->GetDesc(cbReflect, &cbDesc);
        if (FAILED(hr)) continue;
        if (cbDesc.Type != D3D_CT_CBUFFER) continue; // Skip tbuffers etc.

        // Find the bind point (register slot) for this cbuffer
        D3D11_SHADER_INPUT_BIND_DESC bindDesc;
        hr = reflect->lpVtbl->GetResourceBindingDescByName(reflect, cbDesc.Name, &bindDesc);
        int regSlot = SUCCEEDED(hr) ? (int)bindDesc.BindPoint : (int)i;

        int cbIdx = s->cbufferCount;
        s->cbuffers[cbIdx].stage = stage;
        s->cbuffers[cbIdx].registerSlot = regSlot;
        s->cbuffers[cbIdx].byteSize = (int)cbDesc.Size;
        // Create GPU constant buffer (dynamic for frequent updates)
        s->cbuffers[cbIdx].buffer = rlCreateD3D11Buffer(NULL, cbDesc.Size, D3D11_BIND_CONSTANT_BUFFER, true);
        // Allocate CPU-side copy (zeroed)
        s->cbuffers[cbIdx].cpuData = (unsigned char *)RL_CALLOC(1, cbDesc.Size);
        s->cbufferCount++;

        // Enumerate variables within this constant buffer
        for (UINT j = 0; j < cbDesc.Variables && s->uniformCount < RL_D3D11_MAX_UNIFORM_VARS; j++) {
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

            // For struct/array variables, also add entries for array elements and struct members
            // This enables lookups like "lights[0].enabled"
            ID3D11ShaderReflectionType *varType = varReflect->lpVtbl->GetType(varReflect);
            if (varType) {
                D3D11_SHADER_TYPE_DESC typeDesc;
                hr = varType->lpVtbl->GetDesc(varType, &typeDesc);
                if (SUCCEEDED(hr) && typeDesc.Class == D3D_SVC_STRUCT && typeDesc.Elements > 0) {
                    // It's an array of structs — expand entries for member access
                    UINT structStride = varDesc.Size / typeDesc.Elements;
                    for (UINT elem = 0; elem < typeDesc.Elements; elem++) {
                        for (UINT m = 0; m < typeDesc.Members && s->uniformCount < RL_D3D11_MAX_UNIFORM_VARS; m++) {
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
                            // Compute member size from type
                            int memberSize = 4; // default
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

    // Reflect texture/SRV bindings (only for pixel shader stage)
    if (stage == 1) {
        for (UINT i = 0; i < shaderDesc.BoundResources && s->textureBindingCount < RL_D3D11_MAX_TEXTURE_BINDINGS; i++) {
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
    rlD3D11Shader *s = &RLGL.shaders[shaderIdx];
    s->cbufferCount = 0;
    s->uniformCount = 0;
    s->textureBindingCount = 0;

    // Initialize material-map-to-register mapping to identity (slot i → register i)
    for (int i = 0; i < RL_MAX_MATERIAL_MAPS; i++) s->texMaterialMapToRegister[i] = i;

    // Reflect VS constant buffers (stage=0)
    rlReflectShaderStage(shaderIdx, s->vsBlob, 0);
    // Reflect PS constant buffers and texture bindings (stage=1)
    rlReflectShaderStage(shaderIdx, s->psBlob, 1);
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
        TRACELOG(RL_LOG_INFO, "SHADER: [ID %i] Default D3D11 shader loaded successfully", RLGL.State.defaultShaderId);

        // Set default shader attribute locations (input layout slots)
        RLGL.State.defaultShaderLocs[RL_SHADER_LOC_VERTEX_POSITION] = rlGetLocationAttrib(RLGL.State.defaultShaderId, "vertexPosition");
        RLGL.State.defaultShaderLocs[RL_SHADER_LOC_VERTEX_TEXCOORD01] = rlGetLocationAttrib(RLGL.State.defaultShaderId, "vertexTexCoord");
        RLGL.State.defaultShaderLocs[RL_SHADER_LOC_VERTEX_COLOR] = rlGetLocationAttrib(RLGL.State.defaultShaderId, "vertexColor");

        // Set default shader uniform locations (from reflection)
        RLGL.State.defaultShaderLocs[RL_SHADER_LOC_MATRIX_MVP] = rlGetLocationUniform(RLGL.State.defaultShaderId, "mvp");
        RLGL.State.defaultShaderLocs[RL_SHADER_LOC_COLOR_DIFFUSE] = rlGetLocationUniform(RLGL.State.defaultShaderId, "colDiffuse");
        RLGL.State.defaultShaderLocs[RL_SHADER_LOC_MAP_DIFFUSE] = rlGetLocationUniform(RLGL.State.defaultShaderId, "texture0");
    }
}

static void rlUnloadShaderDefault(void)
{
    if (RLGL.State.defaultShaderId > 0) rlUnloadShaderProgram(RLGL.State.defaultShaderId);
    RL_FREE(RLGL.State.defaultShaderLocs);
}

//----------------------------------------------------------------------------------
// Auxiliar math functions (identical to OpenGL backend)
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

// Headless render viewport stubs (compatibility with existing custom functions)
void rlSetHeadlessRenderViewport(int width, int height) { (void)width; (void)height; }
void rlUnsetHeadlessRenderViewport(void) { }

#endif // GRAPHICS_API_DIRECT3D11
