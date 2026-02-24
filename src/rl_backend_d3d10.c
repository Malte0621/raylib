/**********************************************************************************************
*
*   rl_backend_d3d10 - Direct3D 10 rendering backend for rlgl
*
*   This implements all rlgl functions using the Direct3D 10 API.
*   Requires Windows Vista or later.
*
*   Differences from D3D11:
*     - No separate device context (all calls go directly on ID3D10Device)
*     - Shader model 4.0 (vs_4_0 / ps_4_0)
*     - No feature levels
*     - Map/Unmap are methods on the resource, not on a context
*     - Viewport TopLeftX/Y are INT (not FLOAT)
*
*   CONFIGURATION:
*       #define GRAPHICS_API_DIRECT3D10   Select this backend
*
*   LICENSE: zlib/libpng (same as raylib)
*
**********************************************************************************************/

#include "raylib.h"
#include "rlgl.h"
#include "utils.h"
#include "rl_backend.h"

#if defined(GRAPHICS_API_DIRECT3D10)

#include <stdlib.h>
#include <string.h>
#include <math.h>

// Windows / D3D10 headers
#define WIN32_LEAN_AND_MEAN
#define COBJMACROS
#define CINTERFACE
#include <windows.h>
#include <d3d10.h>
#include <d3dcompiler.h>
#include <dxgi.h>

#ifdef _MSC_VER
    #pragma comment(lib, "d3d10.lib")
    #pragma comment(lib, "d3dcompiler.lib")
    #pragma comment(lib, "dxgi.lib")
#endif

//----------------------------------------------------------------------------------
// Defines and Macros
//----------------------------------------------------------------------------------
#ifndef RL_D3D10_MAX_TEXTURES
    #define RL_D3D10_MAX_TEXTURES       4096
#endif
#ifndef RL_D3D10_MAX_SHADERS
    #define RL_D3D10_MAX_SHADERS        256
#endif
#ifndef RL_D3D10_MAX_BUFFERS
    #define RL_D3D10_MAX_BUFFERS        4096
#endif
#ifndef RL_D3D10_MAX_FRAMEBUFFERS
    #define RL_D3D10_MAX_FRAMEBUFFERS   256
#endif

//----------------------------------------------------------------------------------
// Types and Structures
//----------------------------------------------------------------------------------

// Tracked D3D10 texture resource
typedef struct rlD3D10Texture {
    ID3D10Texture2D *texture;
    ID3D10ShaderResourceView *srv;
    ID3D10SamplerState *sampler;
    ID3D10DepthStencilView *dsv;
    ID3D10RenderTargetView *rtv;
    int width, height, mipmaps, format;
} rlD3D10Texture;

// Tracked D3D10 shader
typedef struct rlD3D10Shader {
    ID3D10VertexShader *vertexShader;
    ID3D10PixelShader *pixelShader;
    ID3D10InputLayout *inputLayout;
    ID3D10Blob *vsBlob;
    ID3D10Blob *psBlob;
} rlD3D10Shader;

// Tracked D3D10 buffer (for user VBOs/IBOs)
typedef struct rlD3D10Buffer {
    ID3D10Buffer *buffer;
    unsigned int size;
} rlD3D10Buffer;

// Tracked D3D10 framebuffer
typedef struct rlD3D10Framebuffer {
    unsigned int colorTextureId;
    unsigned int depthTextureId;
    ID3D10RenderTargetView *rtv;
    ID3D10DepthStencilView *dsv;
} rlD3D10Framebuffer;

// Constant buffer layout for MVP matrix (16-byte aligned)
typedef struct RL_ALIGN(16) rlD3D10MatrixCB {
    float mvp[16];
} rlD3D10MatrixCB;

// Constant buffer layout for diffuse color
typedef struct RL_ALIGN(16) rlD3D10ColorCB {
    float colDiffuse[4];
} rlD3D10ColorCB;

//----------------------------------------------------------------------------------
// Global Variables Definition (rlglData equivalent)
//----------------------------------------------------------------------------------
typedef struct {
    struct {
        int currentMatrixMode;
        Matrix *currentMatrix;
        Matrix modelview;
        Matrix projection;
        Matrix transform;
        bool transformRequired;
        Matrix stack[RL_MAX_MATRIX_STACK_SIZE];
        int stackCounter;

        unsigned int defaultTextureId;
        unsigned int activeTextureId;
        unsigned int defaultShaderId;
        int *defaultShaderLocs;
        unsigned int currentShaderId;
        int *currentShaderLocs;

        int framebufferWidth;
        int framebufferHeight;

        bool stereoRender;
        Matrix projectionStereo[2];
        Matrix viewOffsetStereo[2];

        int currentBlendMode;
        int glBlendSrcFactor;
        int glBlendDstFactor;
        int glBlendEquation;
        int glBlendSrcFactorRGB;
        int glBlendDstFactorRGB;
        int glBlendSrcFactorAlpha;
        int glBlendDstFactorAlpha;
        int glBlendEquationRGB;
        int glBlendEquationAlpha;
        bool glCustomBlendModeModified;

        float clearColor[4];

        bool colorBlendEnabled;
        bool depthTestEnabled;
        bool depthWriteEnabled;
        bool backfaceCullingEnabled;
        bool scissorTest;
        float lineWidth;
    } State;

    struct {
        bool vao;
        bool instancing;
        bool texNPOT;
        bool texDepth;
        bool texDepthWebGL;
        bool texFloat32;
        bool texFloat16;
        bool texCompDXT;
        bool texCompETC1;
        bool texCompETC2;
        bool texCompASTC;
        bool computeShader;
        bool ssbo;
        float maxAnisotropyLevel;
        int maxDepthBits;
    } ExtSupported;

    // D3D10 device (no separate context in D3D10)
    ID3D10Device *device;

    // Constant buffers
    ID3D10Buffer *cbMatrix;
    ID3D10Buffer *cbColor;

    // Default sampler
    ID3D10SamplerState *defaultSampler;

    // Render state objects (cached)
    ID3D10BlendState *currentBlendState;
    ID3D10DepthStencilState *currentDSState;
    ID3D10RasterizerState *currentRSState;
    bool stateDirty;

    // Batch rendering vertex buffers
    ID3D10Buffer *batchVertexBuffers[4]; // pos, texcoord, normal, color
    ID3D10Buffer *batchIndexBuffer;

    // Resource tracking
    rlD3D10Texture textures[RL_D3D10_MAX_TEXTURES];
    unsigned int textureCount;

    rlD3D10Shader shaders[RL_D3D10_MAX_SHADERS];
    unsigned int shaderCount;

    rlD3D10Buffer buffers[RL_D3D10_MAX_BUFFERS];
    unsigned int bufferCount;

    rlD3D10Framebuffer framebuffers[RL_D3D10_MAX_FRAMEBUFFERS];
    unsigned int framebufferCount;

    rlRenderBatch defaultBatch;
    rlRenderBatch *currentBatch;
} rlglData;

static rlglData RLGL = { 0 };

//----------------------------------------------------------------------------------
// Forward declarations
//----------------------------------------------------------------------------------
static ID3D10Buffer *rlCreateD3D10Buffer(unsigned int size, unsigned int bindFlags, const void *initData);
static void rlUpdateD3D10States(void);
static DXGI_FORMAT rlGetDXGIFormat(int rlFormat);
static void rlLoadShaderDefault(void);
static void rlUnloadShaderDefault(void);

// Auxiliar math functions
static rl_float16 rlMatrixToFloatV(Matrix mat);
static Matrix rlMatrixIdentity(void);
static Matrix rlMatrixMultiply(Matrix left, Matrix right);
static Matrix rlMatrixTranspose(Matrix mat);
static Matrix rlMatrixInvert(Matrix mat);

//----------------------------------------------------------------------------------
// Default HLSL shaders (Shader Model 4.0)
//----------------------------------------------------------------------------------
static const char *defaultVShaderHLSL =
    "cbuffer MatrixBuffer : register(b0) { float4x4 mvp; };\n"
    "struct VSInput { float3 pos : POSITION; float2 texcoord : TEXCOORD; float3 normal : NORMAL; float4 color : COLOR; };\n"
    "struct VSOutput { float4 pos : SV_POSITION; float2 texcoord : TEXCOORD; float4 color : COLOR; };\n"
    "VSOutput VSMain(VSInput input) {\n"
    "    VSOutput output;\n"
    "    output.pos = mul(mvp, float4(input.pos, 1.0));\n"
    "    output.texcoord = input.texcoord;\n"
    "    output.color = input.color;\n"
    "    return output;\n"
    "}\n";

static const char *defaultPShaderHLSL =
    "cbuffer ColorBuffer : register(b1) { float4 colDiffuse; };\n"
    "Texture2D texture0 : register(t0);\n"
    "SamplerState sampler0 : register(s0);\n"
    "struct PSInput { float4 pos : SV_POSITION; float2 texcoord : TEXCOORD; float4 color : COLOR; };\n"
    "float4 PSMain(PSInput input) : SV_TARGET {\n"
    "    float4 texColor = texture0.Sample(sampler0, input.texcoord);\n"
    "    return texColor * input.color * colDiffuse;\n"
    "}\n";

//----------------------------------------------------------------------------------
// Helper: Create a D3D10 buffer
//----------------------------------------------------------------------------------
static ID3D10Buffer *rlCreateD3D10Buffer(unsigned int size, unsigned int bindFlags, const void *initData)
{
    if (!RLGL.device || size == 0) return NULL;

    D3D10_BUFFER_DESC desc;
    memset(&desc, 0, sizeof(desc));
    desc.ByteWidth = size;
    desc.Usage = D3D10_USAGE_DYNAMIC;
    desc.BindFlags = bindFlags;
    desc.CPUAccessFlags = D3D10_CPU_ACCESS_WRITE;

    ID3D10Buffer *buffer = NULL;

    if (initData) {
        desc.Usage = D3D10_USAGE_DEFAULT;
        desc.CPUAccessFlags = 0;
        D3D10_SUBRESOURCE_DATA srd;
        memset(&srd, 0, sizeof(srd));
        srd.pSysMem = initData;
        ID3D10Device_CreateBuffer(RLGL.device, &desc, &srd, &buffer);
    } else {
        ID3D10Device_CreateBuffer(RLGL.device, &desc, NULL, &buffer);
    }

    return buffer;
}

//----------------------------------------------------------------------------------
// Helper: Update cached D3D10 render states (blend, depth, rasterizer)
//----------------------------------------------------------------------------------
static void rlUpdateD3D10States(void)
{
    if (!RLGL.device || !RLGL.stateDirty) return;
    RLGL.stateDirty = false;

    // Blend state
    {
        if (RLGL.currentBlendState) { ID3D10BlendState_Release(RLGL.currentBlendState); RLGL.currentBlendState = NULL; }

        D3D10_BLEND_DESC bd;
        memset(&bd, 0, sizeof(bd));
        bd.BlendEnable[0] = RLGL.State.colorBlendEnabled ? TRUE : FALSE;
        bd.SrcBlend = D3D10_BLEND_SRC_ALPHA;
        bd.DestBlend = D3D10_BLEND_INV_SRC_ALPHA;
        bd.BlendOp = D3D10_BLEND_OP_ADD;
        bd.SrcBlendAlpha = D3D10_BLEND_ONE;
        bd.DestBlendAlpha = D3D10_BLEND_INV_SRC_ALPHA;
        bd.BlendOpAlpha = D3D10_BLEND_OP_ADD;
        bd.RenderTargetWriteMask[0] = D3D10_COLOR_WRITE_ENABLE_ALL;

        ID3D10Device_CreateBlendState(RLGL.device, &bd, &RLGL.currentBlendState);
        float blendFactor[4] = { 0, 0, 0, 0 };
        ID3D10Device_OMSetBlendState(RLGL.device, RLGL.currentBlendState, blendFactor, 0xFFFFFFFF);
    }

    // Depth-stencil state
    {
        if (RLGL.currentDSState) { ID3D10DepthStencilState_Release(RLGL.currentDSState); RLGL.currentDSState = NULL; }

        D3D10_DEPTH_STENCIL_DESC dsd;
        memset(&dsd, 0, sizeof(dsd));
        dsd.DepthEnable = RLGL.State.depthTestEnabled ? TRUE : FALSE;
        dsd.DepthWriteMask = RLGL.State.depthWriteEnabled ? D3D10_DEPTH_WRITE_MASK_ALL : D3D10_DEPTH_WRITE_MASK_ZERO;
        dsd.DepthFunc = D3D10_COMPARISON_LESS_EQUAL;
        dsd.StencilEnable = FALSE;

        ID3D10Device_CreateDepthStencilState(RLGL.device, &dsd, &RLGL.currentDSState);
        ID3D10Device_OMSetDepthStencilState(RLGL.device, RLGL.currentDSState, 0);
    }

    // Rasterizer state
    {
        if (RLGL.currentRSState) { ID3D10RasterizerState_Release(RLGL.currentRSState); RLGL.currentRSState = NULL; }

        D3D10_RASTERIZER_DESC rd;
        memset(&rd, 0, sizeof(rd));
        rd.FillMode = D3D10_FILL_SOLID;
        rd.CullMode = RLGL.State.backfaceCullingEnabled ? D3D10_CULL_BACK : D3D10_CULL_NONE;
        rd.FrontCounterClockwise = TRUE;
        rd.DepthClipEnable = TRUE;
        rd.ScissorEnable = RLGL.State.scissorTest ? TRUE : FALSE;
        rd.MultisampleEnable = FALSE;
        rd.AntialiasedLineEnable = FALSE;

        ID3D10Device_CreateRasterizerState(RLGL.device, &rd, &RLGL.currentRSState);
        ID3D10Device_RSSetState(RLGL.device, RLGL.currentRSState);
    }
}

//----------------------------------------------------------------------------------
// Helper: Convert raylib pixel format to DXGI_FORMAT
//----------------------------------------------------------------------------------
static DXGI_FORMAT rlGetDXGIFormat(int rlFormat)
{
    switch (rlFormat) {
        case 1:  return DXGI_FORMAT_R8_UNORM;             // GRAYSCALE
        case 2:  return DXGI_FORMAT_R8G8_UNORM;           // GRAY_ALPHA
        case 3:  return DXGI_FORMAT_R5G6B5_UNORM;         // R5G6B5
        case 4:  return DXGI_FORMAT_R8G8B8A8_UNORM;       // R8G8B8 -> expanded to RGBA
        case 5:  return DXGI_FORMAT_R5G5B5A1_UNORM;       // R5G5B5A1
        case 6:  return DXGI_FORMAT_B4G4R4A4_UNORM;       // R4G4B4A4
        case 7:  return DXGI_FORMAT_R8G8B8A8_UNORM;       // R8G8B8A8
        case 8:  return DXGI_FORMAT_R32_FLOAT;             // R32
        case 9:  return DXGI_FORMAT_R32G32B32A32_FLOAT;    // R32G32B32 -> expanded to RGBA32F
        case 10: return DXGI_FORMAT_R32G32B32A32_FLOAT;    // R32G32B32A32
        case 11: return DXGI_FORMAT_R16_FLOAT;             // R16
        case 12: return DXGI_FORMAT_R16G16B16A16_FLOAT;    // R16G16B16 -> expanded to RGBA16F
        case 13: return DXGI_FORMAT_R16G16B16A16_FLOAT;    // R16G16B16A16
        case 14: return DXGI_FORMAT_BC1_UNORM;             // DXT1 RGB
        case 15: return DXGI_FORMAT_BC2_UNORM;             // DXT3 RGBA
        case 16: return DXGI_FORMAT_BC3_UNORM;             // DXT5 RGBA
        case 17: return DXGI_FORMAT_BC1_UNORM;             // DXT1 RGBA
        default: return DXGI_FORMAT_R8G8B8A8_UNORM;
    }
}

//----------------------------------------------------------------------------------
// Matrix state management
//----------------------------------------------------------------------------------
void rlMatrixMode(int mode) { RLGL.State.currentMatrixMode = mode; if (mode == RL_PROJECTION) RLGL.State.currentMatrix = &RLGL.State.projection; else if (mode == RL_MODELVIEW) RLGL.State.currentMatrix = &RLGL.State.modelview; }
void rlPushMatrix(void) { if (RLGL.State.stackCounter >= RL_MAX_MATRIX_STACK_SIZE) { TRACELOG(RL_LOG_ERROR, "RLGL: Matrix stack overflow (RL_MAX_MATRIX_STACK_SIZE)"); return; } if (RLGL.State.currentMatrixMode == RL_MODELVIEW) { RLGL.State.transformRequired = true; RLGL.State.currentMatrix = &RLGL.State.transform; } RLGL.State.stack[RLGL.State.stackCounter] = *RLGL.State.currentMatrix; RLGL.State.stackCounter++; }
void rlPopMatrix(void) { if (RLGL.State.stackCounter > 0) { Matrix mat = RLGL.State.stack[RLGL.State.stackCounter - 1]; *RLGL.State.currentMatrix = mat; RLGL.State.stackCounter--; } if ((RLGL.State.stackCounter == 0) && (RLGL.State.currentMatrixMode == RL_MODELVIEW)) { RLGL.State.currentMatrix = &RLGL.State.modelview; RLGL.State.transformRequired = false; } }
void rlLoadIdentity(void) { *RLGL.State.currentMatrix = rlMatrixIdentity(); }
void rlTranslatef(float x, float y, float z) { Matrix mat = { 1,0,0,x, 0,1,0,y, 0,0,1,z, 0,0,0,1 }; *RLGL.State.currentMatrix = rlMatrixMultiply(mat, *RLGL.State.currentMatrix); }

void rlRotatef(float angle, float x, float y, float z)
{
    Matrix mat = rlMatrixIdentity();
    float len = sqrtf(x*x + y*y + z*z);
    if ((len != 1.0f) && (len != 0.0f)) { float ilen = 1.0f/len; x *= ilen; y *= ilen; z *= ilen; }
    float s = sinf(angle * 0.017453292519943295f);
    float c = cosf(angle * 0.017453292519943295f);
    float t = 1.0f - c;
    mat.m0 = x*x*t+c;   mat.m1 = y*x*t+z*s; mat.m2 = z*x*t-y*s; mat.m3 = 0;
    mat.m4 = x*y*t-z*s; mat.m5 = y*y*t+c;   mat.m6 = z*y*t+x*s; mat.m7 = 0;
    mat.m8 = x*z*t+y*s; mat.m9 = y*z*t-x*s; mat.m10= z*z*t+c;   mat.m11= 0;
    mat.m12= 0;          mat.m13= 0;          mat.m14= 0;          mat.m15= 1;
    *RLGL.State.currentMatrix = rlMatrixMultiply(mat, *RLGL.State.currentMatrix);
}

void rlScalef(float x, float y, float z)
{
    Matrix mat = { x,0,0,0, 0,y,0,0, 0,0,z,0, 0,0,0,1 };
    *RLGL.State.currentMatrix = rlMatrixMultiply(mat, *RLGL.State.currentMatrix);
}

void rlMultMatrixf(const float *matf)
{
    Matrix mat = { matf[0],matf[4],matf[8],matf[12], matf[1],matf[5],matf[9],matf[13], matf[2],matf[6],matf[10],matf[14], matf[3],matf[7],matf[11],matf[15] };
    *RLGL.State.currentMatrix = rlMatrixMultiply(*RLGL.State.currentMatrix, mat);
}

void rlFrustum(double left, double right, double bottom, double top, double znear, double zfar)
{
    Matrix mat = { 0 };
    float rl = (float)(right - left);
    float tb = (float)(top - bottom);
    float fn = (float)(zfar - znear);
    mat.m0 = ((float)znear*2.0f)/rl;
    mat.m5 = ((float)znear*2.0f)/tb;
    mat.m8 = ((float)right+(float)left)/rl;
    mat.m9 = ((float)top+(float)bottom)/tb;
    mat.m10= -((float)zfar+(float)znear)/fn;
    mat.m11= -1.0f;
    mat.m14= -((float)zfar*(float)znear*2.0f)/fn;
    *RLGL.State.currentMatrix = rlMatrixMultiply(*RLGL.State.currentMatrix, mat);
}

void rlOrtho(double left, double right, double bottom, double top, double znear, double zfar)
{
    Matrix mat = { 0 };
    float rl = (float)(right - left);
    float tb = (float)(top - bottom);
    float fn = (float)(zfar - znear);
    mat.m0 = 2.0f/rl;
    mat.m5 = 2.0f/tb;
    mat.m10= -2.0f/fn;
    mat.m12= -((float)left+(float)right)/rl;
    mat.m13= -((float)top+(float)bottom)/tb;
    mat.m14= -((float)zfar+(float)znear)/fn;
    mat.m15= 1.0f;
    *RLGL.State.currentMatrix = rlMatrixMultiply(*RLGL.State.currentMatrix, mat);
}

//----------------------------------------------------------------------------------
// Viewport and state management
//----------------------------------------------------------------------------------
void rlViewport(int x, int y, int width, int height)
{
    if (!RLGL.device) return;
    D3D10_VIEWPORT vp;
    vp.TopLeftX = (INT)x;
    vp.TopLeftY = (INT)y;
    vp.Width = (UINT)width;
    vp.Height = (UINT)height;
    vp.MinDepth = 0.0f;
    vp.MaxDepth = 1.0f;
    ID3D10Device_RSSetViewports(RLGL.device, 1, &vp);

    RLGL.State.framebufferWidth = width;
    RLGL.State.framebufferHeight = height;
}

//----------------------------------------------------------------------------------
// Vertex operations (immediate mode emulation via batching)
//----------------------------------------------------------------------------------
void rlBegin(int mode) {
    if (RLGL.currentBatch->draws[RLGL.currentBatch->drawCounter - 1].mode != mode) {
        if (RLGL.currentBatch->draws[RLGL.currentBatch->drawCounter - 1].vertexCount > 0) {
            if (RLGL.currentBatch->draws[RLGL.currentBatch->drawCounter - 1].mode == RL_LINES)
                RLGL.currentBatch->draws[RLGL.currentBatch->drawCounter - 1].vertexAlignment = ((RLGL.currentBatch->draws[RLGL.currentBatch->drawCounter - 1].vertexCount < 4) ? RLGL.currentBatch->draws[RLGL.currentBatch->drawCounter - 1].vertexCount : RLGL.currentBatch->draws[RLGL.currentBatch->drawCounter - 1].vertexCount%4);
            else if (RLGL.currentBatch->draws[RLGL.currentBatch->drawCounter - 1].mode == RL_TRIANGLES)
                RLGL.currentBatch->draws[RLGL.currentBatch->drawCounter - 1].vertexAlignment = ((RLGL.currentBatch->draws[RLGL.currentBatch->drawCounter - 1].vertexCount < 4) ? 1 : (4 - (RLGL.currentBatch->draws[RLGL.currentBatch->drawCounter - 1].vertexCount%4)));
            else RLGL.currentBatch->draws[RLGL.currentBatch->drawCounter - 1].vertexAlignment = 0;

            if (!rlCheckRenderBatchLimit(RLGL.currentBatch->draws[RLGL.currentBatch->drawCounter - 1].vertexAlignment)) {
                RLGL.currentBatch->vertexBuffer[RLGL.currentBatch->currentBuffer].vCounter += RLGL.currentBatch->draws[RLGL.currentBatch->drawCounter - 1].vertexAlignment;
                RLGL.currentBatch->drawCounter++;
            }
        }

        if (RLGL.currentBatch->drawCounter >= RL_DEFAULT_BATCH_DRAWCALLS) rlDrawRenderBatch(RLGL.currentBatch);

        RLGL.currentBatch->draws[RLGL.currentBatch->drawCounter - 1].mode = mode;
        RLGL.currentBatch->draws[RLGL.currentBatch->drawCounter - 1].vertexCount = 0;
        RLGL.currentBatch->draws[RLGL.currentBatch->drawCounter - 1].textureId = RLGL.State.defaultTextureId;
    }
}

void rlEnd(void) {
    // Nothing to do here; vertex data is already in the batch buffer
}

void rlVertex3f(float x, float y, float z)
{
    float tx = x, ty = y, tz = z;
    if (RLGL.State.transformRequired) {
        tx = RLGL.State.transform.m0*x + RLGL.State.transform.m4*y + RLGL.State.transform.m8*z + RLGL.State.transform.m12;
        ty = RLGL.State.transform.m1*x + RLGL.State.transform.m5*y + RLGL.State.transform.m9*z + RLGL.State.transform.m13;
        tz = RLGL.State.transform.m2*x + RLGL.State.transform.m6*y + RLGL.State.transform.m10*z + RLGL.State.transform.m14;
    }

    if (RLGL.currentBatch->vertexBuffer[RLGL.currentBatch->currentBuffer].vCounter < (RL_DEFAULT_BATCH_BUFFER_ELEMENTS*4)) {
        int idx = RLGL.currentBatch->vertexBuffer[RLGL.currentBatch->currentBuffer].vCounter;
        RLGL.currentBatch->vertexBuffer[RLGL.currentBatch->currentBuffer].vertices[3*idx] = tx;
        RLGL.currentBatch->vertexBuffer[RLGL.currentBatch->currentBuffer].vertices[3*idx + 1] = ty;
        RLGL.currentBatch->vertexBuffer[RLGL.currentBatch->currentBuffer].vertices[3*idx + 2] = tz;

        RLGL.currentBatch->vertexBuffer[RLGL.currentBatch->currentBuffer].vCounter++;
        RLGL.currentBatch->draws[RLGL.currentBatch->drawCounter - 1].vertexCount++;
    } else {
        TRACELOG(RL_LOG_ERROR, "RLGL: Batch elements overflow");
    }
}

void rlVertex2f(float x, float y) { rlVertex3f(x, y, RLGL.currentBatch->currentDepth); }
void rlVertex2i(int x, int y) { rlVertex3f((float)x, (float)y, RLGL.currentBatch->currentDepth); }

void rlTexCoord2f(float x, float y) {
    int idx = RLGL.currentBatch->vertexBuffer[RLGL.currentBatch->currentBuffer].vCounter;
    RLGL.currentBatch->vertexBuffer[RLGL.currentBatch->currentBuffer].texcoords[2*idx] = x;
    RLGL.currentBatch->vertexBuffer[RLGL.currentBatch->currentBuffer].texcoords[2*idx + 1] = y;
}

void rlNormal3f(float x, float y, float z) {
    int idx = RLGL.currentBatch->vertexBuffer[RLGL.currentBatch->currentBuffer].vCounter;
    RLGL.currentBatch->vertexBuffer[RLGL.currentBatch->currentBuffer].normals[3*idx] = x;
    RLGL.currentBatch->vertexBuffer[RLGL.currentBatch->currentBuffer].normals[3*idx + 1] = y;
    RLGL.currentBatch->vertexBuffer[RLGL.currentBatch->currentBuffer].normals[3*idx + 2] = z;
}

void rlColor4ub(unsigned char r, unsigned char g, unsigned char b, unsigned char a) {
    int idx = RLGL.currentBatch->vertexBuffer[RLGL.currentBatch->currentBuffer].vCounter;
    RLGL.currentBatch->vertexBuffer[RLGL.currentBatch->currentBuffer].colors[4*idx] = r;
    RLGL.currentBatch->vertexBuffer[RLGL.currentBatch->currentBuffer].colors[4*idx + 1] = g;
    RLGL.currentBatch->vertexBuffer[RLGL.currentBatch->currentBuffer].colors[4*idx + 2] = b;
    RLGL.currentBatch->vertexBuffer[RLGL.currentBatch->currentBuffer].colors[4*idx + 3] = a;
}

void rlColor4f(float r, float g, float b, float a) { rlColor4ub((unsigned char)(r*255), (unsigned char)(g*255), (unsigned char)(b*255), (unsigned char)(a*255)); }
void rlColor3f(float r, float g, float b) { rlColor4ub((unsigned char)(r*255), (unsigned char)(g*255), (unsigned char)(b*255), 255); }

//----------------------------------------------------------------------------------
// Texture / shader / framebuffer state
//----------------------------------------------------------------------------------
void rlActiveTextureSlot(int slot) { (void)slot; }
void rlEnableTexture(unsigned int id) { if (id > 0 && RLGL.currentBatch->draws[RLGL.currentBatch->drawCounter - 1].textureId != id) { if (RLGL.currentBatch->draws[RLGL.currentBatch->drawCounter - 1].vertexCount > 0) { if (RLGL.currentBatch->draws[RLGL.currentBatch->drawCounter - 1].mode == RL_LINES) RLGL.currentBatch->draws[RLGL.currentBatch->drawCounter - 1].vertexAlignment = ((RLGL.currentBatch->draws[RLGL.currentBatch->drawCounter - 1].vertexCount < 4) ? RLGL.currentBatch->draws[RLGL.currentBatch->drawCounter - 1].vertexCount : RLGL.currentBatch->draws[RLGL.currentBatch->drawCounter - 1].vertexCount%4); else if (RLGL.currentBatch->draws[RLGL.currentBatch->drawCounter - 1].mode == RL_TRIANGLES) RLGL.currentBatch->draws[RLGL.currentBatch->drawCounter - 1].vertexAlignment = ((RLGL.currentBatch->draws[RLGL.currentBatch->drawCounter - 1].vertexCount < 4) ? 1 : (4 - (RLGL.currentBatch->draws[RLGL.currentBatch->drawCounter - 1].vertexCount%4))); else RLGL.currentBatch->draws[RLGL.currentBatch->drawCounter - 1].vertexAlignment = 0; if (!rlCheckRenderBatchLimit(RLGL.currentBatch->draws[RLGL.currentBatch->drawCounter - 1].vertexAlignment)) { RLGL.currentBatch->vertexBuffer[RLGL.currentBatch->currentBuffer].vCounter += RLGL.currentBatch->draws[RLGL.currentBatch->drawCounter - 1].vertexAlignment; RLGL.currentBatch->drawCounter++; } } if (RLGL.currentBatch->drawCounter >= RL_DEFAULT_BATCH_DRAWCALLS) rlDrawRenderBatch(RLGL.currentBatch); RLGL.currentBatch->draws[RLGL.currentBatch->drawCounter - 1].textureId = id; RLGL.currentBatch->draws[RLGL.currentBatch->drawCounter - 1].vertexCount = 0; } }
void rlDisableTexture(void) { if (RLGL.State.defaultTextureId > 0) rlEnableTexture(RLGL.State.defaultTextureId); }
void rlEnableTextureCubemap(unsigned int id) { (void)id; }
void rlDisableTextureCubemap(void) { }
void rlTextureParameters(unsigned int id, int param, int value) { (void)id; (void)param; (void)value; }
void rlCubemapParameters(unsigned int id, int param, int value) { (void)id; (void)param; (void)value; }

void rlEnableShader(unsigned int id) { (void)id; }
void rlDisableShader(void) { }

void rlEnableFramebuffer(unsigned int id) {
    if (id > 0 && id <= RLGL.framebufferCount && RLGL.device) {
        unsigned int idx = id - 1;
        ID3D10Device_OMSetRenderTargets(RLGL.device, 1,
            &RLGL.framebuffers[idx].rtv, RLGL.framebuffers[idx].dsv);
    }
}
void rlDisableFramebuffer(void) { }
unsigned int rlGetActiveFramebuffer(void) { return 0; }

//----------------------------------------------------------------------------------
// Render state management
//----------------------------------------------------------------------------------
void rlEnableColorBlend(void) { RLGL.State.colorBlendEnabled = true; RLGL.stateDirty = true; rlUpdateD3D10States(); }
void rlDisableColorBlend(void) { RLGL.State.colorBlendEnabled = false; RLGL.stateDirty = true; rlUpdateD3D10States(); }
void rlEnableDepthTest(void) { RLGL.State.depthTestEnabled = true; RLGL.stateDirty = true; rlUpdateD3D10States(); }
void rlDisableDepthTest(void) { RLGL.State.depthTestEnabled = false; RLGL.stateDirty = true; rlUpdateD3D10States(); }
void rlEnableDepthMask(void) { RLGL.State.depthWriteEnabled = true; RLGL.stateDirty = true; rlUpdateD3D10States(); }
void rlDisableDepthMask(void) { RLGL.State.depthWriteEnabled = false; RLGL.stateDirty = true; rlUpdateD3D10States(); }
void rlEnableBackfaceCulling(void) { RLGL.State.backfaceCullingEnabled = true; RLGL.stateDirty = true; rlUpdateD3D10States(); }
void rlDisableBackfaceCulling(void) { RLGL.State.backfaceCullingEnabled = false; RLGL.stateDirty = true; rlUpdateD3D10States(); }
void rlColorMask(bool r, bool g, bool b, bool a) { (void)r; (void)g; (void)b; (void)a; }
void rlSetCullFace(int mode) { (void)mode; }
void rlEnableScissorTest(void) { RLGL.State.scissorTest = true; RLGL.stateDirty = true; rlUpdateD3D10States(); }
void rlDisableScissorTest(void) { RLGL.State.scissorTest = false; RLGL.stateDirty = true; rlUpdateD3D10States(); }
void rlScissor(int x, int y, int width, int height) {
    RECT rect = { x, y, x + width, y + height };
    if (RLGL.device) ID3D10Device_RSSetScissorRects(RLGL.device, 1, &rect);
}
void rlEnableWireMode(void) { }
void rlEnablePointMode(void) { }
void rlDisableWireMode(void) { }
void rlSetLineWidth(float width) { RLGL.State.lineWidth = width; }
float rlGetLineWidth(void) { return RLGL.State.lineWidth; }
void rlEnableSmoothLines(void) { }
void rlDisableSmoothLines(void) { }
void rlEnableStereoRender(void) { RLGL.State.stereoRender = true; }
void rlDisableStereoRender(void) { RLGL.State.stereoRender = false; }
bool rlIsStereoRenderEnabled(void) { return RLGL.State.stereoRender; }

void rlClearColor(unsigned char r, unsigned char g, unsigned char b, unsigned char a) {
    RLGL.State.clearColor[0] = r/255.0f; RLGL.State.clearColor[1] = g/255.0f;
    RLGL.State.clearColor[2] = b/255.0f; RLGL.State.clearColor[3] = a/255.0f;
}
void rlClearScreenBuffers(void) { /* Handled externally via DXGI */ }
void rlCheckErrors(void) { }
void rlSetBlendMode(int mode) { RLGL.State.currentBlendMode = mode; }
void rlSetBlendFactors(int glSrcFactor, int glDstFactor, int glEquation) { (void)glSrcFactor; (void)glDstFactor; (void)glEquation; }
void rlSetBlendFactorsSeparate(int glSrcRGB, int glDstRGB, int glSrcAlpha, int glDstAlpha, int glEqRGB, int glEqAlpha) { (void)glSrcRGB; (void)glDstRGB; (void)glSrcAlpha; (void)glDstAlpha; (void)glEqRGB; (void)glEqAlpha; }

//----------------------------------------------------------------------------------
// rlgl initialization and cleanup
//----------------------------------------------------------------------------------
void rlglInit(int width, int height)
{
    memset(&RLGL, 0, sizeof(rlglData));

    // ---- Create D3D10 device ----
    UINT createFlags = 0;
#ifndef NDEBUG
    createFlags |= D3D10_CREATE_DEVICE_DEBUG;
#endif

    HRESULT hr = D3D10CreateDevice(NULL, D3D10_DRIVER_TYPE_HARDWARE, NULL,
        createFlags, D3D10_SDK_VERSION, &RLGL.device);

    if (FAILED(hr)) {
        TRACELOG(RL_LOG_WARNING, "DISPLAY: D3D10 hardware device creation failed, trying WARP");
        hr = D3D10CreateDevice(NULL, D3D10_DRIVER_TYPE_WARP, NULL,
            createFlags, D3D10_SDK_VERSION, &RLGL.device);
    }

    if (FAILED(hr) || !RLGL.device) {
        TRACELOG(RL_LOG_FATAL, "DISPLAY: Failed to create D3D10 device (HRESULT: 0x%08X)", (unsigned int)hr);
        return;
    }

    TRACELOG(RL_LOG_INFO, "DISPLAY: D3D10 device created successfully");

    // ---- Create default sampler state ----
    D3D10_SAMPLER_DESC sampDesc;
    memset(&sampDesc, 0, sizeof(sampDesc));
    sampDesc.Filter = D3D10_FILTER_MIN_MAG_MIP_POINT;
    sampDesc.AddressU = D3D10_TEXTURE_ADDRESS_CLAMP;
    sampDesc.AddressV = D3D10_TEXTURE_ADDRESS_CLAMP;
    sampDesc.AddressW = D3D10_TEXTURE_ADDRESS_CLAMP;
    sampDesc.ComparisonFunc = D3D10_COMPARISON_NEVER;
    sampDesc.MinLOD = 0;
    sampDesc.MaxLOD = D3D10_FLOAT32_MAX;
    ID3D10Device_CreateSamplerState(RLGL.device, &sampDesc, &RLGL.defaultSampler);

    // ---- Create default 1x1 white texture ----
    unsigned char whitePixel[4] = { 255, 255, 255, 255 };
    D3D10_TEXTURE2D_DESC texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.Width = 1;
    texDesc.Height = 1;
    texDesc.MipLevels = 1;
    texDesc.ArraySize = 1;
    texDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    texDesc.SampleDesc.Count = 1;
    texDesc.Usage = D3D10_USAGE_DEFAULT;
    texDesc.BindFlags = D3D10_BIND_SHADER_RESOURCE;

    D3D10_SUBRESOURCE_DATA texData;
    memset(&texData, 0, sizeof(texData));
    texData.pSysMem = whitePixel;
    texData.SysMemPitch = 4;

    ID3D10Texture2D *defaultTex = NULL;
    ID3D10Device_CreateTexture2D(RLGL.device, &texDesc, &texData, &defaultTex);
    if (defaultTex) {
        unsigned int idx = RLGL.textureCount;
        RLGL.textures[idx].texture = defaultTex;
        RLGL.textures[idx].width = 1;
        RLGL.textures[idx].height = 1;
        RLGL.textures[idx].mipmaps = 1;
        RLGL.textures[idx].format = 7; // RGBA8

        D3D10_SHADER_RESOURCE_VIEW_DESC srvDesc;
        memset(&srvDesc, 0, sizeof(srvDesc));
        srvDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        srvDesc.ViewDimension = D3D10_SRV_DIMENSION_TEXTURE2D;
        srvDesc.Texture2D.MipLevels = 1;
        ID3D10Device_CreateShaderResourceView(RLGL.device, (ID3D10Resource *)defaultTex, &srvDesc, &RLGL.textures[idx].srv);

        RLGL.textureCount++;
        RLGL.State.defaultTextureId = idx + 1;
        TRACELOG(RL_LOG_INFO, "TEXTURE: [ID %i] Default texture loaded successfully", RLGL.State.defaultTextureId);
    }

    // ---- Create constant buffers ----
    RLGL.cbMatrix = rlCreateD3D10Buffer(sizeof(rlD3D10MatrixCB), D3D10_BIND_CONSTANT_BUFFER, NULL);
    RLGL.cbColor = rlCreateD3D10Buffer(sizeof(rlD3D10ColorCB), D3D10_BIND_CONSTANT_BUFFER, NULL);

    // ---- Load default shader ----
    rlLoadShaderDefault();

    RLGL.currentBatch = &RLGL.defaultBatch;
    RLGL.defaultBatch = rlLoadRenderBatch(1, RL_DEFAULT_BATCH_BUFFER_ELEMENTS);
    RLGL.currentBatch = &RLGL.defaultBatch;

    RLGL.State.activeTextureId = RLGL.State.defaultTextureId;
    RLGL.State.currentShaderId = RLGL.State.defaultShaderId;
    RLGL.State.currentShaderLocs = RLGL.State.defaultShaderLocs;

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
    rlUpdateD3D10States();

    TRACELOG(RL_LOG_INFO, "RLGL: Direct3D 10 backend initialized successfully");
}

void rlglClose(void)
{
    rlUnloadRenderBatch(RLGL.defaultBatch);
    rlUnloadShaderDefault();

    if (RLGL.State.defaultTextureId > 0) rlUnloadTexture(RLGL.State.defaultTextureId);

    // Release all tracked resources
    for (unsigned int i = 0; i < RLGL.textureCount; i++) {
        if (RLGL.textures[i].srv) ID3D10ShaderResourceView_Release(RLGL.textures[i].srv);
        if (RLGL.textures[i].texture) ID3D10Texture2D_Release(RLGL.textures[i].texture);
        if (RLGL.textures[i].sampler) ID3D10SamplerState_Release(RLGL.textures[i].sampler);
        if (RLGL.textures[i].dsv) ID3D10DepthStencilView_Release(RLGL.textures[i].dsv);
        if (RLGL.textures[i].rtv) ID3D10RenderTargetView_Release(RLGL.textures[i].rtv);
    }
    for (unsigned int i = 0; i < RLGL.shaderCount; i++) {
        if (RLGL.shaders[i].vertexShader) ID3D10VertexShader_Release(RLGL.shaders[i].vertexShader);
        if (RLGL.shaders[i].pixelShader) ID3D10PixelShader_Release(RLGL.shaders[i].pixelShader);
        if (RLGL.shaders[i].inputLayout) ID3D10InputLayout_Release(RLGL.shaders[i].inputLayout);
        if (RLGL.shaders[i].vsBlob) ID3D10Blob_Release(RLGL.shaders[i].vsBlob);
        if (RLGL.shaders[i].psBlob) ID3D10Blob_Release(RLGL.shaders[i].psBlob);
    }
    for (unsigned int i = 0; i < RLGL.bufferCount; i++) {
        if (RLGL.buffers[i].buffer) ID3D10Buffer_Release(RLGL.buffers[i].buffer);
    }

    if (RLGL.currentBlendState) ID3D10BlendState_Release(RLGL.currentBlendState);
    if (RLGL.currentDSState) ID3D10DepthStencilState_Release(RLGL.currentDSState);
    if (RLGL.currentRSState) ID3D10RasterizerState_Release(RLGL.currentRSState);
    if (RLGL.cbMatrix) ID3D10Buffer_Release(RLGL.cbMatrix);
    if (RLGL.cbColor) ID3D10Buffer_Release(RLGL.cbColor);
    if (RLGL.defaultSampler) ID3D10SamplerState_Release(RLGL.defaultSampler);

    for (int i = 0; i < 4; i++) {
        if (RLGL.batchVertexBuffers[i]) ID3D10Buffer_Release(RLGL.batchVertexBuffers[i]);
    }
    if (RLGL.batchIndexBuffer) ID3D10Buffer_Release(RLGL.batchIndexBuffer);

    // No separate context in D3D10 - release device directly
    if (RLGL.device) { ID3D10Device_Release(RLGL.device); RLGL.device = NULL; }

    TRACELOG(RL_LOG_INFO, "RLGL: Direct3D 10 backend closed successfully");
}

void rlLoadExtensions(void *loader)
{
    (void)loader;
    // D3D10 capabilities (no feature levels, always 10.0)
    RLGL.ExtSupported.computeShader = false;  // No compute shaders in D3D10
    RLGL.ExtSupported.ssbo = false;           // No SSBOs in D3D10
    RLGL.ExtSupported.maxAnisotropyLevel = 16.0f;
    RLGL.ExtSupported.maxDepthBits = 32;

    TRACELOG(RL_LOG_INFO, "D3D10: Device capabilities loaded");
}

int rlGetVersion(void) { return RL_DIRECT3D_10; }

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
    batch.bufferCount = numBuffers;
    batch.currentBuffer = 0;

    batch.vertexBuffer = (rlVertexBuffer *)RL_MALLOC(numBuffers * sizeof(rlVertexBuffer));
    for (int i = 0; i < numBuffers; i++) {
        batch.vertexBuffer[i].elementCount = bufferElements;
        batch.vertexBuffer[i].vertices  = (float *)RL_MALLOC(bufferElements * 4 * 3 * sizeof(float));
        batch.vertexBuffer[i].texcoords = (float *)RL_MALLOC(bufferElements * 4 * 2 * sizeof(float));
        batch.vertexBuffer[i].normals   = (float *)RL_MALLOC(bufferElements * 4 * 3 * sizeof(float));
        batch.vertexBuffer[i].colors    = (unsigned char *)RL_MALLOC(bufferElements * 4 * 4 * sizeof(unsigned char));

        memset(batch.vertexBuffer[i].vertices,  0, bufferElements * 4 * 3 * sizeof(float));
        memset(batch.vertexBuffer[i].texcoords, 0, bufferElements * 4 * 2 * sizeof(float));
        memset(batch.vertexBuffer[i].normals,   0, bufferElements * 4 * 3 * sizeof(float));
        memset(batch.vertexBuffer[i].colors,    0, bufferElements * 4 * 4 * sizeof(unsigned char));

        // Build index buffer for quads
        int indexCount = bufferElements * 6;
        batch.vertexBuffer[i].indices = (unsigned short *)RL_MALLOC(indexCount * sizeof(unsigned short));
        for (int q = 0, k = 0; q < bufferElements; q++, k += 6) {
            batch.vertexBuffer[i].indices[k+0] = (unsigned short)(4*q);
            batch.vertexBuffer[i].indices[k+1] = (unsigned short)(4*q+1);
            batch.vertexBuffer[i].indices[k+2] = (unsigned short)(4*q+2);
            batch.vertexBuffer[i].indices[k+3] = (unsigned short)(4*q);
            batch.vertexBuffer[i].indices[k+4] = (unsigned short)(4*q+2);
            batch.vertexBuffer[i].indices[k+5] = (unsigned short)(4*q+3);
        }

        batch.vertexBuffer[i].vCounter = 0;
        batch.vertexBuffer[i].tcCounter = 0;
        batch.vertexBuffer[i].ncCounter = 0;
        batch.vertexBuffer[i].cCounter = 0;
    }

    // Create GPU buffers for batching
    if (RLGL.device) {
        unsigned int vbSize = bufferElements * 4;
        RLGL.batchVertexBuffers[0] = rlCreateD3D10Buffer(vbSize * 3 * sizeof(float),           D3D10_BIND_VERTEX_BUFFER, NULL);
        RLGL.batchVertexBuffers[1] = rlCreateD3D10Buffer(vbSize * 2 * sizeof(float),           D3D10_BIND_VERTEX_BUFFER, NULL);
        RLGL.batchVertexBuffers[2] = rlCreateD3D10Buffer(vbSize * 3 * sizeof(float),           D3D10_BIND_VERTEX_BUFFER, NULL);
        RLGL.batchVertexBuffers[3] = rlCreateD3D10Buffer(vbSize * 4 * sizeof(unsigned char),   D3D10_BIND_VERTEX_BUFFER, NULL);

        // Create index buffer (immutable)
        {
            D3D10_BUFFER_DESC ibd;
            memset(&ibd, 0, sizeof(ibd));
            ibd.ByteWidth = bufferElements * 6 * sizeof(unsigned short);
            ibd.Usage = D3D10_USAGE_DEFAULT;
            ibd.BindFlags = D3D10_BIND_INDEX_BUFFER;
            D3D10_SUBRESOURCE_DATA isd;
            memset(&isd, 0, sizeof(isd));
            isd.pSysMem = batch.vertexBuffer[0].indices;
            ID3D10Device_CreateBuffer(RLGL.device, &ibd, &isd, &RLGL.batchIndexBuffer);
        }
    }

    batch.draws = (rlDrawCall *)RL_MALLOC(RL_DEFAULT_BATCH_DRAWCALLS * sizeof(rlDrawCall));
    for (int i = 0; i < RL_DEFAULT_BATCH_DRAWCALLS; i++) {
        batch.draws[i].mode = RL_QUADS;
        batch.draws[i].vertexCount = 0;
        batch.draws[i].vertexAlignment = 0;
        batch.draws[i].textureId = RLGL.State.defaultTextureId;
    }
    batch.drawCounter = 1;
    batch.currentDepth = -1.0f;

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

void rlSetRenderBatchActive(rlRenderBatch *batch) {
    rlDrawRenderBatch(RLGL.currentBatch);
    RLGL.currentBatch = (batch != NULL) ? batch : &RLGL.defaultBatch;
}

void rlDrawRenderBatchActive(void) {
    rlDrawRenderBatch(RLGL.currentBatch);
}

bool rlCheckRenderBatchLimit(int vCount)
{
    bool overflow = false;
    if ((RLGL.currentBatch->vertexBuffer[RLGL.currentBatch->currentBuffer].vCounter + vCount) >=
        (RLGL.currentBatch->vertexBuffer[RLGL.currentBatch->currentBuffer].elementCount * 4)) {
        overflow = true;
        int currentMode = RLGL.currentBatch->draws[RLGL.currentBatch->drawCounter - 1].mode;
        unsigned int currentTexture = RLGL.currentBatch->draws[RLGL.currentBatch->drawCounter - 1].textureId;
        rlDrawRenderBatch(RLGL.currentBatch);
        RLGL.currentBatch->draws[RLGL.currentBatch->drawCounter - 1].mode = currentMode;
        RLGL.currentBatch->draws[RLGL.currentBatch->drawCounter - 1].textureId = currentTexture;
    }
    return overflow;
}

//----------------------------------------------------------------------------------
// Draw render batch (upload vertices and issue draw calls)
//----------------------------------------------------------------------------------
void rlDrawRenderBatch(rlRenderBatch *batch)
{
    if (batch->vertexBuffer[batch->currentBuffer].vCounter == 0) return;
    if (!RLGL.device) return;

    int vCount = batch->vertexBuffer[batch->currentBuffer].vCounter;

    // Upload vertex data to GPU via Map/Unmap (D3D10 maps on the resource directly)
    {
        void *pData = NULL;

        // Positions
        if (RLGL.batchVertexBuffers[0]) {
            HRESULT hr = ID3D10Buffer_Map(RLGL.batchVertexBuffers[0], D3D10_MAP_WRITE_DISCARD, 0, &pData);
            if (SUCCEEDED(hr)) {
                memcpy(pData, batch->vertexBuffer[batch->currentBuffer].vertices, vCount * 3 * sizeof(float));
                ID3D10Buffer_Unmap(RLGL.batchVertexBuffers[0]);
            }
        }
        // Texcoords
        if (RLGL.batchVertexBuffers[1]) {
            HRESULT hr = ID3D10Buffer_Map(RLGL.batchVertexBuffers[1], D3D10_MAP_WRITE_DISCARD, 0, &pData);
            if (SUCCEEDED(hr)) {
                memcpy(pData, batch->vertexBuffer[batch->currentBuffer].texcoords, vCount * 2 * sizeof(float));
                ID3D10Buffer_Unmap(RLGL.batchVertexBuffers[1]);
            }
        }
        // Normals
        if (RLGL.batchVertexBuffers[2]) {
            HRESULT hr = ID3D10Buffer_Map(RLGL.batchVertexBuffers[2], D3D10_MAP_WRITE_DISCARD, 0, &pData);
            if (SUCCEEDED(hr)) {
                memcpy(pData, batch->vertexBuffer[batch->currentBuffer].normals, vCount * 3 * sizeof(float));
                ID3D10Buffer_Unmap(RLGL.batchVertexBuffers[2]);
            }
        }
        // Colors
        if (RLGL.batchVertexBuffers[3]) {
            HRESULT hr = ID3D10Buffer_Map(RLGL.batchVertexBuffers[3], D3D10_MAP_WRITE_DISCARD, 0, &pData);
            if (SUCCEEDED(hr)) {
                memcpy(pData, batch->vertexBuffer[batch->currentBuffer].colors, vCount * 4 * sizeof(unsigned char));
                ID3D10Buffer_Unmap(RLGL.batchVertexBuffers[3]);
            }
        }
    }

    // Set vertex buffers
    {
        UINT strides[4] = { 3*sizeof(float), 2*sizeof(float), 3*sizeof(float), 4*sizeof(unsigned char) };
        UINT offsets[4] = { 0, 0, 0, 0 };
        ID3D10Buffer *vbs[4] = {
            RLGL.batchVertexBuffers[0], RLGL.batchVertexBuffers[1],
            RLGL.batchVertexBuffers[2], RLGL.batchVertexBuffers[3]
        };
        ID3D10Device_IASetVertexBuffers(RLGL.device, 0, 4, vbs, strides, offsets);
    }

    // Set index buffer
    ID3D10Device_IASetIndexBuffer(RLGL.device, RLGL.batchIndexBuffer, DXGI_FORMAT_R16_UINT, 0);

    // Set shader state
    unsigned int shaderIdx = RLGL.State.currentShaderId > 0 ? RLGL.State.currentShaderId - 1 : 0;
    if (shaderIdx < RLGL.shaderCount) {
        if (RLGL.shaders[shaderIdx].inputLayout)
            ID3D10Device_IASetInputLayout(RLGL.device, RLGL.shaders[shaderIdx].inputLayout);
        if (RLGL.shaders[shaderIdx].vertexShader)
            ID3D10Device_VSSetShader(RLGL.device, RLGL.shaders[shaderIdx].vertexShader);
        if (RLGL.shaders[shaderIdx].pixelShader)
            ID3D10Device_PSSetShader(RLGL.device, RLGL.shaders[shaderIdx].pixelShader);
    }

    // Update MVP constant buffer
    {
        Matrix matMVP = rlMatrixMultiply(RLGL.State.modelview, RLGL.State.projection);
        rl_float16 mvpFloats = rlMatrixToFloatV(matMVP);
        void *pData = NULL;
        if (RLGL.cbMatrix) {
            HRESULT hr = ID3D10Buffer_Map(RLGL.cbMatrix, D3D10_MAP_WRITE_DISCARD, 0, &pData);
            if (SUCCEEDED(hr)) {
                memcpy(pData, mvpFloats.v, sizeof(float)*16);
                ID3D10Buffer_Unmap(RLGL.cbMatrix);
            }
        }
        ID3D10Device_VSSetConstantBuffers(RLGL.device, 0, 1, &RLGL.cbMatrix);
    }

    // Update color constant buffer (default white)
    {
        float white[4] = { 1.0f, 1.0f, 1.0f, 1.0f };
        void *pData = NULL;
        if (RLGL.cbColor) {
            HRESULT hr = ID3D10Buffer_Map(RLGL.cbColor, D3D10_MAP_WRITE_DISCARD, 0, &pData);
            if (SUCCEEDED(hr)) {
                memcpy(pData, white, sizeof(float)*4);
                ID3D10Buffer_Unmap(RLGL.cbColor);
            }
        }
        ID3D10Device_PSSetConstantBuffers(RLGL.device, 1, 1, &RLGL.cbColor);
    }

    // Set default sampler
    ID3D10Device_PSSetSamplers(RLGL.device, 0, 1, &RLGL.defaultSampler);

    // Ensure render states are up to date
    if (RLGL.stateDirty) rlUpdateD3D10States();

    // Issue draw calls
    int vertexOffset = 0;
    for (int i = 0; i < batch->drawCounter; i++) {
        int drawVertexCount = batch->draws[i].vertexCount;
        if (drawVertexCount == 0) { vertexOffset += batch->draws[i].vertexAlignment; continue; }

        // Bind texture SRV for this draw call
        unsigned int texId = batch->draws[i].textureId;
        if (texId > 0 && texId <= RLGL.textureCount) {
            unsigned int texIdx = texId - 1;
            if (RLGL.textures[texIdx].srv) {
                ID3D10Device_PSSetShaderResources(RLGL.device, 0, 1, &RLGL.textures[texIdx].srv);
            }
        }

        int mode = batch->draws[i].mode;
        if (mode == RL_QUADS) {
            ID3D10Device_IASetPrimitiveTopology(RLGL.device, D3D10_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            int quadCount = drawVertexCount / 4;
            int indexOffset = (vertexOffset / 4) * 6;
            ID3D10Device_DrawIndexed(RLGL.device, quadCount * 6, indexOffset, 0);
        } else if (mode == RL_TRIANGLES) {
            ID3D10Device_IASetPrimitiveTopology(RLGL.device, D3D10_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            ID3D10Device_Draw(RLGL.device, drawVertexCount, vertexOffset);
        } else if (mode == RL_LINES) {
            ID3D10Device_IASetPrimitiveTopology(RLGL.device, D3D10_PRIMITIVE_TOPOLOGY_LINELIST);
            ID3D10Device_Draw(RLGL.device, drawVertexCount, vertexOffset);
        }

        vertexOffset += drawVertexCount + batch->draws[i].vertexAlignment;
    }

    // Reset batch
    batch->vertexBuffer[batch->currentBuffer].vCounter = 0;
    batch->vertexBuffer[batch->currentBuffer].tcCounter = 0;
    batch->vertexBuffer[batch->currentBuffer].ncCounter = 0;
    batch->vertexBuffer[batch->currentBuffer].cCounter = 0;

    for (int i = 0; i < RL_DEFAULT_BATCH_DRAWCALLS; i++) {
        batch->draws[i].mode = RL_QUADS;
        batch->draws[i].vertexCount = 0;
        batch->draws[i].vertexAlignment = 0;
        batch->draws[i].textureId = RLGL.State.defaultTextureId;
    }
    batch->drawCounter = 1;
    batch->currentDepth = -1.0f;

    RLGL.State.modelview = rlMatrixIdentity();
    RLGL.State.currentMatrix = &RLGL.State.modelview;
}

//----------------------------------------------------------------------------------
// Texture management
//----------------------------------------------------------------------------------
unsigned int rlLoadTexture(const void *data, int width, int height, int format, int mipmapCount)
{
    if (!RLGL.device) return 0;
    if (RLGL.textureCount >= RL_D3D10_MAX_TEXTURES) {
        TRACELOG(RL_LOG_WARNING, "TEXTURE: Max D3D10 texture limit reached (%d)", RL_D3D10_MAX_TEXTURES);
        return 0;
    }

    // Determine bits-per-pixel for row pitch calculation
    int bpp = 0;
    switch (format) {
        case 1: bpp = 8; break;
        case 2: case 3: case 5: case 6: bpp = 16; break;
        case 7: bpp = 32; break;
        case 4: bpp = 24; break;     // RGB8 (will be expanded to RGBA8)
        case 8: bpp = 32; break;
        case 9: bpp = 32*3; break;   // RGB32F (will be expanded to RGBA32F)
        case 10: bpp = 32*4; break;
        case 11: bpp = 16; break;
        case 12: bpp = 16*3; break;  // RGB16F (will be expanded to RGBA16F)
        case 13: bpp = 16*4; break;
        case 14: case 15: case 18: case 19: case 21: case 22: bpp = 4; break;
        case 16: case 17: case 20: case 23: bpp = 8; break;
        case 24: bpp = 2; break;
        default: break;
    }

    // Handle 3-channel formats by expanding to 4-channel
    const void *uploadData = data;
    void *expandedData = NULL;

    if (data != NULL) {
        if (format == 4) {
            // RGB8 -> RGBA8
            int pixelCount = width * height;
            expandedData = RL_MALLOC(pixelCount * 4);
            const unsigned char *src = (const unsigned char *)data;
            unsigned char *dst = (unsigned char *)expandedData;
            for (int i = 0; i < pixelCount; i++) {
                dst[4*i+0] = src[3*i+0];
                dst[4*i+1] = src[3*i+1];
                dst[4*i+2] = src[3*i+2];
                dst[4*i+3] = 255;
            }
            uploadData = expandedData;
            bpp = 32;
        }
        else if (format == 9) {
            // RGB32F -> RGBA32F
            int pixelCount = width * height;
            expandedData = RL_MALLOC(pixelCount * 16);
            const float *src = (const float *)data;
            float *dst = (float *)expandedData;
            for (int i = 0; i < pixelCount; i++) {
                dst[4*i+0] = src[3*i+0];
                dst[4*i+1] = src[3*i+1];
                dst[4*i+2] = src[3*i+2];
                dst[4*i+3] = 1.0f;
            }
            uploadData = expandedData;
            bpp = 32*4;
        }
        else if (format == 12) {
            // RGB16F -> RGBA16F
            int pixelCount = width * height;
            expandedData = RL_MALLOC(pixelCount * 8);
            const unsigned short *src = (const unsigned short *)data;
            unsigned short *dst = (unsigned short *)expandedData;
            for (int i = 0; i < pixelCount; i++) {
                dst[4*i+0] = src[3*i+0];
                dst[4*i+1] = src[3*i+1];
                dst[4*i+2] = src[3*i+2];
                dst[4*i+3] = 0x3C00; // 1.0 in half-float
            }
            uploadData = expandedData;
            bpp = 16*4;
        }
    }

    DXGI_FORMAT dxgiFormat = rlGetDXGIFormat(format);

    D3D10_TEXTURE2D_DESC desc;
    memset(&desc, 0, sizeof(desc));
    desc.Width = width;
    desc.Height = height;
    desc.MipLevels = (mipmapCount > 0) ? mipmapCount : 1;
    desc.ArraySize = 1;
    desc.Format = dxgiFormat;
    desc.SampleDesc.Count = 1;
    desc.Usage = D3D10_USAGE_DEFAULT;
    desc.BindFlags = D3D10_BIND_SHADER_RESOURCE;

    D3D10_SUBRESOURCE_DATA srd;
    memset(&srd, 0, sizeof(srd));
    srd.pSysMem = uploadData;
    srd.SysMemPitch = (bpp > 0) ? (width * bpp / 8) : (width * 4);

    ID3D10Texture2D *texture = NULL;
    HRESULT hr = ID3D10Device_CreateTexture2D(RLGL.device, &desc, uploadData ? &srd : NULL, &texture);

    if (expandedData) RL_FREE(expandedData);

    if (FAILED(hr) || !texture) {
        TRACELOG(RL_LOG_WARNING, "TEXTURE: Failed to create D3D10 texture (format %d, %dx%d, HRESULT: 0x%08X)", format, width, height, (unsigned int)hr);
        return 0;
    }

    // Create SRV
    D3D10_SHADER_RESOURCE_VIEW_DESC srvDesc;
    memset(&srvDesc, 0, sizeof(srvDesc));
    srvDesc.Format = dxgiFormat;
    srvDesc.ViewDimension = D3D10_SRV_DIMENSION_TEXTURE2D;
    srvDesc.Texture2D.MostDetailedMip = 0;
    srvDesc.Texture2D.MipLevels = (mipmapCount > 0) ? mipmapCount : 1;

    ID3D10ShaderResourceView *srv = NULL;
    ID3D10Device_CreateShaderResourceView(RLGL.device, (ID3D10Resource *)texture, &srvDesc, &srv);

    unsigned int idx = RLGL.textureCount;
    RLGL.textures[idx].texture = texture;
    RLGL.textures[idx].srv = srv;
    RLGL.textures[idx].sampler = NULL;
    RLGL.textures[idx].dsv = NULL;
    RLGL.textures[idx].rtv = NULL;
    RLGL.textures[idx].width = width;
    RLGL.textures[idx].height = height;
    RLGL.textures[idx].mipmaps = (mipmapCount > 0) ? mipmapCount : 1;
    RLGL.textures[idx].format = format;
    RLGL.textureCount++;

    unsigned int id = idx + 1;
    TRACELOG(RL_LOG_INFO, "TEXTURE: [ID %i] D3D10 texture loaded (%ix%i, format: %i)", id, width, height, format);
    return id;
}

unsigned int rlLoadTextureDepth(int width, int height, bool useRenderBuffer)
{
    (void)useRenderBuffer;
    if (!RLGL.device || RLGL.textureCount >= RL_D3D10_MAX_TEXTURES) return 0;

    DXGI_FORMAT format = DXGI_FORMAT_R32_TYPELESS;

    D3D10_TEXTURE2D_DESC desc;
    memset(&desc, 0, sizeof(desc));
    desc.Width = width;
    desc.Height = height;
    desc.MipLevels = 1;
    desc.ArraySize = 1;
    desc.Format = format;
    desc.SampleDesc.Count = 1;
    desc.Usage = D3D10_USAGE_DEFAULT;
    desc.BindFlags = D3D10_BIND_DEPTH_STENCIL | D3D10_BIND_SHADER_RESOURCE;

    ID3D10Texture2D *texture = NULL;
    HRESULT hr = ID3D10Device_CreateTexture2D(RLGL.device, &desc, NULL, &texture);
    if (FAILED(hr) || !texture) return 0;

    // Create depth stencil view
    D3D10_DEPTH_STENCIL_VIEW_DESC dsvDesc;
    memset(&dsvDesc, 0, sizeof(dsvDesc));
    dsvDesc.Format = (format == DXGI_FORMAT_R32_TYPELESS) ? DXGI_FORMAT_D32_FLOAT : DXGI_FORMAT_D24_UNORM_S8_UINT;
    dsvDesc.ViewDimension = D3D10_DSV_DIMENSION_TEXTURE2D;
    dsvDesc.Texture2D.MipSlice = 0;

    ID3D10DepthStencilView *dsv = NULL;
    ID3D10Device_CreateDepthStencilView(RLGL.device, (ID3D10Resource *)texture, &dsvDesc, &dsv);

    // Create SRV for reading depth
    D3D10_SHADER_RESOURCE_VIEW_DESC srvDesc;
    memset(&srvDesc, 0, sizeof(srvDesc));
    srvDesc.Format = DXGI_FORMAT_R32_FLOAT;
    srvDesc.ViewDimension = D3D10_SRV_DIMENSION_TEXTURE2D;
    srvDesc.Texture2D.MipLevels = 1;

    ID3D10ShaderResourceView *srv = NULL;
    ID3D10Device_CreateShaderResourceView(RLGL.device, (ID3D10Resource *)texture, &srvDesc, &srv);

    unsigned int idx = RLGL.textureCount;
    RLGL.textures[idx].texture = texture;
    RLGL.textures[idx].srv = srv;
    RLGL.textures[idx].dsv = dsv;
    RLGL.textures[idx].width = width;
    RLGL.textures[idx].height = height;
    RLGL.textures[idx].mipmaps = 1;
    RLGL.textureCount++;

    return idx + 1;
}

unsigned int rlLoadTextureCubemap(const void *data, int size, int format)
{
    (void)data; (void)size; (void)format;
    TRACELOG(RL_LOG_WARNING, "TEXTURE: Cubemap textures not yet implemented for D3D10");
    return 0;
}

void rlUpdateTexture(unsigned int id, int offsetX, int offsetY, int width, int height, int format, const void *pixels)
{
    if (id == 0 || id > RLGL.textureCount || !RLGL.device || !pixels) return;
    unsigned int idx = id - 1;

    // Calculate bytes per pixel for row pitch
    int bpp = 0;
    switch (format) {
        case 1: bpp = 8; break;
        case 2: case 3: case 5: case 6: bpp = 16; break;
        case 7: bpp = 32; break;
        case 4: bpp = 32; break;  // RGB expanded to RGBA
        case 8: bpp = 32; break;
        case 9: bpp = 32*4; break;
        case 10: bpp = 32*4; break;
        case 11: bpp = 16; break;
        case 12: bpp = 16*4; break;
        case 13: bpp = 16*4; break;
        default: break;
    }
    unsigned int texWidthBpp = width;
    if (bpp > 0) texWidthBpp = width * bpp / 8;

    ID3D10Device_UpdateSubresource(RLGL.device, (ID3D10Resource *)RLGL.textures[idx].texture, 0, NULL, pixels, texWidthBpp, 0);
}

void rlUpdateTextureRec(unsigned int id, Rectangle rec, int format, const void *pixels)
{
    rlUpdateTexture(id, (int)rec.x, (int)rec.y, (int)rec.width, (int)rec.height, format, pixels);
}

void rlUnloadTexture(unsigned int id)
{
    if (id == 0 || id > RLGL.textureCount) return;
    unsigned int idx = id - 1;
    if (RLGL.textures[idx].srv) { ID3D10ShaderResourceView_Release(RLGL.textures[idx].srv); RLGL.textures[idx].srv = NULL; }
    if (RLGL.textures[idx].texture) { ID3D10Texture2D_Release(RLGL.textures[idx].texture); RLGL.textures[idx].texture = NULL; }
    if (RLGL.textures[idx].sampler) { ID3D10SamplerState_Release(RLGL.textures[idx].sampler); RLGL.textures[idx].sampler = NULL; }
    if (RLGL.textures[idx].dsv) { ID3D10DepthStencilView_Release(RLGL.textures[idx].dsv); RLGL.textures[idx].dsv = NULL; }
    if (RLGL.textures[idx].rtv) { ID3D10RenderTargetView_Release(RLGL.textures[idx].rtv); RLGL.textures[idx].rtv = NULL; }
}

void rlGenTextureMipmaps(unsigned int id, int width, int height, int format, int *mipmaps) { (void)id; (void)width; (void)height; (void)format; (void)mipmaps; }

void *rlReadTexturePixels(unsigned int id, int width, int height, int format)
{
    if (id == 0 || id > RLGL.textureCount || !RLGL.device) return NULL;
    unsigned int idx = id - 1;
    if (!RLGL.textures[idx].texture) return NULL;

    // Create a staging texture for readback
    D3D10_TEXTURE2D_DESC desc;
    ID3D10Texture2D_GetDesc(RLGL.textures[idx].texture, &desc);
    desc.Usage = D3D10_USAGE_STAGING;
    desc.BindFlags = 0;
    desc.CPUAccessFlags = D3D10_CPU_ACCESS_READ;
    desc.MiscFlags = 0;

    ID3D10Texture2D *staging = NULL;
    HRESULT hr = ID3D10Device_CreateTexture2D(RLGL.device, &desc, NULL, &staging);
    if (FAILED(hr) || !staging) return NULL;

    ID3D10Device_CopyResource(RLGL.device, (ID3D10Resource *)staging, (ID3D10Resource *)RLGL.textures[idx].texture);

    D3D10_MAPPED_TEXTURE2D mapped;
    hr = ID3D10Texture2D_Map(staging, 0, D3D10_MAP_READ, 0, &mapped);
    if (FAILED(hr)) {
        ID3D10Texture2D_Release(staging);
        return NULL;
    }

    int bpp = 4;  // default RGBA8
    switch (format) {
        case 1: bpp = 1; break;
        case 2: case 3: case 5: case 6: bpp = 2; break;
        case 7: case 4: bpp = 4; break;
        case 8: bpp = 4; break;
        case 9: case 10: bpp = 16; break;
        case 11: bpp = 2; break;
        case 12: case 13: bpp = 8; break;
        default: break;
    }

    int dataSize = width * height * bpp;
    void *pixels = RL_MALLOC(dataSize);

    // Copy row by row (pitch may differ)
    int rowSize = width * bpp;
    unsigned char *dst = (unsigned char *)pixels;
    unsigned char *src = (unsigned char *)mapped.pData;
    for (int y = 0; y < height; y++) {
        memcpy(dst + y * rowSize, src + y * mapped.RowPitch, rowSize);
    }

    ID3D10Texture2D_Unmap(staging, 0);
    ID3D10Texture2D_Release(staging);

    return pixels;
}

//----------------------------------------------------------------------------------
// Framebuffer management
//----------------------------------------------------------------------------------
unsigned int rlLoadFramebuffer(void)
{
    if (RLGL.framebufferCount >= RL_D3D10_MAX_FRAMEBUFFERS) return 0;
    unsigned int idx = RLGL.framebufferCount;
    memset(&RLGL.framebuffers[idx], 0, sizeof(rlD3D10Framebuffer));
    RLGL.framebufferCount++;
    return idx + 1;
}

void rlFramebufferAttach(unsigned int fboId, unsigned int texId, int attachType, int texType, int mipLevel)
{
    (void)texType; (void)mipLevel;
    if (fboId == 0 || fboId > RLGL.framebufferCount) return;
    if (texId == 0 || texId > RLGL.textureCount) return;
    unsigned int fboIdx = fboId - 1;
    unsigned int texIdx = texId - 1;

    if (attachType == RL_ATTACHMENT_COLOR_CHANNEL0) {
        RLGL.framebuffers[fboIdx].colorTextureId = texId;
        if (!RLGL.textures[texIdx].rtv && RLGL.textures[texIdx].texture) {
            D3D10_RENDER_TARGET_VIEW_DESC rtvDesc;
            memset(&rtvDesc, 0, sizeof(rtvDesc));
            rtvDesc.Format = rlGetDXGIFormat(RLGL.textures[texIdx].format);
            rtvDesc.ViewDimension = D3D10_RTV_DIMENSION_TEXTURE2D;
            rtvDesc.Texture2D.MipSlice = 0;
            ID3D10Device_CreateRenderTargetView(RLGL.device, (ID3D10Resource *)RLGL.textures[texIdx].texture, &rtvDesc, &RLGL.textures[texIdx].rtv);
        }
        RLGL.framebuffers[fboIdx].rtv = RLGL.textures[texIdx].rtv;
    } else if (attachType == RL_ATTACHMENT_DEPTH) {
        RLGL.framebuffers[fboIdx].depthTextureId = texId;
        RLGL.framebuffers[fboIdx].dsv = RLGL.textures[texIdx].dsv;
    }
}

bool rlFramebufferComplete(unsigned int id) {
    if (id == 0 || id > RLGL.framebufferCount) return false;
    unsigned int idx = id - 1;
    return (RLGL.framebuffers[idx].rtv != NULL);
}

void rlUnloadFramebuffer(unsigned int id) {
    if (id == 0 || id > RLGL.framebufferCount) return;
    // RTVs and DSVs are owned by the textures, not the framebuffer
}

//----------------------------------------------------------------------------------
// Vertex buffer management (user VBOs)
//----------------------------------------------------------------------------------
unsigned int rlLoadVertexArray(void) { return 0; }
void rlSetVertexAttribute(unsigned int index, int compSize, int type, bool normalized, int stride, int offset) { (void)index; (void)compSize; (void)type; (void)normalized; (void)stride; (void)offset; }
void rlSetVertexAttributeDivisor(unsigned int index, int divisor) { (void)index; (void)divisor; }
void rlSetVertexAttributeDefault(int locIndex, const void *value, int attribType, int count) { (void)locIndex; (void)value; (void)attribType; (void)count; }

unsigned int rlLoadVertexBuffer(const void *buffer, int size, bool dynamic)
{
    if (!RLGL.device || RLGL.bufferCount >= RL_D3D10_MAX_BUFFERS) return 0;
    unsigned int idx = RLGL.bufferCount;

    D3D10_BUFFER_DESC desc;
    memset(&desc, 0, sizeof(desc));
    desc.ByteWidth = size;
    desc.BindFlags = D3D10_BIND_VERTEX_BUFFER;

    if (dynamic) {
        desc.Usage = D3D10_USAGE_DYNAMIC;
        desc.CPUAccessFlags = D3D10_CPU_ACCESS_WRITE;
    } else {
        desc.Usage = D3D10_USAGE_DEFAULT;
    }

    D3D10_SUBRESOURCE_DATA srd = { 0 };
    srd.pSysMem = buffer;

    ID3D10Device_CreateBuffer(RLGL.device, &desc, buffer ? &srd : NULL, &RLGL.buffers[idx].buffer);
    RLGL.buffers[idx].size = size;
    RLGL.bufferCount++;
    return idx + 1;
}

unsigned int rlLoadVertexBufferElement(const void *buffer, int size, bool dynamic)
{
    if (!RLGL.device || RLGL.bufferCount >= RL_D3D10_MAX_BUFFERS) return 0;
    unsigned int idx = RLGL.bufferCount;

    D3D10_BUFFER_DESC desc;
    memset(&desc, 0, sizeof(desc));
    desc.ByteWidth = size;
    desc.BindFlags = D3D10_BIND_INDEX_BUFFER;

    if (dynamic) {
        desc.Usage = D3D10_USAGE_DYNAMIC;
        desc.CPUAccessFlags = D3D10_CPU_ACCESS_WRITE;
    } else {
        desc.Usage = D3D10_USAGE_DEFAULT;
    }

    D3D10_SUBRESOURCE_DATA srd = { 0 };
    srd.pSysMem = buffer;

    ID3D10Device_CreateBuffer(RLGL.device, &desc, buffer ? &srd : NULL, &RLGL.buffers[idx].buffer);
    RLGL.buffers[idx].size = size;
    RLGL.bufferCount++;
    return idx + 1;
}

void rlUpdateVertexBuffer(unsigned int bufferId, const void *data, int dataSize, int offset)
{
    if (bufferId == 0 || bufferId > RLGL.bufferCount || !RLGL.device || !data) return;
    unsigned int idx = bufferId - 1;
    if (!RLGL.buffers[idx].buffer) return;

    // D3D10 Map on the buffer directly
    void *pData = NULL;
    HRESULT hr = ID3D10Buffer_Map(RLGL.buffers[idx].buffer, D3D10_MAP_WRITE_DISCARD, 0, &pData);
    if (SUCCEEDED(hr)) {
        memcpy((unsigned char *)pData + offset, data, dataSize);
        ID3D10Buffer_Unmap(RLGL.buffers[idx].buffer);
    }
}

void rlUpdateVertexBufferElements(unsigned int bufferId, const void *data, int dataSize, int offset)
{
    rlUpdateVertexBuffer(bufferId, data, dataSize, offset);
}

void rlUnloadVertexArray(unsigned int vaoId) { (void)vaoId; }

void rlUnloadVertexBuffer(unsigned int vboId)
{
    if (vboId == 0 || vboId > RLGL.bufferCount) return;
    unsigned int idx = vboId - 1;
    if (RLGL.buffers[idx].buffer) { ID3D10Buffer_Release(RLGL.buffers[idx].buffer); RLGL.buffers[idx].buffer = NULL; }
}

void rlBindVertexArray(unsigned int vaoId) { (void)vaoId; }
void rlEnableVertexArray(unsigned int vaoId) { (void)vaoId; }
void rlDisableVertexArray(void) { }
void rlEnableVertexBuffer(unsigned int id) {
    if (id == 0 || id > RLGL.bufferCount || !RLGL.device) return;
    unsigned int idx = id - 1;
    UINT stride = RLGL.buffers[idx].size;
    UINT offset = 0;
    ID3D10Device_IASetVertexBuffers(RLGL.device, 0, 1, &RLGL.buffers[idx].buffer, &stride, &offset);
}
void rlDisableVertexBuffer(void) { }
void rlEnableVertexBufferElement(unsigned int id) {
    if (id == 0 || id > RLGL.bufferCount || !RLGL.device) return;
    unsigned int idx = id - 1;
    ID3D10Device_IASetIndexBuffer(RLGL.device, RLGL.buffers[idx].buffer, DXGI_FORMAT_R16_UINT, 0);
}
void rlDisableVertexBufferElement(void) { }
void rlEnableVertexAttribute(unsigned int index) { (void)index; }
void rlDisableVertexAttribute(unsigned int index) { (void)index; }

void rlDrawVertexArray(int offset, int count) {
    if (!RLGL.device) return;
    ID3D10Device_IASetPrimitiveTopology(RLGL.device, D3D10_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    ID3D10Device_Draw(RLGL.device, count, offset);
}

void rlDrawVertexArrayElements(int offset, int count, const void *buffer) {
    (void)buffer;
    if (!RLGL.device) return;
    ID3D10Device_IASetPrimitiveTopology(RLGL.device, D3D10_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    ID3D10Device_DrawIndexed(RLGL.device, count, offset, 0);
}

void rlDrawVertexArrayInstanced(int offset, int count, int instances) {
    (void)offset;
    if (!RLGL.device) return;
    ID3D10Device_IASetPrimitiveTopology(RLGL.device, D3D10_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    ID3D10Device_DrawInstanced(RLGL.device, count, instances, 0, 0);
}

void rlDrawVertexArrayElementsInstanced(int offset, int count, const void *buffer, int instances) {
    (void)buffer;
    if (!RLGL.device) return;
    ID3D10Device_IASetPrimitiveTopology(RLGL.device, D3D10_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    ID3D10Device_DrawIndexedInstanced(RLGL.device, count, instances, offset, 0, 0);
}

//----------------------------------------------------------------------------------
// Shader management
//----------------------------------------------------------------------------------
unsigned int rlCompileShader(const char *shaderCode, int type)
{
    if (!RLGL.device || RLGL.shaderCount >= RL_D3D10_MAX_SHADERS) return 0;

    const char *target = (type == RL_VERTEX_SHADER) ? "vs_4_0" : "ps_4_0";
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
        if (RLGL.shaderCount < RL_D3D10_MAX_SHADERS) {
            RLGL.shaders[RLGL.shaderCount].vsBlob = blob;
            ID3D10Device_CreateVertexShader(RLGL.device, ID3D10Blob_GetBufferPointer(blob),
                ID3D10Blob_GetBufferSize(blob), &RLGL.shaders[RLGL.shaderCount].vertexShader);
        }
    } else {
        if (RLGL.shaderCount < RL_D3D10_MAX_SHADERS) {
            RLGL.shaders[RLGL.shaderCount].psBlob = blob;
            ID3D10Device_CreatePixelShader(RLGL.device, ID3D10Blob_GetBufferPointer(blob),
                ID3D10Blob_GetBufferSize(blob), &RLGL.shaders[RLGL.shaderCount].pixelShader);
        }
    }

    return id;
}

unsigned int rlLoadShaderCode(const char *vsCode, const char *fsCode)
{
    if (!vsCode) vsCode = defaultVShaderHLSL;
    if (!fsCode) fsCode = defaultPShaderHLSL;

    if (RLGL.shaderCount >= RL_D3D10_MAX_SHADERS) return 0;
    unsigned int idx = RLGL.shaderCount;

    // Compile vertex shader (Shader Model 4.0)
    ID3DBlob *vsBlob = NULL, *psBlob = NULL, *errors = NULL;
    HRESULT hr = D3DCompile(vsCode, strlen(vsCode), NULL, NULL, NULL, "VSMain", "vs_4_0",
        D3DCOMPILE_OPTIMIZATION_LEVEL3 | D3DCOMPILE_PACK_MATRIX_COLUMN_MAJOR, 0, &vsBlob, &errors);
    if (FAILED(hr)) {
        if (errors) { TRACELOG(RL_LOG_WARNING, "SHADER: VS compile error: %s", (char *)ID3D10Blob_GetBufferPointer(errors)); ID3D10Blob_Release(errors); }
        return RLGL.State.defaultShaderId;
    }
    if (errors) ID3D10Blob_Release(errors);
    errors = NULL;

    // Compile pixel shader (Shader Model 4.0)
    hr = D3DCompile(fsCode, strlen(fsCode), NULL, NULL, NULL, "PSMain", "ps_4_0",
        D3DCOMPILE_OPTIMIZATION_LEVEL3 | D3DCOMPILE_PACK_MATRIX_COLUMN_MAJOR, 0, &psBlob, &errors);
    if (FAILED(hr)) {
        if (errors) { TRACELOG(RL_LOG_WARNING, "SHADER: PS compile error: %s", (char *)ID3D10Blob_GetBufferPointer(errors)); ID3D10Blob_Release(errors); }
        ID3D10Blob_Release(vsBlob);
        return RLGL.State.defaultShaderId;
    }
    if (errors) ID3D10Blob_Release(errors);

    // Create shaders (D3D10: no class linkage parameter)
    ID3D10Device_CreateVertexShader(RLGL.device, ID3D10Blob_GetBufferPointer(vsBlob), ID3D10Blob_GetBufferSize(vsBlob), &RLGL.shaders[idx].vertexShader);
    ID3D10Device_CreatePixelShader(RLGL.device, ID3D10Blob_GetBufferPointer(psBlob), ID3D10Blob_GetBufferSize(psBlob), &RLGL.shaders[idx].pixelShader);

    // Create input layout
    D3D10_INPUT_ELEMENT_DESC layout[] = {
        { "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT,    0, 0,                            D3D10_INPUT_PER_VERTEX_DATA, 0 },
        { "TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT,       1, 0,                            D3D10_INPUT_PER_VERTEX_DATA, 0 },
        { "NORMAL",   0, DXGI_FORMAT_R32G32B32_FLOAT,    2, 0,                            D3D10_INPUT_PER_VERTEX_DATA, 0 },
        { "COLOR",    0, DXGI_FORMAT_R8G8B8A8_UNORM,     3, 0,                            D3D10_INPUT_PER_VERTEX_DATA, 0 },
    };
    ID3D10Device_CreateInputLayout(RLGL.device, layout, 4,
        ID3D10Blob_GetBufferPointer(vsBlob), ID3D10Blob_GetBufferSize(vsBlob), &RLGL.shaders[idx].inputLayout);

    RLGL.shaders[idx].vsBlob = vsBlob;
    RLGL.shaders[idx].psBlob = psBlob;
    RLGL.shaderCount++;

    unsigned int id = idx + 1;
    TRACELOG(RL_LOG_INFO, "SHADER: [ID %i] D3D10 shader loaded successfully", id);
    return id;
}

unsigned int rlLoadShaderProgram(unsigned int vShaderId, unsigned int fShaderId) {
    (void)vShaderId; (void)fShaderId;
    return 0; // Not used in D3D10 path - rlLoadShaderCode handles everything
}

void rlUnloadShaderProgram(unsigned int id) {
    if (id == 0 || id > RLGL.shaderCount) return;
    unsigned int idx = id - 1;
    if (RLGL.shaders[idx].vertexShader) { ID3D10VertexShader_Release(RLGL.shaders[idx].vertexShader); RLGL.shaders[idx].vertexShader = NULL; }
    if (RLGL.shaders[idx].pixelShader) { ID3D10PixelShader_Release(RLGL.shaders[idx].pixelShader); RLGL.shaders[idx].pixelShader = NULL; }
    if (RLGL.shaders[idx].inputLayout) { ID3D10InputLayout_Release(RLGL.shaders[idx].inputLayout); RLGL.shaders[idx].inputLayout = NULL; }
    if (RLGL.shaders[idx].vsBlob) { ID3D10Blob_Release(RLGL.shaders[idx].vsBlob); RLGL.shaders[idx].vsBlob = NULL; }
    if (RLGL.shaders[idx].psBlob) { ID3D10Blob_Release(RLGL.shaders[idx].psBlob); RLGL.shaders[idx].psBlob = NULL; }
}

int rlGetLocationUniform(unsigned int shaderId, const char *uniformName) {
    (void)shaderId; (void)uniformName;
    return -1; // D3D10 uses constant buffers, not individual uniform locations
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

// Compute shader stubs (not available in D3D10)
unsigned int rlLoadComputeShaderProgram(unsigned int shaderId) {
    (void)shaderId;
    TRACELOG(RL_LOG_WARNING, "SHADER: Compute shaders not supported in D3D10");
    return 0;
}
void rlComputeShaderDispatch(unsigned int groupX, unsigned int groupY, unsigned int groupZ) { (void)groupX; (void)groupY; (void)groupZ; }

// SSBO stubs (not available in D3D10)
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
        // Left/Right/Bottom/Top faces
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
        TRACELOG(RL_LOG_INFO, "SHADER: [ID %i] Default D3D10 shader loaded successfully", RLGL.State.defaultShaderId);
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
// Auxiliar math functions (identical to other backends)
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

#endif // GRAPHICS_API_DIRECT3D10
