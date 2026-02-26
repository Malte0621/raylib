/**********************************************************************************************
*
*   rl_backend_d3d9 - Direct3D 9 rendering backend for rlgl
*
*   This implements all rlgl functions using the Direct3D 9 API.
*   Requires Windows XP or later with Shader Model 3.0 hardware.
*
*   Key differences from D3D10+ backends:
*     - No separate device context or DXGI; uses IDirect3D9 + IDirect3DDevice9
*     - No state objects; render states set directly via SetRenderState/SetSamplerState
*     - No constant buffers; uses SetVertexShaderConstantF / SetPixelShaderConstantF
*     - No SRVs; textures bound directly via SetTexture
*     - Shader Model 3.0 (vs_3_0 / ps_3_0) with combined sampler2D / tex2D syntax
*     - Lock/Unlock on resources instead of Map/Unmap
*     - Separate IDirect3DVertexBuffer9 / IDirect3DIndexBuffer9 types
*     - D3DFMT_A8R8G8B8 is BGRA in memory; requires RGBA<->BGRA swizzle
*     - BeginScene / EndScene required around draw calls
*     - Vertex declarations instead of input layouts
*
*   CONFIGURATION:
*       #define GRAPHICS_API_DIRECT3D9    Select this backend
*
*   LICENSE: zlib/libpng (same as raylib)
*
**********************************************************************************************/

#include "raylib.h"
#include "rlgl.h"
#include "utils.h"
#include "rl_backend.h"

#if defined(GRAPHICS_API_DIRECT3D9)

#include <stdlib.h>
#include <string.h>
#include <math.h>

// Windows / D3D9 headers
#define WIN32_LEAN_AND_MEAN
#define NOGDI               // Prevent wingdi.h Rectangle() function conflicting with raylib's Rectangle type
#define COBJMACROS
#define CINTERFACE
#include <windows.h>
#include <d3d9.h>
#include <d3dcompiler.h>

#ifdef _MSC_VER
    #pragma comment(lib, "d3d9.lib")
    #pragma comment(lib, "d3dcompiler.lib")
#endif

//----------------------------------------------------------------------------------
// Defines and Macros
//----------------------------------------------------------------------------------
#ifndef RL_D3D9_MAX_TEXTURES
    #define RL_D3D9_MAX_TEXTURES       4096
#endif
#ifndef RL_D3D9_MAX_SHADERS
    #define RL_D3D9_MAX_SHADERS        256
#endif
#ifndef RL_D3D9_MAX_BUFFERS
    #define RL_D3D9_MAX_BUFFERS        4096
#endif
#ifndef RL_D3D9_MAX_FRAMEBUFFERS
    #define RL_D3D9_MAX_FRAMEBUFFERS   256
#endif

//----------------------------------------------------------------------------------
// Types and Structures
//----------------------------------------------------------------------------------

// Tracked D3D9 texture resource
typedef struct rlD3D9Texture {
    IDirect3DTexture9 *texture;         // Regular or render target texture (NULL for depth-only)
    IDirect3DSurface9 *surface;         // Level 0 surface (for RT/depth binding)
    int width, height, mipmaps, format;
    bool isRenderTarget;
    bool isDepth;
} rlD3D9Texture;

// Tracked D3D9 shader
typedef struct rlD3D9Shader {
    IDirect3DVertexShader9 *vertexShader;
    IDirect3DPixelShader9 *pixelShader;
    IDirect3DVertexDeclaration9 *vertexDecl;
    ID3DBlob *vsBlob;
    ID3DBlob *psBlob;
} rlD3D9Shader;

// Tracked D3D9 buffer (for user VBOs/IBOs)
typedef struct rlD3D9Buffer {
    IDirect3DVertexBuffer9 *vb;         // Non-NULL for vertex buffers
    IDirect3DIndexBuffer9 *ib;          // Non-NULL for index buffers
    unsigned int size;
} rlD3D9Buffer;

// Tracked D3D9 framebuffer
typedef struct rlD3D9Framebuffer {
    unsigned int colorTextureId;
    unsigned int depthTextureId;
    IDirect3DSurface9 *colorSurface;
    IDirect3DSurface9 *depthSurface;
} rlD3D9Framebuffer;

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

    // D3D9 objects
    IDirect3D9 *d3d9;                  // D3D9 factory
    IDirect3DDevice9 *device;           // D3D9 device

    // Default vertex declaration
    IDirect3DVertexDeclaration9 *defaultVertexDecl;

    // Batch rendering vertex/index buffers
    IDirect3DVertexBuffer9 *batchVertexBuffers[4]; // pos, texcoord, normal, color
    IDirect3DIndexBuffer9 *batchIndexBuffer;

    // Render state tracking
    bool stateDirty;

    // Resource tracking
    rlD3D9Texture textures[RL_D3D9_MAX_TEXTURES];
    unsigned int textureCount;

    rlD3D9Shader shaders[RL_D3D9_MAX_SHADERS];
    unsigned int shaderCount;

    rlD3D9Buffer buffers[RL_D3D9_MAX_BUFFERS];
    unsigned int bufferCount;

    rlD3D9Framebuffer framebuffers[RL_D3D9_MAX_FRAMEBUFFERS];
    unsigned int framebufferCount;

    rlRenderBatch defaultBatch;
    rlRenderBatch *currentBatch;
} rlglData;

static rlglData RLGL = { 0 };

//----------------------------------------------------------------------------------
// Forward declarations
//----------------------------------------------------------------------------------
static D3DFORMAT rlGetD3D9Format(int rlFormat);
static void rlUpdateD3D9States(void);
static void rlSwizzleRGBAtoBGRA(const unsigned char *src, unsigned char *dst, int pixelCount);
static void rlSwizzleBGRAtoRGBA(const unsigned char *src, unsigned char *dst, int pixelCount);
static void rlLoadShaderDefault(void);
static void rlUnloadShaderDefault(void);

// Auxiliar math types and functions
typedef struct rl_float16 { float v[16]; } rl_float16;
static rl_float16 rlMatrixToFloatV(Matrix mat);
static Matrix rlMatrixIdentity(void);
static Matrix rlMatrixMultiply(Matrix left, Matrix right);
static Matrix rlMatrixTranspose(Matrix mat);
static Matrix rlMatrixInvert(Matrix mat);

//----------------------------------------------------------------------------------
// Default HLSL shaders (Shader Model 3.0)
// D3D9 HLSL differences from D3D10+:
//   - Output position semantic: POSITION (not SV_POSITION)
//   - Output color semantic: COLOR0 (not SV_TARGET)
//   - Texture sampling: tex2D(sampler2D, texcoord)
//   - Combined sampler2D instead of Texture2D + SamplerState
//   - Constants via registers, not cbuffers
//----------------------------------------------------------------------------------
static const char *defaultVShaderHLSL =
    "float4x4 mvp : register(c0);\n"
    "struct VSInput { float3 pos : POSITION; float2 texcoord : TEXCOORD0; float3 normal : NORMAL; float4 color : COLOR0; };\n"
    "struct VSOutput { float4 pos : POSITION; float2 texcoord : TEXCOORD0; float4 color : COLOR0; };\n"
    "VSOutput VSMain(VSInput input) {\n"
    "    VSOutput output;\n"
    "    output.pos = mul(mvp, float4(input.pos, 1.0));\n"
    "    output.texcoord = input.texcoord;\n"
    "    output.color = input.color;\n"
    "    return output;\n"
    "}\n";

static const char *defaultPShaderHLSL =
    "float4 colDiffuse : register(c0);\n"
    "sampler2D sampler0 : register(s0);\n"
    "struct PSInput { float2 texcoord : TEXCOORD0; float4 color : COLOR0; };\n"
    "float4 PSMain(PSInput input) : COLOR0 {\n"
    "    float4 texColor = tex2D(sampler0, input.texcoord);\n"
    "    return texColor * input.color * colDiffuse;\n"
    "}\n";

//----------------------------------------------------------------------------------
// Helper: Convert raylib pixel format to D3DFORMAT
// NOTE: D3DFMT_A8R8G8B8 stores BGRA in memory on little-endian;
//       8-bit RGBA pixel data must be swizzled before upload.
//----------------------------------------------------------------------------------
static D3DFORMAT rlGetD3D9Format(int rlFormat)
{
    switch (rlFormat) {
        case 1:  return D3DFMT_L8;                  // GRAYSCALE
        case 2:  return D3DFMT_A8L8;                // GRAY_ALPHA
        case 3:  return D3DFMT_R5G6B5;              // R5G6B5
        case 4:  return D3DFMT_A8R8G8B8;            // R8G8B8 -> expanded to BGRA
        case 5:  return D3DFMT_A1R5G5B5;            // R5G5B5A1
        case 6:  return D3DFMT_A4R4G4B4;            // R4G4B4A4
        case 7:  return D3DFMT_A8R8G8B8;            // R8G8B8A8 (with RGBA->BGRA swizzle)
        case 8:  return D3DFMT_R32F;                // R32
        case 9:  return D3DFMT_A32B32G32R32F;       // R32G32B32 -> expanded to RGBA32F
        case 10: return D3DFMT_A32B32G32R32F;       // R32G32B32A32
        case 11: return D3DFMT_R16F;                // R16
        case 12: return D3DFMT_A16B16G16R16F;       // R16G16B16 -> expanded to RGBA16F
        case 13: return D3DFMT_A16B16G16R16F;       // R16G16B16A16
        case 14: return D3DFMT_DXT1;                // DXT1 RGB
        case 15: return D3DFMT_DXT3;                // DXT3 RGBA
        case 16: return D3DFMT_DXT5;                // DXT5 RGBA
        case 17: return D3DFMT_DXT1;                // DXT1 RGBA
        default: return D3DFMT_A8R8G8B8;
    }
}

//----------------------------------------------------------------------------------
// Helper: Swizzle RGBA -> BGRA (for D3DFMT_A8R8G8B8 upload)
//----------------------------------------------------------------------------------
static void rlSwizzleRGBAtoBGRA(const unsigned char *src, unsigned char *dst, int pixelCount)
{
    for (int i = 0; i < pixelCount; i++) {
        dst[4*i + 0] = src[4*i + 2]; // B <- R
        dst[4*i + 1] = src[4*i + 1]; // G <- G
        dst[4*i + 2] = src[4*i + 0]; // R <- B
        dst[4*i + 3] = src[4*i + 3]; // A <- A
    }
}

//----------------------------------------------------------------------------------
// Helper: Swizzle BGRA -> RGBA (for D3DFMT_A8R8G8B8 readback)
//----------------------------------------------------------------------------------
static void rlSwizzleBGRAtoRGBA(const unsigned char *src, unsigned char *dst, int pixelCount)
{
    rlSwizzleRGBAtoBGRA(src, dst, pixelCount); // Same swap: byte 0 <-> byte 2
}

//----------------------------------------------------------------------------------
// Helper: Update cached D3D9 render states (direct SetRenderState calls)
//----------------------------------------------------------------------------------
static void rlUpdateD3D9States(void)
{
    if (!RLGL.device || !RLGL.stateDirty) return;
    RLGL.stateDirty = false;

    // Blend state
    IDirect3DDevice9_SetRenderState(RLGL.device, D3DRS_ALPHABLENDENABLE, RLGL.State.colorBlendEnabled ? TRUE : FALSE);
    IDirect3DDevice9_SetRenderState(RLGL.device, D3DRS_SRCBLEND, D3DBLEND_SRCALPHA);
    IDirect3DDevice9_SetRenderState(RLGL.device, D3DRS_DESTBLEND, D3DBLEND_INVSRCALPHA);
    IDirect3DDevice9_SetRenderState(RLGL.device, D3DRS_BLENDOP, D3DBLENDOP_ADD);
    IDirect3DDevice9_SetRenderState(RLGL.device, D3DRS_SEPARATEALPHABLENDENABLE, TRUE);
    IDirect3DDevice9_SetRenderState(RLGL.device, D3DRS_SRCBLENDALPHA, D3DBLEND_ONE);
    IDirect3DDevice9_SetRenderState(RLGL.device, D3DRS_DESTBLENDALPHA, D3DBLEND_INVSRCALPHA);
    IDirect3DDevice9_SetRenderState(RLGL.device, D3DRS_BLENDOPALPHA, D3DBLENDOP_ADD);

    // Depth state
    IDirect3DDevice9_SetRenderState(RLGL.device, D3DRS_ZENABLE, RLGL.State.depthTestEnabled ? D3DZB_TRUE : D3DZB_FALSE);
    IDirect3DDevice9_SetRenderState(RLGL.device, D3DRS_ZWRITEENABLE, RLGL.State.depthWriteEnabled ? TRUE : FALSE);
    IDirect3DDevice9_SetRenderState(RLGL.device, D3DRS_ZFUNC, D3DCMP_LESSEQUAL);

    // Rasterizer state
    IDirect3DDevice9_SetRenderState(RLGL.device, D3DRS_CULLMODE, RLGL.State.backfaceCullingEnabled ? D3DCULL_CW : D3DCULL_NONE);
    IDirect3DDevice9_SetRenderState(RLGL.device, D3DRS_FILLMODE, D3DFILL_SOLID);
    IDirect3DDevice9_SetRenderState(RLGL.device, D3DRS_SCISSORTESTENABLE, RLGL.State.scissorTest ? TRUE : FALSE);

    // Lighting off (we use programmable shaders)
    IDirect3DDevice9_SetRenderState(RLGL.device, D3DRS_LIGHTING, FALSE);
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
    D3DVIEWPORT9 vp;
    vp.X = (DWORD)x;
    vp.Y = (DWORD)y;
    vp.Width = (DWORD)width;
    vp.Height = (DWORD)height;
    vp.MinZ = 0.0f;
    vp.MaxZ = 1.0f;
    IDirect3DDevice9_SetViewport(RLGL.device, &vp);

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

void rlEnd(void) { }

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
        if (RLGL.framebuffers[idx].colorSurface)
            IDirect3DDevice9_SetRenderTarget(RLGL.device, 0, RLGL.framebuffers[idx].colorSurface);
        if (RLGL.framebuffers[idx].depthSurface)
            IDirect3DDevice9_SetDepthStencilSurface(RLGL.device, RLGL.framebuffers[idx].depthSurface);
    }
}
void rlDisableFramebuffer(void) { }
unsigned int rlGetActiveFramebuffer(void) { return 0; }

//----------------------------------------------------------------------------------
// Render state management
//----------------------------------------------------------------------------------
void rlEnableColorBlend(void) { RLGL.State.colorBlendEnabled = true; RLGL.stateDirty = true; rlUpdateD3D9States(); }
void rlDisableColorBlend(void) { RLGL.State.colorBlendEnabled = false; RLGL.stateDirty = true; rlUpdateD3D9States(); }
void rlEnableDepthTest(void) { RLGL.State.depthTestEnabled = true; RLGL.stateDirty = true; rlUpdateD3D9States(); }
void rlDisableDepthTest(void) { RLGL.State.depthTestEnabled = false; RLGL.stateDirty = true; rlUpdateD3D9States(); }
void rlEnableDepthMask(void) { RLGL.State.depthWriteEnabled = true; RLGL.stateDirty = true; rlUpdateD3D9States(); }
void rlDisableDepthMask(void) { RLGL.State.depthWriteEnabled = false; RLGL.stateDirty = true; rlUpdateD3D9States(); }
void rlEnableBackfaceCulling(void) { RLGL.State.backfaceCullingEnabled = true; RLGL.stateDirty = true; rlUpdateD3D9States(); }
void rlDisableBackfaceCulling(void) { RLGL.State.backfaceCullingEnabled = false; RLGL.stateDirty = true; rlUpdateD3D9States(); }
void rlColorMask(bool r, bool g, bool b, bool a) { (void)r; (void)g; (void)b; (void)a; }
void rlSetCullFace(int mode) { (void)mode; }
void rlEnableScissorTest(void) { RLGL.State.scissorTest = true; RLGL.stateDirty = true; rlUpdateD3D9States(); }
void rlDisableScissorTest(void) { RLGL.State.scissorTest = false; RLGL.stateDirty = true; rlUpdateD3D9States(); }
void rlScissor(int x, int y, int width, int height) {
    RECT rect = { x, y, x + width, y + height };
    if (RLGL.device) IDirect3DDevice9_SetScissorRect(RLGL.device, &rect);
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
void rlClearScreenBuffers(void) { /* Handled externally */ }
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

    // ---- Create D3D9 factory ----
    RLGL.d3d9 = Direct3DCreate9(D3D_SDK_VERSION);
    if (!RLGL.d3d9) {
        TRACELOG(RL_LOG_FATAL, "DISPLAY: Failed to create IDirect3D9 interface");
        return;
    }

    // ---- Create D3D9 device ----
    D3DPRESENT_PARAMETERS pp;
    memset(&pp, 0, sizeof(pp));
    pp.Windowed = TRUE;
    pp.SwapEffect = D3DSWAPEFFECT_DISCARD;
    pp.BackBufferFormat = D3DFMT_UNKNOWN;
    pp.BackBufferWidth = (UINT)width;
    pp.BackBufferHeight = (UINT)height;
    pp.hDeviceWindow = GetDesktopWindow();
    pp.PresentationInterval = D3DPRESENT_INTERVAL_IMMEDIATE;
    pp.EnableAutoDepthStencil = FALSE;

    HRESULT hr = IDirect3D9_CreateDevice(RLGL.d3d9, D3DADAPTER_DEFAULT, D3DDEVTYPE_HAL,
        pp.hDeviceWindow, D3DCREATE_HARDWARE_VERTEXPROCESSING | D3DCREATE_FPU_PRESERVE,
        &pp, &RLGL.device);

    if (FAILED(hr)) {
        TRACELOG(RL_LOG_WARNING, "DISPLAY: D3D9 hardware VP device creation failed, trying software VP");
        hr = IDirect3D9_CreateDevice(RLGL.d3d9, D3DADAPTER_DEFAULT, D3DDEVTYPE_HAL,
            pp.hDeviceWindow, D3DCREATE_SOFTWARE_VERTEXPROCESSING | D3DCREATE_FPU_PRESERVE,
            &pp, &RLGL.device);
    }

    if (FAILED(hr) || !RLGL.device) {
        TRACELOG(RL_LOG_FATAL, "DISPLAY: Failed to create D3D9 device (HRESULT: 0x%08X)", (unsigned int)hr);
        return;
    }

    TRACELOG(RL_LOG_INFO, "DISPLAY: D3D9 device created successfully");

    // ---- Set default sampler state ----
    IDirect3DDevice9_SetSamplerState(RLGL.device, 0, D3DSAMP_MINFILTER, D3DTEXF_POINT);
    IDirect3DDevice9_SetSamplerState(RLGL.device, 0, D3DSAMP_MAGFILTER, D3DTEXF_POINT);
    IDirect3DDevice9_SetSamplerState(RLGL.device, 0, D3DSAMP_MIPFILTER, D3DTEXF_POINT);
    IDirect3DDevice9_SetSamplerState(RLGL.device, 0, D3DSAMP_ADDRESSU, D3DTADDRESS_CLAMP);
    IDirect3DDevice9_SetSamplerState(RLGL.device, 0, D3DSAMP_ADDRESSV, D3DTADDRESS_CLAMP);
    IDirect3DDevice9_SetSamplerState(RLGL.device, 0, D3DSAMP_ADDRESSW, D3DTADDRESS_CLAMP);

    // ---- Create default vertex declaration ----
    D3DVERTEXELEMENT9 elements[] = {
        { 0, 0, D3DDECLTYPE_FLOAT3,   D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_POSITION, 0 },
        { 1, 0, D3DDECLTYPE_FLOAT2,   D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_TEXCOORD, 0 },
        { 2, 0, D3DDECLTYPE_FLOAT3,   D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_NORMAL,   0 },
        { 3, 0, D3DDECLTYPE_UBYTE4N,  D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_COLOR,    0 },
        D3DDECL_END()
    };
    IDirect3DDevice9_CreateVertexDeclaration(RLGL.device, elements, &RLGL.defaultVertexDecl);

    // ---- Create default 1x1 white texture ----
    // D3DFMT_A8R8G8B8 expects BGRA byte order; white = all 255, no swizzle needed
    unsigned char whitePixelBGRA[4] = { 255, 255, 255, 255 };
    IDirect3DTexture9 *defaultTex = NULL;
    hr = IDirect3DDevice9_CreateTexture(RLGL.device, 1, 1, 1, 0,
        D3DFMT_A8R8G8B8, D3DPOOL_MANAGED, &defaultTex, NULL);
    if (SUCCEEDED(hr) && defaultTex) {
        D3DLOCKED_RECT locked;
        hr = IDirect3DTexture9_LockRect(defaultTex, 0, &locked, NULL, 0);
        if (SUCCEEDED(hr)) {
            memcpy(locked.pBits, whitePixelBGRA, 4);
            IDirect3DTexture9_UnlockRect(defaultTex, 0);
        }

        unsigned int idx = RLGL.textureCount;
        RLGL.textures[idx].texture = defaultTex;
        RLGL.textures[idx].width = 1;
        RLGL.textures[idx].height = 1;
        RLGL.textures[idx].mipmaps = 1;
        RLGL.textures[idx].format = 7; // RGBA8
        RLGL.textureCount++;
        RLGL.State.defaultTextureId = idx + 1;
        TRACELOG(RL_LOG_INFO, "TEXTURE: [ID %i] Default texture loaded successfully", RLGL.State.defaultTextureId);
    }

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
    rlUpdateD3D9States();

    TRACELOG(RL_LOG_INFO, "RLGL: Direct3D 9 backend initialized successfully");
}

void rlglClose(void)
{
    rlUnloadRenderBatch(RLGL.defaultBatch);
    rlUnloadShaderDefault();

    if (RLGL.State.defaultTextureId > 0) rlUnloadTexture(RLGL.State.defaultTextureId);

    // Release all tracked resources
    for (unsigned int i = 0; i < RLGL.textureCount; i++) {
        if (RLGL.textures[i].surface) IDirect3DSurface9_Release(RLGL.textures[i].surface);
        if (RLGL.textures[i].texture) IDirect3DTexture9_Release(RLGL.textures[i].texture);
    }
    for (unsigned int i = 0; i < RLGL.shaderCount; i++) {
        if (RLGL.shaders[i].vertexShader) IDirect3DVertexShader9_Release(RLGL.shaders[i].vertexShader);
        if (RLGL.shaders[i].pixelShader) IDirect3DPixelShader9_Release(RLGL.shaders[i].pixelShader);
        if (RLGL.shaders[i].vertexDecl) IDirect3DVertexDeclaration9_Release(RLGL.shaders[i].vertexDecl);
        if (RLGL.shaders[i].vsBlob) ID3D10Blob_Release(RLGL.shaders[i].vsBlob);
        if (RLGL.shaders[i].psBlob) ID3D10Blob_Release(RLGL.shaders[i].psBlob);
    }
    for (unsigned int i = 0; i < RLGL.bufferCount; i++) {
        if (RLGL.buffers[i].vb) IDirect3DVertexBuffer9_Release(RLGL.buffers[i].vb);
        if (RLGL.buffers[i].ib) IDirect3DIndexBuffer9_Release(RLGL.buffers[i].ib);
    }

    for (int i = 0; i < 4; i++) {
        if (RLGL.batchVertexBuffers[i]) IDirect3DVertexBuffer9_Release(RLGL.batchVertexBuffers[i]);
    }
    if (RLGL.batchIndexBuffer) IDirect3DIndexBuffer9_Release(RLGL.batchIndexBuffer);

    if (RLGL.defaultVertexDecl) IDirect3DVertexDeclaration9_Release(RLGL.defaultVertexDecl);
    if (RLGL.device) { IDirect3DDevice9_Release(RLGL.device); RLGL.device = NULL; }
    if (RLGL.d3d9) { IDirect3D9_Release(RLGL.d3d9); RLGL.d3d9 = NULL; }

    TRACELOG(RL_LOG_INFO, "RLGL: Direct3D 9 backend closed successfully");
}

void rlLoadExtensions(void *loader)
{
    (void)loader;
    // D3D9 SM3.0 capabilities
    RLGL.ExtSupported.computeShader = false;  // No compute shaders in D3D9
    RLGL.ExtSupported.ssbo = false;           // No SSBOs in D3D9
    RLGL.ExtSupported.texNPOT = true;         // SM3.0 hardware supports NPOT
    RLGL.ExtSupported.texCompDXT = true;      // DXT always available in D3D9
    RLGL.ExtSupported.maxAnisotropyLevel = 16.0f;
    RLGL.ExtSupported.maxDepthBits = 24;

    TRACELOG(RL_LOG_INFO, "D3D9: Device capabilities loaded");
}

int rlGetVersion(void) { return RL_DIRECT3D_9; }

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

        // Dynamic vertex buffers (D3DPOOL_DEFAULT + DYNAMIC + WRITEONLY)
        IDirect3DDevice9_CreateVertexBuffer(RLGL.device, vbSize * 3 * sizeof(float),
            D3DUSAGE_DYNAMIC | D3DUSAGE_WRITEONLY, 0, D3DPOOL_DEFAULT, &RLGL.batchVertexBuffers[0], NULL);
        IDirect3DDevice9_CreateVertexBuffer(RLGL.device, vbSize * 2 * sizeof(float),
            D3DUSAGE_DYNAMIC | D3DUSAGE_WRITEONLY, 0, D3DPOOL_DEFAULT, &RLGL.batchVertexBuffers[1], NULL);
        IDirect3DDevice9_CreateVertexBuffer(RLGL.device, vbSize * 3 * sizeof(float),
            D3DUSAGE_DYNAMIC | D3DUSAGE_WRITEONLY, 0, D3DPOOL_DEFAULT, &RLGL.batchVertexBuffers[2], NULL);
        IDirect3DDevice9_CreateVertexBuffer(RLGL.device, vbSize * 4 * sizeof(unsigned char),
            D3DUSAGE_DYNAMIC | D3DUSAGE_WRITEONLY, 0, D3DPOOL_DEFAULT, &RLGL.batchVertexBuffers[3], NULL);

        // Static index buffer (D3DPOOL_MANAGED for simple fill+forget)
        IDirect3DDevice9_CreateIndexBuffer(RLGL.device, bufferElements * 6 * sizeof(unsigned short),
            D3DUSAGE_WRITEONLY, D3DFMT_INDEX16, D3DPOOL_MANAGED, &RLGL.batchIndexBuffer, NULL);

        if (RLGL.batchIndexBuffer) {
            void *pData;
            if (SUCCEEDED(IDirect3DIndexBuffer9_Lock(RLGL.batchIndexBuffer, 0, 0, &pData, 0))) {
                memcpy(pData, batch.vertexBuffer[0].indices, bufferElements * 6 * sizeof(unsigned short));
                IDirect3DIndexBuffer9_Unlock(RLGL.batchIndexBuffer);
            }
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

    // Upload vertex data via Lock/Unlock (D3D9 uses Lock on the buffer directly)
    {
        void *pData = NULL;

        // Positions
        if (RLGL.batchVertexBuffers[0]) {
            if (SUCCEEDED(IDirect3DVertexBuffer9_Lock(RLGL.batchVertexBuffers[0], 0, vCount * 3 * sizeof(float), &pData, D3DLOCK_DISCARD))) {
                memcpy(pData, batch->vertexBuffer[batch->currentBuffer].vertices, vCount * 3 * sizeof(float));
                IDirect3DVertexBuffer9_Unlock(RLGL.batchVertexBuffers[0]);
            }
        }
        // Texcoords
        if (RLGL.batchVertexBuffers[1]) {
            if (SUCCEEDED(IDirect3DVertexBuffer9_Lock(RLGL.batchVertexBuffers[1], 0, vCount * 2 * sizeof(float), &pData, D3DLOCK_DISCARD))) {
                memcpy(pData, batch->vertexBuffer[batch->currentBuffer].texcoords, vCount * 2 * sizeof(float));
                IDirect3DVertexBuffer9_Unlock(RLGL.batchVertexBuffers[1]);
            }
        }
        // Normals
        if (RLGL.batchVertexBuffers[2]) {
            if (SUCCEEDED(IDirect3DVertexBuffer9_Lock(RLGL.batchVertexBuffers[2], 0, vCount * 3 * sizeof(float), &pData, D3DLOCK_DISCARD))) {
                memcpy(pData, batch->vertexBuffer[batch->currentBuffer].normals, vCount * 3 * sizeof(float));
                IDirect3DVertexBuffer9_Unlock(RLGL.batchVertexBuffers[2]);
            }
        }
        // Colors
        if (RLGL.batchVertexBuffers[3]) {
            if (SUCCEEDED(IDirect3DVertexBuffer9_Lock(RLGL.batchVertexBuffers[3], 0, vCount * 4 * sizeof(unsigned char), &pData, D3DLOCK_DISCARD))) {
                memcpy(pData, batch->vertexBuffer[batch->currentBuffer].colors, vCount * 4 * sizeof(unsigned char));
                IDirect3DVertexBuffer9_Unlock(RLGL.batchVertexBuffers[3]);
            }
        }
    }

    // Set stream sources (D3D9 equivalent of IASetVertexBuffers)
    IDirect3DDevice9_SetStreamSource(RLGL.device, 0, RLGL.batchVertexBuffers[0], 0, 3*sizeof(float));
    IDirect3DDevice9_SetStreamSource(RLGL.device, 1, RLGL.batchVertexBuffers[1], 0, 2*sizeof(float));
    IDirect3DDevice9_SetStreamSource(RLGL.device, 2, RLGL.batchVertexBuffers[2], 0, 3*sizeof(float));
    IDirect3DDevice9_SetStreamSource(RLGL.device, 3, RLGL.batchVertexBuffers[3], 0, 4*sizeof(unsigned char));

    // Set index buffer
    IDirect3DDevice9_SetIndices(RLGL.device, RLGL.batchIndexBuffer);

    // Set vertex declaration
    if (RLGL.defaultVertexDecl)
        IDirect3DDevice9_SetVertexDeclaration(RLGL.device, RLGL.defaultVertexDecl);

    // Set shader
    unsigned int shaderIdx = RLGL.State.currentShaderId > 0 ? RLGL.State.currentShaderId - 1 : 0;
    if (shaderIdx < RLGL.shaderCount) {
        if (RLGL.shaders[shaderIdx].vertexShader)
            IDirect3DDevice9_SetVertexShader(RLGL.device, RLGL.shaders[shaderIdx].vertexShader);
        if (RLGL.shaders[shaderIdx].pixelShader)
            IDirect3DDevice9_SetPixelShader(RLGL.device, RLGL.shaders[shaderIdx].pixelShader);
    }

    // Update MVP via SetVertexShaderConstantF (registers c0-c3, 4 float4s)
    {
        Matrix matMVP = rlMatrixMultiply(RLGL.State.modelview, RLGL.State.projection);
        rl_float16 mvpFloats = rlMatrixToFloatV(matMVP);
        IDirect3DDevice9_SetVertexShaderConstantF(RLGL.device, 0, mvpFloats.v, 4);
    }

    // Update diffuse color via SetPixelShaderConstantF (register c0, 1 float4)
    {
        float white[4] = { 1.0f, 1.0f, 1.0f, 1.0f };
        IDirect3DDevice9_SetPixelShaderConstantF(RLGL.device, 0, white, 1);
    }

    // Ensure render states are up to date
    if (RLGL.stateDirty) rlUpdateD3D9States();

    // Begin scene
    IDirect3DDevice9_BeginScene(RLGL.device);

    // Issue draw calls
    int vertexOffset = 0;
    for (int i = 0; i < batch->drawCounter; i++) {
        int drawVertexCount = batch->draws[i].vertexCount;
        if (drawVertexCount == 0) { vertexOffset += batch->draws[i].vertexAlignment; continue; }

        // Bind texture for this draw call (D3D9: SetTexture directly, no SRV)
        unsigned int texId = batch->draws[i].textureId;
        if (texId > 0 && texId <= RLGL.textureCount) {
            unsigned int texIdx = texId - 1;
            if (RLGL.textures[texIdx].texture) {
                IDirect3DDevice9_SetTexture(RLGL.device, 0, (IDirect3DBaseTexture9 *)RLGL.textures[texIdx].texture);
            }
        }

        int mode = batch->draws[i].mode;
        if (mode == RL_QUADS) {
            int quadCount = drawVertexCount / 4;
            int indexOffset = (vertexOffset / 4) * 6;
            // DrawIndexedPrimitive: Type, BaseVertexIndex, MinVertexIndex, NumVertices, StartIndex, PrimitiveCount
            IDirect3DDevice9_DrawIndexedPrimitive(RLGL.device, D3DPT_TRIANGLELIST,
                0, 0, vCount, indexOffset, quadCount * 2);
        } else if (mode == RL_TRIANGLES) {
            // DrawPrimitive: Type, StartVertex, PrimitiveCount
            IDirect3DDevice9_DrawPrimitive(RLGL.device, D3DPT_TRIANGLELIST,
                vertexOffset, drawVertexCount / 3);
        } else if (mode == RL_LINES) {
            IDirect3DDevice9_DrawPrimitive(RLGL.device, D3DPT_LINELIST,
                vertexOffset, drawVertexCount / 2);
        }

        vertexOffset += drawVertexCount + batch->draws[i].vertexAlignment;
    }

    // End scene
    IDirect3DDevice9_EndScene(RLGL.device);

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
    if (RLGL.textureCount >= RL_D3D9_MAX_TEXTURES) {
        TRACELOG(RL_LOG_WARNING, "TEXTURE: Max D3D9 texture limit reached (%d)", RL_D3D9_MAX_TEXTURES);
        return 0;
    }

    // Determine bits-per-pixel for pitch calculation
    int bpp = 0;
    switch (format) {
        case 1: bpp = 8; break;
        case 2: case 3: case 5: case 6: bpp = 16; break;
        case 7: bpp = 32; break;
        case 4: bpp = 24; break;     // RGB8 (will be expanded to BGRA)
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

    // Handle 3-channel formats by expanding to 4-channel, and RGBA->BGRA swizzle
    const void *uploadData = data;
    void *expandedData = NULL;

    if (data != NULL) {
        if (format == 4) {
            // RGB8 -> BGRA8 (expand + swizzle in one pass)
            int pixelCount = width * height;
            expandedData = RL_MALLOC(pixelCount * 4);
            const unsigned char *src = (const unsigned char *)data;
            unsigned char *dst = (unsigned char *)expandedData;
            for (int i = 0; i < pixelCount; i++) {
                dst[4*i+0] = src[3*i+2]; // B
                dst[4*i+1] = src[3*i+1]; // G
                dst[4*i+2] = src[3*i+0]; // R
                dst[4*i+3] = 255;         // A
            }
            uploadData = expandedData;
            bpp = 32;
        }
        else if (format == 7) {
            // RGBA8 -> BGRA8 swizzle
            int pixelCount = width * height;
            expandedData = RL_MALLOC(pixelCount * 4);
            rlSwizzleRGBAtoBGRA((const unsigned char *)data, (unsigned char *)expandedData, pixelCount);
            uploadData = expandedData;
        }
        else if (format == 9) {
            // RGB32F -> RGBA32F (float formats don't need byte swizzle)
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

    D3DFORMAT d3dFormat = rlGetD3D9Format(format);
    int mipLevels = (mipmapCount > 0) ? mipmapCount : 1;

    // Create texture in managed pool (automatic GPU upload + CPU access for Lock)
    IDirect3DTexture9 *texture = NULL;
    HRESULT hr = IDirect3DDevice9_CreateTexture(RLGL.device, width, height, mipLevels,
        0, d3dFormat, D3DPOOL_MANAGED, &texture, NULL);

    if (FAILED(hr) || !texture) {
        if (expandedData) RL_FREE(expandedData);
        TRACELOG(RL_LOG_WARNING, "TEXTURE: Failed to create D3D9 texture (format %d, %dx%d, HRESULT: 0x%08X)", format, width, height, (unsigned int)hr);
        return 0;
    }

    // Upload pixel data via Lock/Unlock
    if (uploadData) {
        D3DLOCKED_RECT locked;
        hr = IDirect3DTexture9_LockRect(texture, 0, &locked, NULL, 0);
        if (SUCCEEDED(hr)) {
            int uploadBpp = bpp;
            // After expansion, formats 4,7 are 32 bpp; use the possibly updated bpp
            if (format == 4 || format == 7) uploadBpp = 32;
            int srcPitch = width * uploadBpp / 8;
            // Copy row by row (locked pitch may differ from source pitch)
            for (int y = 0; y < height; y++) {
                memcpy((unsigned char *)locked.pBits + y * locked.Pitch,
                       (const unsigned char *)uploadData + y * srcPitch, srcPitch);
            }
            IDirect3DTexture9_UnlockRect(texture, 0);
        }
    }

    if (expandedData) RL_FREE(expandedData);

    unsigned int idx = RLGL.textureCount;
    RLGL.textures[idx].texture = texture;
    RLGL.textures[idx].surface = NULL;
    RLGL.textures[idx].width = width;
    RLGL.textures[idx].height = height;
    RLGL.textures[idx].mipmaps = mipLevels;
    RLGL.textures[idx].format = format;
    RLGL.textures[idx].isRenderTarget = false;
    RLGL.textures[idx].isDepth = false;
    RLGL.textureCount++;

    unsigned int id = idx + 1;
    TRACELOG(RL_LOG_INFO, "TEXTURE: [ID %i] D3D9 texture loaded (%ix%i, format: %i)", id, width, height, format);
    return id;
}

unsigned int rlLoadTextureDepth(int width, int height, bool useRenderBuffer)
{
    (void)useRenderBuffer;
    if (!RLGL.device || RLGL.textureCount >= RL_D3D9_MAX_TEXTURES) return 0;

    // D3D9: create a depth stencil surface (not a texture — D3D9 can't natively sample depth)
    IDirect3DSurface9 *depthSurface = NULL;
    HRESULT hr = IDirect3DDevice9_CreateDepthStencilSurface(RLGL.device, width, height,
        D3DFMT_D24S8, D3DMULTISAMPLE_NONE, 0, TRUE, &depthSurface, NULL);

    if (FAILED(hr) || !depthSurface) return 0;

    unsigned int idx = RLGL.textureCount;
    RLGL.textures[idx].texture = NULL;
    RLGL.textures[idx].surface = depthSurface;
    RLGL.textures[idx].width = width;
    RLGL.textures[idx].height = height;
    RLGL.textures[idx].mipmaps = 1;
    RLGL.textures[idx].isRenderTarget = false;
    RLGL.textures[idx].isDepth = true;
    RLGL.textureCount++;

    return idx + 1;
}

unsigned int rlLoadTextureCubemap(const void *data, int size, int format)
{
    (void)data; (void)size; (void)format;
    TRACELOG(RL_LOG_WARNING, "TEXTURE: Cubemap textures not yet implemented for D3D9");
    return 0;
}

void rlUpdateTexture(unsigned int id, int offsetX, int offsetY, int width, int height, int format, const void *pixels)
{
    if (id == 0 || id > RLGL.textureCount || !RLGL.device || !pixels) return;
    unsigned int idx = id - 1;
    if (!RLGL.textures[idx].texture) return;

    // For RGBA8 formats (4, 7), we need BGRA swizzle
    const void *uploadPixels = pixels;
    void *swizzled = NULL;
    int storedFormat = RLGL.textures[idx].format;

    if (storedFormat == 4 || storedFormat == 7) {
        int pixelCount = width * height;
        swizzled = RL_MALLOC(pixelCount * 4);
        rlSwizzleRGBAtoBGRA((const unsigned char *)pixels, (unsigned char *)swizzled, pixelCount);
        uploadPixels = swizzled;
    }

    // Lock a sub-rect and copy data
    RECT rect = { offsetX, offsetY, offsetX + width, offsetY + height };
    D3DLOCKED_RECT locked;
    HRESULT hr = IDirect3DTexture9_LockRect(RLGL.textures[idx].texture, 0, &locked, &rect, 0);
    if (SUCCEEDED(hr)) {
        int bpp = 4; // default for RGBA
        switch (storedFormat) {
            case 1: bpp = 1; break;
            case 2: case 3: case 5: case 6: bpp = 2; break;
            case 7: case 4: bpp = 4; break;
            case 8: bpp = 4; break;
            case 10: bpp = 16; break;
            case 11: bpp = 2; break;
            case 13: bpp = 8; break;
            default: break;
        }
        int srcPitch = width * bpp;
        for (int y = 0; y < height; y++) {
            memcpy((unsigned char *)locked.pBits + y * locked.Pitch,
                   (const unsigned char *)uploadPixels + y * srcPitch, srcPitch);
        }
        IDirect3DTexture9_UnlockRect(RLGL.textures[idx].texture, 0);
    }

    if (swizzled) RL_FREE(swizzled);
}

void rlUpdateTextureRec(unsigned int id, Rectangle rec, int format, const void *pixels)
{
    rlUpdateTexture(id, (int)rec.x, (int)rec.y, (int)rec.width, (int)rec.height, format, pixels);
}

void rlUnloadTexture(unsigned int id)
{
    if (id == 0 || id > RLGL.textureCount) return;
    unsigned int idx = id - 1;
    if (RLGL.textures[idx].surface) { IDirect3DSurface9_Release(RLGL.textures[idx].surface); RLGL.textures[idx].surface = NULL; }
    if (RLGL.textures[idx].texture) { IDirect3DTexture9_Release(RLGL.textures[idx].texture); RLGL.textures[idx].texture = NULL; }
}

void rlGenTextureMipmaps(unsigned int id, int width, int height, int format, int *mipmaps) { (void)id; (void)width; (void)height; (void)format; (void)mipmaps; }

void *rlReadTexturePixels(unsigned int id, int width, int height, int format)
{
    if (id == 0 || id > RLGL.textureCount || !RLGL.device) return NULL;
    unsigned int idx = id - 1;
    if (!RLGL.textures[idx].texture) return NULL;

    // D3DPOOL_MANAGED textures can be locked directly for readback
    D3DLOCKED_RECT locked;
    HRESULT hr = IDirect3DTexture9_LockRect(RLGL.textures[idx].texture, 0, &locked, NULL, D3DLOCK_READONLY);
    if (FAILED(hr)) return NULL;

    int bpp = 4; // default RGBA8
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
    unsigned char *src = (unsigned char *)locked.pBits;
    for (int y = 0; y < height; y++) {
        memcpy(dst + y * rowSize, src + y * locked.Pitch, rowSize);
    }

    IDirect3DTexture9_UnlockRect(RLGL.textures[idx].texture, 0);

    // BGRA -> RGBA swizzle for 8-bit formats
    int storedFormat = RLGL.textures[idx].format;
    if (storedFormat == 4 || storedFormat == 7) {
        int pixelCount = width * height;
        unsigned char *tmp = (unsigned char *)RL_MALLOC(pixelCount * 4);
        rlSwizzleBGRAtoRGBA((unsigned char *)pixels, tmp, pixelCount);
        memcpy(pixels, tmp, pixelCount * 4);
        RL_FREE(tmp);
    }

    return pixels;
}

//----------------------------------------------------------------------------------
// Framebuffer management
//----------------------------------------------------------------------------------
unsigned int rlLoadFramebuffer(void)
{
    if (RLGL.framebufferCount >= RL_D3D9_MAX_FRAMEBUFFERS) return 0;
    unsigned int idx = RLGL.framebufferCount;
    memset(&RLGL.framebuffers[idx], 0, sizeof(rlD3D9Framebuffer));
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
        // Get level 0 surface from color texture for SetRenderTarget
        if (RLGL.textures[texIdx].texture && !RLGL.textures[texIdx].surface) {
            // Need to create as render target texture
            if (!RLGL.textures[texIdx].isRenderTarget && RLGL.device) {
                // Re-create texture as render target in DEFAULT pool
                int w = RLGL.textures[texIdx].width;
                int h = RLGL.textures[texIdx].height;
                D3DFORMAT fmt = rlGetD3D9Format(RLGL.textures[texIdx].format);

                IDirect3DTexture9 *rtTex = NULL;
                HRESULT hr = IDirect3DDevice9_CreateTexture(RLGL.device, w, h, 1,
                    D3DUSAGE_RENDERTARGET, fmt, D3DPOOL_DEFAULT, &rtTex, NULL);
                if (SUCCEEDED(hr) && rtTex) {
                    IDirect3DTexture9_Release(RLGL.textures[texIdx].texture);
                    RLGL.textures[texIdx].texture = rtTex;
                    RLGL.textures[texIdx].isRenderTarget = true;
                }
            }
            IDirect3DTexture9_GetSurfaceLevel(RLGL.textures[texIdx].texture, 0, &RLGL.textures[texIdx].surface);
        }
        RLGL.framebuffers[fboIdx].colorSurface = RLGL.textures[texIdx].surface;
    } else if (attachType == RL_ATTACHMENT_DEPTH) {
        RLGL.framebuffers[fboIdx].depthTextureId = texId;
        RLGL.framebuffers[fboIdx].depthSurface = RLGL.textures[texIdx].surface;
    }
}

bool rlFramebufferComplete(unsigned int id) {
    if (id == 0 || id > RLGL.framebufferCount) return false;
    unsigned int idx = id - 1;
    return (RLGL.framebuffers[idx].colorSurface != NULL);
}

void rlUnloadFramebuffer(unsigned int id) {
    if (id == 0 || id > RLGL.framebufferCount) return;
    // Surfaces are owned by the textures, not the framebuffer
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
    if (!RLGL.device || RLGL.bufferCount >= RL_D3D9_MAX_BUFFERS) return 0;
    unsigned int idx = RLGL.bufferCount;

    DWORD usage = dynamic ? (D3DUSAGE_DYNAMIC | D3DUSAGE_WRITEONLY) : D3DUSAGE_WRITEONLY;
    D3DPOOL pool = dynamic ? D3DPOOL_DEFAULT : D3DPOOL_MANAGED;

    IDirect3DVertexBuffer9 *vb = NULL;
    HRESULT hr = IDirect3DDevice9_CreateVertexBuffer(RLGL.device, size, usage, 0, pool, &vb, NULL);
    if (FAILED(hr) || !vb) return 0;

    // Fill with initial data
    if (buffer && vb) {
        void *pData;
        if (SUCCEEDED(IDirect3DVertexBuffer9_Lock(vb, 0, size, &pData, dynamic ? D3DLOCK_DISCARD : 0))) {
            memcpy(pData, buffer, size);
            IDirect3DVertexBuffer9_Unlock(vb);
        }
    }

    RLGL.buffers[idx].vb = vb;
    RLGL.buffers[idx].ib = NULL;
    RLGL.buffers[idx].size = size;
    RLGL.bufferCount++;
    return idx + 1;
}

unsigned int rlLoadVertexBufferElement(const void *buffer, int size, bool dynamic)
{
    if (!RLGL.device || RLGL.bufferCount >= RL_D3D9_MAX_BUFFERS) return 0;
    unsigned int idx = RLGL.bufferCount;

    DWORD usage = dynamic ? (D3DUSAGE_DYNAMIC | D3DUSAGE_WRITEONLY) : D3DUSAGE_WRITEONLY;
    D3DPOOL pool = dynamic ? D3DPOOL_DEFAULT : D3DPOOL_MANAGED;

    IDirect3DIndexBuffer9 *ib = NULL;
    HRESULT hr = IDirect3DDevice9_CreateIndexBuffer(RLGL.device, size, usage, D3DFMT_INDEX16, pool, &ib, NULL);
    if (FAILED(hr) || !ib) return 0;

    if (buffer && ib) {
        void *pData;
        if (SUCCEEDED(IDirect3DIndexBuffer9_Lock(ib, 0, size, &pData, dynamic ? D3DLOCK_DISCARD : 0))) {
            memcpy(pData, buffer, size);
            IDirect3DIndexBuffer9_Unlock(ib);
        }
    }

    RLGL.buffers[idx].vb = NULL;
    RLGL.buffers[idx].ib = ib;
    RLGL.buffers[idx].size = size;
    RLGL.bufferCount++;
    return idx + 1;
}

void rlUpdateVertexBuffer(unsigned int bufferId, const void *data, int dataSize, int offset)
{
    if (bufferId == 0 || bufferId > RLGL.bufferCount || !RLGL.device || !data) return;
    unsigned int idx = bufferId - 1;

    if (RLGL.buffers[idx].vb) {
        void *pData;
        if (SUCCEEDED(IDirect3DVertexBuffer9_Lock(RLGL.buffers[idx].vb, 0, 0, &pData, D3DLOCK_DISCARD))) {
            memcpy((unsigned char *)pData + offset, data, dataSize);
            IDirect3DVertexBuffer9_Unlock(RLGL.buffers[idx].vb);
        }
    } else if (RLGL.buffers[idx].ib) {
        void *pData;
        if (SUCCEEDED(IDirect3DIndexBuffer9_Lock(RLGL.buffers[idx].ib, 0, 0, &pData, D3DLOCK_DISCARD))) {
            memcpy((unsigned char *)pData + offset, data, dataSize);
            IDirect3DIndexBuffer9_Unlock(RLGL.buffers[idx].ib);
        }
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
    if (RLGL.buffers[idx].vb) { IDirect3DVertexBuffer9_Release(RLGL.buffers[idx].vb); RLGL.buffers[idx].vb = NULL; }
    if (RLGL.buffers[idx].ib) { IDirect3DIndexBuffer9_Release(RLGL.buffers[idx].ib); RLGL.buffers[idx].ib = NULL; }
}

void rlBindVertexArray(unsigned int vaoId) { (void)vaoId; }
void rlEnableVertexArray(unsigned int vaoId) { (void)vaoId; }
void rlDisableVertexArray(void) { }
void rlEnableVertexBuffer(unsigned int id) {
    if (id == 0 || id > RLGL.bufferCount || !RLGL.device) return;
    unsigned int idx = id - 1;
    if (RLGL.buffers[idx].vb) {
        UINT stride = RLGL.buffers[idx].size;
        IDirect3DDevice9_SetStreamSource(RLGL.device, 0, RLGL.buffers[idx].vb, 0, stride);
    }
}
void rlDisableVertexBuffer(void) { }
void rlEnableVertexBufferElement(unsigned int id) {
    if (id == 0 || id > RLGL.bufferCount || !RLGL.device) return;
    unsigned int idx = id - 1;
    if (RLGL.buffers[idx].ib)
        IDirect3DDevice9_SetIndices(RLGL.device, RLGL.buffers[idx].ib);
}
void rlDisableVertexBufferElement(void) { }
void rlEnableVertexAttribute(unsigned int index) { (void)index; }
void rlDisableVertexAttribute(unsigned int index) { (void)index; }

void rlDrawVertexArray(int offset, int count) {
    if (!RLGL.device) return;
    IDirect3DDevice9_BeginScene(RLGL.device);
    IDirect3DDevice9_DrawPrimitive(RLGL.device, D3DPT_TRIANGLELIST, offset, count / 3);
    IDirect3DDevice9_EndScene(RLGL.device);
}

void rlDrawVertexArrayElements(int offset, int count, const void *buffer) {
    (void)buffer;
    if (!RLGL.device) return;
    IDirect3DDevice9_BeginScene(RLGL.device);
    IDirect3DDevice9_DrawIndexedPrimitive(RLGL.device, D3DPT_TRIANGLELIST, 0, 0, count, offset, count / 3);
    IDirect3DDevice9_EndScene(RLGL.device);
}

void rlDrawVertexArrayInstanced(int offset, int count, int instances) {
    (void)offset; (void)count; (void)instances;
    TRACELOG(RL_LOG_WARNING, "D3D9: Instanced drawing not supported in this backend");
}

void rlDrawVertexArrayElementsInstanced(int offset, int count, const void *buffer, int instances) {
    (void)offset; (void)count; (void)buffer; (void)instances;
    TRACELOG(RL_LOG_WARNING, "D3D9: Instanced drawing not supported in this backend");
}

//----------------------------------------------------------------------------------
// Shader management
//----------------------------------------------------------------------------------
unsigned int rlCompileShader(const char *shaderCode, int type)
{
    if (!RLGL.device || RLGL.shaderCount >= RL_D3D9_MAX_SHADERS) return 0;

    // Shader Model 3.0 targets
    const char *target = (type == RL_VERTEX_SHADER) ? "vs_3_0" : "ps_3_0";
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
        if (RLGL.shaderCount < RL_D3D9_MAX_SHADERS) {
            RLGL.shaders[RLGL.shaderCount].vsBlob = blob;
            IDirect3DDevice9_CreateVertexShader(RLGL.device,
                (const DWORD *)ID3D10Blob_GetBufferPointer(blob),
                &RLGL.shaders[RLGL.shaderCount].vertexShader);
        }
    } else {
        if (RLGL.shaderCount < RL_D3D9_MAX_SHADERS) {
            RLGL.shaders[RLGL.shaderCount].psBlob = blob;
            IDirect3DDevice9_CreatePixelShader(RLGL.device,
                (const DWORD *)ID3D10Blob_GetBufferPointer(blob),
                &RLGL.shaders[RLGL.shaderCount].pixelShader);
        }
    }

    return id;
}

unsigned int rlLoadShaderCode(const char *vsCode, const char *fsCode)
{
    if (!vsCode) vsCode = defaultVShaderHLSL;
    if (!fsCode) fsCode = defaultPShaderHLSL;

    if (RLGL.shaderCount >= RL_D3D9_MAX_SHADERS) return 0;
    unsigned int idx = RLGL.shaderCount;

    // Compile vertex shader (SM 3.0)
    ID3DBlob *vsBlob = NULL, *psBlob = NULL, *errors = NULL;
    HRESULT hr = D3DCompile(vsCode, strlen(vsCode), NULL, NULL, NULL, "VSMain", "vs_3_0",
        D3DCOMPILE_OPTIMIZATION_LEVEL3 | D3DCOMPILE_PACK_MATRIX_COLUMN_MAJOR, 0, &vsBlob, &errors);
    if (FAILED(hr)) {
        if (errors) { TRACELOG(RL_LOG_WARNING, "SHADER: VS compile error: %s", (char *)ID3D10Blob_GetBufferPointer(errors)); ID3D10Blob_Release(errors); }
        return RLGL.State.defaultShaderId;
    }
    if (errors) ID3D10Blob_Release(errors);
    errors = NULL;

    // Compile pixel shader (SM 3.0)
    hr = D3DCompile(fsCode, strlen(fsCode), NULL, NULL, NULL, "PSMain", "ps_3_0",
        D3DCOMPILE_OPTIMIZATION_LEVEL3 | D3DCOMPILE_PACK_MATRIX_COLUMN_MAJOR, 0, &psBlob, &errors);
    if (FAILED(hr)) {
        if (errors) { TRACELOG(RL_LOG_WARNING, "SHADER: PS compile error: %s", (char *)ID3D10Blob_GetBufferPointer(errors)); ID3D10Blob_Release(errors); }
        ID3D10Blob_Release(vsBlob);
        return RLGL.State.defaultShaderId;
    }
    if (errors) ID3D10Blob_Release(errors);

    // Create D3D9 shaders (takes DWORD* bytecode, no size parameter)
    IDirect3DDevice9_CreateVertexShader(RLGL.device,
        (const DWORD *)ID3D10Blob_GetBufferPointer(vsBlob),
        &RLGL.shaders[idx].vertexShader);
    IDirect3DDevice9_CreatePixelShader(RLGL.device,
        (const DWORD *)ID3D10Blob_GetBufferPointer(psBlob),
        &RLGL.shaders[idx].pixelShader);

    // Create vertex declaration (D3D9 equivalent of input layout)
    D3DVERTEXELEMENT9 layout[] = {
        { 0, 0, D3DDECLTYPE_FLOAT3,   D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_POSITION, 0 },
        { 1, 0, D3DDECLTYPE_FLOAT2,   D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_TEXCOORD, 0 },
        { 2, 0, D3DDECLTYPE_FLOAT3,   D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_NORMAL,   0 },
        { 3, 0, D3DDECLTYPE_UBYTE4N,  D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_COLOR,    0 },
        D3DDECL_END()
    };
    IDirect3DDevice9_CreateVertexDeclaration(RLGL.device, layout, &RLGL.shaders[idx].vertexDecl);

    RLGL.shaders[idx].vsBlob = vsBlob;
    RLGL.shaders[idx].psBlob = psBlob;
    RLGL.shaderCount++;

    unsigned int id = idx + 1;
    TRACELOG(RL_LOG_INFO, "SHADER: [ID %i] D3D9 shader loaded successfully", id);
    return id;
}

unsigned int rlLoadShaderProgram(unsigned int vShaderId, unsigned int fShaderId) {
    (void)vShaderId; (void)fShaderId;
    return 0; // Not used in D3D9 path
}

void rlUnloadShaderProgram(unsigned int id) {
    if (id == 0 || id > RLGL.shaderCount) return;
    unsigned int idx = id - 1;
    if (RLGL.shaders[idx].vertexShader) { IDirect3DVertexShader9_Release(RLGL.shaders[idx].vertexShader); RLGL.shaders[idx].vertexShader = NULL; }
    if (RLGL.shaders[idx].pixelShader) { IDirect3DPixelShader9_Release(RLGL.shaders[idx].pixelShader); RLGL.shaders[idx].pixelShader = NULL; }
    if (RLGL.shaders[idx].vertexDecl) { IDirect3DVertexDeclaration9_Release(RLGL.shaders[idx].vertexDecl); RLGL.shaders[idx].vertexDecl = NULL; }
    if (RLGL.shaders[idx].vsBlob) { ID3D10Blob_Release(RLGL.shaders[idx].vsBlob); RLGL.shaders[idx].vsBlob = NULL; }
    if (RLGL.shaders[idx].psBlob) { ID3D10Blob_Release(RLGL.shaders[idx].psBlob); RLGL.shaders[idx].psBlob = NULL; }
}

int rlGetLocationUniform(unsigned int shaderId, const char *uniformName) {
    (void)shaderId; (void)uniformName;
    return -1; // D3D9 uses shader constant registers, not named uniform locations
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

// Compute shader stubs (not available in D3D9)
unsigned int rlLoadComputeShaderProgram(unsigned int shaderId) {
    (void)shaderId;
    TRACELOG(RL_LOG_WARNING, "SHADER: Compute shaders not supported in D3D9");
    return 0;
}
void rlComputeShaderDispatch(unsigned int groupX, unsigned int groupY, unsigned int groupZ) { (void)groupX; (void)groupY; (void)groupZ; }

// SSBO stubs (not available in D3D9)
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
        TRACELOG(RL_LOG_INFO, "SHADER: [ID %i] Default D3D9 shader loaded successfully", RLGL.State.defaultShaderId);
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

#endif // GRAPHICS_API_DIRECT3D9
