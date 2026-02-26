/**********************************************************************************************
*
*   rl_backend_metal - Metal rendering backend for rlgl
*
*   DESCRIPTION:
*       Complete Metal implementation of the rlgl rendering API.
*       Provides all functions required by rlgl.h when GRAPHICS_API_METAL is defined.
*       Uses Objective-C for Metal API calls (must be compiled as .m on Apple platforms).
*
*   COMPILATION:
*       Only compiled when GRAPHICS_API_METAL is defined (Apple platforms only).
*       Requires: Metal.framework, MetalKit.framework, QuartzCore.framework
*
*   LICENSE: zlib/libpng (same as raylib)
*
**********************************************************************************************/

#if defined(GRAPHICS_API_METAL)

#include "rlgl.h"
#include "rl_backend.h"

#import <Metal/Metal.h>
#import <MetalKit/MetalKit.h>
#import <QuartzCore/CAMetalLayer.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

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
#define RL_MTL_MAX_TEXTURES       4096
#define RL_MTL_MAX_SHADERS        256
#define RL_MTL_MAX_BUFFERS        4096
#define RL_MTL_MAX_FRAMEBUFFERS   256
#define RL_MTL_MAX_FRAMES_IN_FLIGHT 3

//----------------------------------------------------------------------------------
// Types - Resource tracking
//----------------------------------------------------------------------------------
typedef struct {
    id<MTLTexture> texture;
    id<MTLSamplerState> sampler;
    int width, height, format;
    bool isCubemap;
    bool isDepth;
} rlMtlTexture;

typedef struct {
    id<MTLRenderPipelineState> pipelineState;
    id<MTLDepthStencilState> depthStencilState;
    id<MTLFunction> vertexFunction;
    id<MTLFunction> fragmentFunction;
} rlMtlShader;

typedef struct {
    id<MTLBuffer> buffer;
    int size;
    bool dynamic;
    bool isIndex;
} rlMtlBuffer;

typedef struct {
    id<MTLTexture> colorTexture;
    id<MTLTexture> depthTexture;
    bool complete;
} rlMtlFramebuffer;

//----------------------------------------------------------------------------------
// Uniform structure for push via buffer (per-draw constants)
//----------------------------------------------------------------------------------
typedef struct RL_ALIGN(16) {
    float mvp[16];
    float colDiffuse[4];
} rlMtlUniforms;

//----------------------------------------------------------------------------------
// Internal rlgl state for Metal
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

    // Metal core objects
    id<MTLDevice> device;
    id<MTLCommandQueue> commandQueue;
    id<MTLLibrary> defaultLibrary;

    // Current frame rendering
    id<MTLCommandBuffer> currentCommandBuffer;
    id<MTLRenderCommandEncoder> currentEncoder;
    dispatch_semaphore_t frameSemaphore;

    // Default pipeline state
    id<MTLRenderPipelineState> defaultPipelineState;
    id<MTLDepthStencilState> defaultDepthStencilState;
    id<MTLSamplerState> defaultSamplerState;

    // Batch GPU buffers (triple buffered)
    id<MTLBuffer> batchVertexBuffers[4];   // pos, texcoord, normal, color
    id<MTLBuffer> batchIndexBuffer;
    int batchBufferSize;
    int currentFrameIndex;

    // Render state
    bool pipelineDirty;
    CAMetalLayer *metalLayer;

    // Resource tracking
    rlMtlTexture textures[RL_MTL_MAX_TEXTURES];
    unsigned int textureCount;

    rlMtlShader shaders[RL_MTL_MAX_SHADERS];
    unsigned int shaderCount;

    rlMtlBuffer buffers[RL_MTL_MAX_BUFFERS];
    unsigned int bufferCount;

    rlMtlFramebuffer framebuffers[RL_MTL_MAX_FRAMEBUFFERS];
    unsigned int framebufferCount;

    unsigned int activeFramebuffer;

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
static MTLPixelFormat rlGetMTLPixelFormat(int format);

//----------------------------------------------------------------------------------
// Global state
//----------------------------------------------------------------------------------
static double rlCullDistanceNear = RL_CULL_DISTANCE_NEAR;
static double rlCullDistanceFar = RL_CULL_DISTANCE_FAR;
static rlglData RLGL = { 0 };

//----------------------------------------------------------------------------------
// Default MSL Shaders
//----------------------------------------------------------------------------------
static NSString *defaultMetalShaderSource = @
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "\n"
    "struct VertexIn {\n"
    "    float3 position [[attribute(0)]];\n"
    "    float2 texcoord [[attribute(1)]];\n"
    "    float3 normal   [[attribute(2)]];\n"
    "    float4 color    [[attribute(3)]];\n"
    "};\n"
    "\n"
    "struct VertexOut {\n"
    "    float4 position [[position]];\n"
    "    float2 texcoord;\n"
    "    float4 color;\n"
    "};\n"
    "\n"
    "struct Uniforms {\n"
    "    float4x4 mvp;\n"
    "    float4 colDiffuse;\n"
    "};\n"
    "\n"
    "vertex VertexOut vertexShader(VertexIn in [[stage_in]],\n"
    "                              constant Uniforms &uniforms [[buffer(4)]]) {\n"
    "    VertexOut out;\n"
    "    out.position = uniforms.mvp * float4(in.position, 1.0);\n"
    "    out.texcoord = in.texcoord;\n"
    "    out.color = in.color;\n"
    "    return out;\n"
    "}\n"
    "\n"
    "fragment float4 fragmentShader(VertexOut in [[stage_in]],\n"
    "                                texture2d<float> texture0 [[texture(0)]],\n"
    "                                sampler sampler0 [[sampler(0)]],\n"
    "                                constant Uniforms &uniforms [[buffer(4)]]) {\n"
    "    float4 texColor = texture0.sample(sampler0, in.texcoord);\n"
    "    return texColor * uniforms.colDiffuse * in.color;\n"
    "}\n";

//----------------------------------------------------------------------------------
// Helper: Get MTLPixelFormat from rlgl format
//----------------------------------------------------------------------------------
static MTLPixelFormat rlGetMTLPixelFormat(int format)
{
    switch (format) {
        case 1:  return MTLPixelFormatR8Unorm;
        case 2:  return MTLPixelFormatRG8Unorm;
        case 3:  return MTLPixelFormatB5G6R5Unorm;
        case 4:  return MTLPixelFormatRGBA8Unorm;  // No RGB8 in Metal
        case 5:  return MTLPixelFormatBGR5A1Unorm;
        case 6:  return MTLPixelFormatABGR4Unorm;
        case 7:  return MTLPixelFormatRGBA8Unorm;
        case 8:  return MTLPixelFormatR32Float;
        case 9:  return MTLPixelFormatRGBA32Float; // No RGB32, use RGBA32
        case 10: return MTLPixelFormatRGBA32Float;
        case 11: return MTLPixelFormatR16Float;
        case 12: return MTLPixelFormatRGBA16Float;
        case 13: return MTLPixelFormatRGBA16Float;
#if !TARGET_OS_IPHONE
        case 14: return MTLPixelFormatBC1_RGBA;
        case 15: return MTLPixelFormatBC1_RGBA;
        case 16: return MTLPixelFormatBC2_RGBA;
        case 17: return MTLPixelFormatBC3_RGBA;
#endif
        default: return MTLPixelFormatRGBA8Unorm;
    }
}

//----------------------------------------------------------------------------------
// Matrix operations (identical logic to other backends)
//----------------------------------------------------------------------------------
void rlMatrixMode(int mode)
{
    if (mode == RL_PROJECTION) RLGL.State.currentMatrix = &RLGL.State.projection;
    else if (mode == RL_MODELVIEW) RLGL.State.currentMatrix = &RLGL.State.modelview;
    RLGL.State.currentMatrixMode = mode;
}

void rlPushMatrix(void)
{
    if (RLGL.State.stackCounter >= RL_MAX_MATRIX_STACK_SIZE) return;
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
    if (RLGL.currentEncoder) {
        MTLViewport vp = { (double)x, (double)y, (double)width, (double)height, 0.0, 1.0 };
        [RLGL.currentEncoder setViewport:vp];
    }
}

void rlSetClipPlanes(double nearPlane, double farPlane)
{
    rlCullDistanceNear = nearPlane;
    rlCullDistanceFar = farPlane;
}

double rlGetCullDistanceNear(void) { return rlCullDistanceNear; }
double rlGetCullDistanceFar(void) { return rlCullDistanceFar; }

//----------------------------------------------------------------------------------
// Vertex-level operations (CPU batch accumulation)
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

void rlEnd(void) { RLGL.currentBatch->currentDepth += (1.0f/20000.0f); }

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

void rlActiveTextureSlot(int slot) { (void)slot; }
void rlEnableTexture(unsigned int id) {
    if (id > 0 && id <= RLGL.textureCount && RLGL.currentEncoder)
        [RLGL.currentEncoder setFragmentTexture:RLGL.textures[id-1].texture atIndex:0];
}
void rlDisableTexture(void) { }
void rlEnableTextureCubemap(unsigned int id) { rlEnableTexture(id); }
void rlDisableTextureCubemap(void) { }
void rlTextureParameters(unsigned int id, int param, int value) { (void)id; (void)param; (void)value; }
void rlCubemapParameters(unsigned int id, int param, int value) { (void)id; (void)param; (void)value; }

//----------------------------------------------------------------------------------
// Shader state
//----------------------------------------------------------------------------------
void rlEnableShader(unsigned int id) {
    if (id > 0 && id <= RLGL.shaderCount && RLGL.currentEncoder)
        [RLGL.currentEncoder setRenderPipelineState:RLGL.shaders[id-1].pipelineState];
}
void rlDisableShader(void) { }

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
void rlBindFramebuffer(unsigned int target, unsigned int framebuffer) { (void)target; RLGL.activeFramebuffer = framebuffer; }

//----------------------------------------------------------------------------------
// Render state
//----------------------------------------------------------------------------------
void rlEnableColorBlend(void) { RLGL.State.colorBlendEnabled = true; RLGL.pipelineDirty = true; }
void rlDisableColorBlend(void) { RLGL.State.colorBlendEnabled = false; RLGL.pipelineDirty = true; }
void rlEnableDepthTest(void) { RLGL.State.depthTestEnabled = true; RLGL.pipelineDirty = true; }
void rlDisableDepthTest(void) { RLGL.State.depthTestEnabled = false; RLGL.pipelineDirty = true; }
void rlEnableDepthMask(void) { RLGL.State.depthWriteEnabled = true; RLGL.pipelineDirty = true; }
void rlDisableDepthMask(void) { RLGL.State.depthWriteEnabled = false; RLGL.pipelineDirty = true; }
void rlEnableBackfaceCulling(void) { RLGL.State.backfaceCullingEnabled = true; }
void rlDisableBackfaceCulling(void) { RLGL.State.backfaceCullingEnabled = false; }
void rlColorMask(bool r, bool g, bool b, bool a) { (void)r; (void)g; (void)b; (void)a; }
void rlSetCullFace(int mode) {
    RLGL.State.cullFaceMode = (mode == RL_CULL_FACE_FRONT) ? 1 : 0;
    if (RLGL.currentEncoder) {
        [RLGL.currentEncoder setCullMode:(RLGL.State.backfaceCullingEnabled ?
            (RLGL.State.cullFaceMode == 1 ? MTLCullModeFront : MTLCullModeBack) : MTLCullModeNone)];
    }
}
void rlEnableScissorTest(void) { RLGL.State.scissorTestEnabled = true; }
void rlDisableScissorTest(void) { RLGL.State.scissorTestEnabled = false; }
void rlScissor(int x, int y, int width, int height) {
    if (RLGL.currentEncoder) {
        MTLScissorRect rect = { (NSUInteger)x, (NSUInteger)y, (NSUInteger)width, (NSUInteger)height };
        [RLGL.currentEncoder setScissorRect:rect];
    }
}
void rlEnableWireMode(void) {
    RLGL.State.wireMode = true;
    if (RLGL.currentEncoder) [RLGL.currentEncoder setTriangleFillMode:MTLTriangleFillModeLines];
}
void rlEnablePointMode(void) { rlEnableWireMode(); }
void rlDisableWireMode(void) {
    RLGL.State.wireMode = false;
    if (RLGL.currentEncoder) [RLGL.currentEncoder setTriangleFillMode:MTLTriangleFillModeFill];
}
void rlSetLineWidth(float width) { RLGL.State.lineWidth = width; }
float rlGetLineWidth(void) { return RLGL.State.lineWidth; }
void rlEnableSmoothLines(void) { }
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
    // Metal clears via MTLRenderPassDescriptor loadAction
}

void rlCheckErrors(void) { }

void rlSetBlendMode(int mode) {
    if (RLGL.State.currentBlendMode != mode) {
        rlDrawRenderBatch(RLGL.currentBatch);
        RLGL.State.currentBlendMode = mode;
        RLGL.pipelineDirty = true;
    }
}

void rlSetBlendFactors(int glSrcFactor, int glDstFactor, int glEquation) { (void)glSrcFactor; (void)glDstFactor; (void)glEquation; }
void rlSetBlendFactorsSeparate(int glSrcRGB, int glDstRGB, int glSrcAlpha, int glDstAlpha, int glEqRGB, int glEqAlpha) {
    (void)glSrcRGB; (void)glDstRGB; (void)glSrcAlpha; (void)glDstAlpha; (void)glEqRGB; (void)glEqAlpha;
}

//----------------------------------------------------------------------------------
// rlgl initialization
//----------------------------------------------------------------------------------
void rlglInit(int width, int height, bool headless)
{
    (void)headless;
    @autoreleasepool {
        // Create Metal device
        RLGL.device = MTLCreateSystemDefaultDevice();
        if (!RLGL.device) {
            TRACELOG(RL_LOG_ERROR, "METAL: Failed to create system default device");
            return;
        }
        TRACELOG(RL_LOG_INFO, "METAL: GPU: %s", [[RLGL.device name] UTF8String]);

        // Create command queue
        RLGL.commandQueue = [RLGL.device newCommandQueue];
        RLGL.frameSemaphore = dispatch_semaphore_create(RL_MTL_MAX_FRAMES_IN_FLIGHT);

        // Compile default shader library
        NSError *error = nil;
        RLGL.defaultLibrary = [RLGL.device newLibraryWithSource:defaultMetalShaderSource options:nil error:&error];
        if (error) {
            TRACELOG(RL_LOG_ERROR, "METAL: Shader compile error: %s", [[error localizedDescription] UTF8String]);
        }

        // Create default sampler
        MTLSamplerDescriptor *samplerDesc = [[MTLSamplerDescriptor alloc] init];
        samplerDesc.minFilter = MTLSamplerMinMagFilterLinear;
        samplerDesc.magFilter = MTLSamplerMinMagFilterLinear;
        samplerDesc.sAddressMode = MTLSamplerAddressModeRepeat;
        samplerDesc.tAddressMode = MTLSamplerAddressModeRepeat;
        RLGL.defaultSamplerState = [RLGL.device newSamplerStateWithDescriptor:samplerDesc];

        // Create default depth stencil state
        MTLDepthStencilDescriptor *dsDesc = [[MTLDepthStencilDescriptor alloc] init];
        dsDesc.depthCompareFunction = MTLCompareFunctionLessEqual;
        dsDesc.depthWriteEnabled = YES;
        RLGL.defaultDepthStencilState = [RLGL.device newDepthStencilStateWithDescriptor:dsDesc];

        // Create default pipeline state
        if (RLGL.defaultLibrary) {
            id<MTLFunction> vertexFunc = [RLGL.defaultLibrary newFunctionWithName:@"vertexShader"];
            id<MTLFunction> fragFunc = [RLGL.defaultLibrary newFunctionWithName:@"fragmentShader"];

            if (vertexFunc && fragFunc) {
                MTLRenderPipelineDescriptor *pipeDesc = [[MTLRenderPipelineDescriptor alloc] init];
                pipeDesc.vertexFunction = vertexFunc;
                pipeDesc.fragmentFunction = fragFunc;
                pipeDesc.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;
                pipeDesc.depthAttachmentPixelFormat = MTLPixelFormatDepth32Float;

                // Alpha blending
                pipeDesc.colorAttachments[0].blendingEnabled = YES;
                pipeDesc.colorAttachments[0].rgbBlendOperation = MTLBlendOperationAdd;
                pipeDesc.colorAttachments[0].alphaBlendOperation = MTLBlendOperationAdd;
                pipeDesc.colorAttachments[0].sourceRGBBlendFactor = MTLBlendFactorSourceAlpha;
                pipeDesc.colorAttachments[0].sourceAlphaBlendFactor = MTLBlendFactorOne;
                pipeDesc.colorAttachments[0].destinationRGBBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
                pipeDesc.colorAttachments[0].destinationAlphaBlendFactor = MTLBlendFactorOneMinusSourceAlpha;

                // Vertex descriptor
                MTLVertexDescriptor *vertexDesc = [[MTLVertexDescriptor alloc] init];
                // position
                vertexDesc.attributes[0].format = MTLVertexFormatFloat3;
                vertexDesc.attributes[0].offset = 0;
                vertexDesc.attributes[0].bufferIndex = 0;
                vertexDesc.layouts[0].stride = 3 * sizeof(float);
                // texcoord
                vertexDesc.attributes[1].format = MTLVertexFormatFloat2;
                vertexDesc.attributes[1].offset = 0;
                vertexDesc.attributes[1].bufferIndex = 1;
                vertexDesc.layouts[1].stride = 2 * sizeof(float);
                // normal
                vertexDesc.attributes[2].format = MTLVertexFormatFloat3;
                vertexDesc.attributes[2].offset = 0;
                vertexDesc.attributes[2].bufferIndex = 2;
                vertexDesc.layouts[2].stride = 3 * sizeof(float);
                // color
                vertexDesc.attributes[3].format = MTLVertexFormatUChar4Normalized;
                vertexDesc.attributes[3].offset = 0;
                vertexDesc.attributes[3].bufferIndex = 3;
                vertexDesc.layouts[3].stride = 4 * sizeof(unsigned char);

                pipeDesc.vertexDescriptor = vertexDesc;

                NSError *pipeError = nil;
                RLGL.defaultPipelineState = [RLGL.device newRenderPipelineStateWithDescriptor:pipeDesc error:&pipeError];
                if (pipeError)
                    TRACELOG(RL_LOG_ERROR, "METAL: Pipeline state error: %s", [[pipeError localizedDescription] UTF8String]);
            }
        }

        // Init default texture
        {
            unsigned char pixels[4] = { 255, 255, 255, 255 };
            RLGL.State.defaultTextureId = rlLoadTexture(pixels, 1, 1, 7, 1);
        }

        // Init default shader tracking
        rlLoadShaderDefault();
        RLGL.State.currentShaderId = RLGL.State.defaultShaderId;
        RLGL.State.currentShaderLocs = RLGL.State.defaultShaderLocs;

        // Init batch
        RLGL.defaultBatch = rlLoadRenderBatch(RL_DEFAULT_BATCH_BUFFERS, RL_DEFAULT_BATCH_BUFFER_ELEMENTS);
        RLGL.currentBatch = &RLGL.defaultBatch;

        // Init matrices
        for (int i = 0; i < RL_MAX_MATRIX_STACK_SIZE; i++) RLGL.State.stack[i] = rlMatrixIdentity();
        RLGL.State.transform = rlMatrixIdentity();
        RLGL.State.projection = rlMatrixIdentity();
        RLGL.State.modelview = rlMatrixIdentity();
        RLGL.State.currentMatrix = &RLGL.State.modelview;

        // Init defaults
        RLGL.State.colorBlendEnabled = true;
        RLGL.State.depthTestEnabled = false;
        RLGL.State.depthWriteEnabled = true;
        RLGL.State.backfaceCullingEnabled = true;
        RLGL.State.lineWidth = 1.0f;

        RLGL.State.framebufferWidth = width;
        RLGL.State.framebufferHeight = height;

        TRACELOG(RL_LOG_INFO, "RLGL: Metal backend initialized successfully");
    }
}

void rlglClose(void)
{
    @autoreleasepool {
        rlUnloadRenderBatch(RLGL.defaultBatch);
        rlUnloadShaderDefault();

        // Release tracked resources
        for (unsigned int i = 0; i < RLGL.textureCount; i++) {
            RLGL.textures[i].texture = nil;
            RLGL.textures[i].sampler = nil;
        }
        for (unsigned int i = 0; i < RLGL.bufferCount; i++) {
            RLGL.buffers[i].buffer = nil;
        }

        // Release batch buffers
        for (int i = 0; i < 4; i++) RLGL.batchVertexBuffers[i] = nil;
        RLGL.batchIndexBuffer = nil;

        // Release core objects
        RLGL.defaultPipelineState = nil;
        RLGL.defaultDepthStencilState = nil;
        RLGL.defaultSamplerState = nil;
        RLGL.defaultLibrary = nil;
        RLGL.commandQueue = nil;
        RLGL.device = nil;

        TRACELOG(RL_LOG_INFO, "RLGL: Metal backend closed successfully");
    }
}

void rlLoadExtensions(void *loader)
{
    (void)loader;
    RLGL.ExtSupported.computeShader = true;
    RLGL.ExtSupported.ssbo = true;
    RLGL.ExtSupported.maxAnisotropyLevel = 16.0f;
    RLGL.ExtSupported.maxDepthBits = 32;
}

int rlGetVersion(void) { return RL_METAL; }

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

    // Create GPU buffers (shared memory on Metal)
    @autoreleasepool {
        int totalVertices = bufferElements * 4;
        if (RLGL.device) {
            RLGL.batchVertexBuffers[0] = [RLGL.device newBufferWithLength:totalVertices*3*sizeof(float) options:MTLResourceStorageModeShared];
            RLGL.batchVertexBuffers[1] = [RLGL.device newBufferWithLength:totalVertices*2*sizeof(float) options:MTLResourceStorageModeShared];
            RLGL.batchVertexBuffers[2] = [RLGL.device newBufferWithLength:totalVertices*3*sizeof(float) options:MTLResourceStorageModeShared];
            RLGL.batchVertexBuffers[3] = [RLGL.device newBufferWithLength:totalVertices*4*sizeof(unsigned char) options:MTLResourceStorageModeShared];
            RLGL.batchIndexBuffer = [RLGL.device newBufferWithBytes:batch.vertexBuffer[0].indices
                length:bufferElements*6*sizeof(unsigned int) options:MTLResourceStorageModeShared];
            RLGL.batchBufferSize = totalVertices;
        }
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
    if (!RLGL.device) return;
    if (RLGL.State.vertexCounter <= 0) {
        RLGL.State.vertexCounter = 0;
        batch->currentDepth = -1.0f;
        for (int i = 0; i < RL_DEFAULT_BATCH_DRAWCALLS; i++) {
            batch->draws[i].mode = RL_QUADS;
            batch->draws[i].vertexCount = 0;
            batch->draws[i].textureId = RLGL.State.defaultTextureId;
        }
        batch->drawCounter = 1;
        batch->currentBuffer++;
        if (batch->currentBuffer >= batch->bufferCount) batch->currentBuffer = 0;
        return;
    }

    @autoreleasepool {
        // Upload vertex data to Metal buffers (shared memory = direct memcpy)
        if (RLGL.batchVertexBuffers[0])
            memcpy([RLGL.batchVertexBuffers[0] contents], batch->vertexBuffer[batch->currentBuffer].vertices,
                   RLGL.State.vertexCounter*3*sizeof(float));
        if (RLGL.batchVertexBuffers[1])
            memcpy([RLGL.batchVertexBuffers[1] contents], batch->vertexBuffer[batch->currentBuffer].texcoords,
                   RLGL.State.vertexCounter*2*sizeof(float));
        if (RLGL.batchVertexBuffers[2])
            memcpy([RLGL.batchVertexBuffers[2] contents], batch->vertexBuffer[batch->currentBuffer].normals,
                   RLGL.State.vertexCounter*3*sizeof(float));
        if (RLGL.batchVertexBuffers[3])
            memcpy([RLGL.batchVertexBuffers[3] contents], batch->vertexBuffer[batch->currentBuffer].colors,
                   RLGL.State.vertexCounter*4*sizeof(unsigned char));

        // Record draw commands
        if (RLGL.currentEncoder) {
            // Set vertex buffers
            for (int i = 0; i < 4; i++)
                [RLGL.currentEncoder setVertexBuffer:RLGL.batchVertexBuffers[i] offset:0 atIndex:i];

            // Set pipeline state
            if (RLGL.defaultPipelineState)
                [RLGL.currentEncoder setRenderPipelineState:RLGL.defaultPipelineState];

            // Set depth stencil state
            if (RLGL.defaultDepthStencilState)
                [RLGL.currentEncoder setDepthStencilState:RLGL.defaultDepthStencilState];

            // Set cull mode
            [RLGL.currentEncoder setCullMode:(RLGL.State.backfaceCullingEnabled ?
                (RLGL.State.cullFaceMode == 1 ? MTLCullModeFront : MTLCullModeBack) : MTLCullModeNone)];
            [RLGL.currentEncoder setFrontFacingWinding:MTLWindingCounterClockwise];

            // Set wireframe mode
            [RLGL.currentEncoder setTriangleFillMode:(RLGL.State.wireMode ? MTLTriangleFillModeLines : MTLTriangleFillModeFill)];

            // Set sampler
            [RLGL.currentEncoder setFragmentSamplerState:RLGL.defaultSamplerState atIndex:0];

            // Update uniforms
            rlMtlUniforms uniforms;
            Matrix matMVP = rlMatrixMultiply(RLGL.State.modelview, RLGL.State.projection);
            rl_float16 matData = rlMatrixToFloatV(matMVP);
            memcpy(uniforms.mvp, matData.v, sizeof(float)*16);
            uniforms.colDiffuse[0] = 1.0f; uniforms.colDiffuse[1] = 1.0f;
            uniforms.colDiffuse[2] = 1.0f; uniforms.colDiffuse[3] = 1.0f;
            [RLGL.currentEncoder setVertexBytes:&uniforms length:sizeof(rlMtlUniforms) atIndex:4];
            [RLGL.currentEncoder setFragmentBytes:&uniforms length:sizeof(rlMtlUniforms) atIndex:4];

            // Draw each batch draw call
            for (int i = 0, vertexOffset = 0; i < batch->drawCounter; i++) {
                if (batch->draws[i].vertexCount <= 0) { vertexOffset += batch->draws[i].vertexAlignment; continue; }

                // Bind texture
                unsigned int texId = batch->draws[i].textureId;
                if (texId > 0 && texId <= RLGL.textureCount && RLGL.textures[texId-1].texture)
                    [RLGL.currentEncoder setFragmentTexture:RLGL.textures[texId-1].texture atIndex:0];

                if ((batch->draws[i].mode == RL_LINES) || (batch->draws[i].mode == RL_TRIANGLES)) {
                    MTLPrimitiveType primType = (batch->draws[i].mode == RL_LINES) ?
                        MTLPrimitiveTypeLine : MTLPrimitiveTypeTriangle;
                    [RLGL.currentEncoder drawPrimitives:primType
                        vertexStart:vertexOffset vertexCount:batch->draws[i].vertexCount];
                } else { // QUADS
                    [RLGL.currentEncoder drawIndexedPrimitives:MTLPrimitiveTypeTriangle
                        indexCount:batch->draws[i].vertexCount/4*6
                        indexType:MTLIndexTypeUInt32
                        indexBuffer:RLGL.batchIndexBuffer
                        indexBufferOffset:vertexOffset/4*6*sizeof(unsigned int)];
                }
                vertexOffset += (batch->draws[i].vertexCount + batch->draws[i].vertexAlignment);
            }
        }
    }

    // Reset batch
    RLGL.State.vertexCounter = 0;
    batch->currentDepth = -1.0f;
    for (int i = 0; i < RL_DEFAULT_BATCH_DRAWCALLS; i++) {
        batch->draws[i].mode = RL_QUADS;
        batch->draws[i].vertexCount = 0;
        batch->draws[i].textureId = RLGL.State.defaultTextureId;
    }
    batch->drawCounter = 1;
    batch->currentBuffer++;
    if (batch->currentBuffer >= batch->bufferCount) batch->currentBuffer = 0;
}

void rlSetRenderBatchActive(rlRenderBatch *batch) {
    rlDrawRenderBatch(RLGL.currentBatch);
    RLGL.currentBatch = batch ? batch : &RLGL.defaultBatch;
}

void rlDrawRenderBatchActive(void) { rlDrawRenderBatch(RLGL.currentBatch); }

bool rlCheckRenderBatchLimit(int vCount) {
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
    if (!RLGL.device || RLGL.textureCount >= RL_MTL_MAX_TEXTURES) return 0;

    @autoreleasepool {
        unsigned int idx = RLGL.textureCount;
        MTLPixelFormat pixFormat = rlGetMTLPixelFormat(format);

        MTLTextureDescriptor *texDesc = [[MTLTextureDescriptor alloc] init];
        texDesc.pixelFormat = pixFormat;
        texDesc.width = width;
        texDesc.height = height;
        texDesc.mipmapLevelCount = mipmapCount;
        texDesc.usage = MTLTextureUsageShaderRead;
        texDesc.storageMode = MTLStorageModeShared;

        RLGL.textures[idx].texture = [RLGL.device newTextureWithDescriptor:texDesc];
        if (!RLGL.textures[idx].texture) return 0;

        if (data) {
            int bpp = 4;
            switch (format) { case 1: bpp=1; break; case 2: bpp=2; break; case 3: bpp=2; break; default: bpp=4; break; }
            MTLRegion region = MTLRegionMake2D(0, 0, width, height);
            [RLGL.textures[idx].texture replaceRegion:region mipmapLevel:0 withBytes:data bytesPerRow:width*bpp];
        }

        RLGL.textures[idx].width = width;
        RLGL.textures[idx].height = height;
        RLGL.textures[idx].format = format;
        RLGL.textureCount++;

        return idx + 1;
    }
}

unsigned int rlLoadTextureDepth(int width, int height, bool useRenderBuffer) {
    (void)useRenderBuffer;
    if (!RLGL.device || RLGL.textureCount >= RL_MTL_MAX_TEXTURES) return 0;

    @autoreleasepool {
        unsigned int idx = RLGL.textureCount;
        MTLTextureDescriptor *desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatDepth32Float
            width:width height:height mipmapped:NO];
        desc.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
        desc.storageMode = MTLStorageModePrivate;
        RLGL.textures[idx].texture = [RLGL.device newTextureWithDescriptor:desc];
        RLGL.textures[idx].width = width;
        RLGL.textures[idx].height = height;
        RLGL.textures[idx].isDepth = true;
        RLGL.textureCount++;
        return idx + 1;
    }
}

unsigned int rlLoadTextureCubemap(const void *data, int size, int format, int mipmapCount) {
    (void)data; (void)size; (void)format; (void)mipmapCount; return 0;
}

void rlUpdateTexture(unsigned int id, int offsetX, int offsetY, int width, int height, int format, const void *data) {
    if (id == 0 || id > RLGL.textureCount) return;
    @autoreleasepool {
        unsigned int idx = id - 1;
        int bpp = 4;
        switch (format) { case 1: bpp=1; break; case 2: bpp=2; break; default: bpp=4; break; }
        MTLRegion region = MTLRegionMake2D(offsetX, offsetY, width, height);
        [RLGL.textures[idx].texture replaceRegion:region mipmapLevel:0 withBytes:data bytesPerRow:width*bpp];
    }
}

void rlGetGlTextureFormats(int format, unsigned int *glInternalFormat, unsigned int *glFormat, unsigned int *glType) {
    *glInternalFormat = (unsigned int)rlGetMTLPixelFormat(format);
    *glFormat = *glInternalFormat;
    *glType = 0;
}

const char *rlGetPixelFormatName(unsigned int format) {
    switch (format) {
        case 1: return "GRAYSCALE"; case 2: return "GRAY_ALPHA"; case 3: return "R5G6B5";
        case 4: return "R8G8B8"; case 7: return "R8G8B8A8"; case 8: return "R32";
        default: return "UNKNOWN";
    }
}

void rlUnloadTexture(unsigned int id) {
    if (id == 0 || id > RLGL.textureCount) return;
    RLGL.textures[id-1].texture = nil;
    RLGL.textures[id-1].sampler = nil;
}

void rlGenTextureMipmaps(unsigned int id, int width, int height, int format, int *mipmaps) {
    (void)id; (void)format;
    if (mipmaps) { int max = (width > height) ? width : height; *mipmaps = 1 + (int)floor(log((double)max)/log(2.0)); }
}

void *rlReadTexturePixels(unsigned int id, int width, int height, int format) {
    (void)id; (void)width; (void)height; (void)format; return NULL;
}

unsigned char *rlReadScreenPixels(int width, int height) {
    return (unsigned char *)RL_CALLOC(width*height*4, sizeof(unsigned char));
}

//----------------------------------------------------------------------------------
// Framebuffer management
//----------------------------------------------------------------------------------
unsigned int rlLoadFramebuffer(void) {
    if (RLGL.framebufferCount >= RL_MTL_MAX_FRAMEBUFFERS) return 0;
    RLGL.framebufferCount++;
    return RLGL.framebufferCount;
}

void rlFramebufferAttach(unsigned int fboId, unsigned int texId, int attachType, int texType, int mipLevel) {
    (void)fboId; (void)texId; (void)attachType; (void)texType; (void)mipLevel;
}
bool rlFramebufferComplete(unsigned int id) {
    if (id == 0 || id > RLGL.framebufferCount) return false;
    RLGL.framebuffers[id-1].complete = true; return true;
}
void rlUnloadFramebuffer(unsigned int id) {
    if (id == 0 || id > RLGL.framebufferCount) return;
    RLGL.framebuffers[id-1].colorTexture = nil;
    RLGL.framebuffers[id-1].depthTexture = nil;
}

//----------------------------------------------------------------------------------
// Vertex buffer management
//----------------------------------------------------------------------------------
unsigned int rlLoadVertexArray(void) { return 1; }
void rlUnloadVertexArray(unsigned int vaoId) { (void)vaoId; }
bool rlEnableVertexArray(unsigned int vaoId) { (void)vaoId; return true; }
void rlDisableVertexArray(void) { }

unsigned int rlLoadVertexBuffer(const void *buffer, int size, bool dynamic) {
    if (!RLGL.device || RLGL.bufferCount >= RL_MTL_MAX_BUFFERS) return 0;
    @autoreleasepool {
        unsigned int idx = RLGL.bufferCount;
        if (buffer) RLGL.buffers[idx].buffer = [RLGL.device newBufferWithBytes:buffer length:size options:MTLResourceStorageModeShared];
        else RLGL.buffers[idx].buffer = [RLGL.device newBufferWithLength:size options:MTLResourceStorageModeShared];
        RLGL.buffers[idx].size = size;
        RLGL.buffers[idx].dynamic = dynamic;
        if (!RLGL.buffers[idx].buffer) return 0;
        RLGL.bufferCount++;
        return idx + 1;
    }
}

unsigned int rlLoadVertexBufferElement(const void *buffer, int size, bool dynamic) {
    if (!RLGL.device || RLGL.bufferCount >= RL_MTL_MAX_BUFFERS) return 0;
    @autoreleasepool {
        unsigned int idx = RLGL.bufferCount;
        if (buffer) RLGL.buffers[idx].buffer = [RLGL.device newBufferWithBytes:buffer length:size options:MTLResourceStorageModeShared];
        else RLGL.buffers[idx].buffer = [RLGL.device newBufferWithLength:size options:MTLResourceStorageModeShared];
        RLGL.buffers[idx].size = size;
        RLGL.buffers[idx].dynamic = dynamic;
        RLGL.buffers[idx].isIndex = true;
        if (!RLGL.buffers[idx].buffer) return 0;
        RLGL.bufferCount++;
        return idx + 1;
    }
}

void rlUpdateVertexBuffer(unsigned int id, const void *data, int dataSize, int offset) {
    if (id == 0 || id > RLGL.bufferCount) return;
    memcpy((char *)[RLGL.buffers[id-1].buffer contents] + offset, data, dataSize);
}

void rlUpdateVertexBufferElements(unsigned int id, const void *data, int dataSize, int offset) { rlUpdateVertexBuffer(id, data, dataSize, offset); }

void rlUnloadVertexBuffer(unsigned int vboId) {
    if (vboId == 0 || vboId > RLGL.bufferCount) return;
    RLGL.buffers[vboId-1].buffer = nil;
}

void rlEnableVertexBuffer(unsigned int id) { (void)id; }
void rlDisableVertexBuffer(void) { }
void rlEnableVertexBufferElement(unsigned int id) { (void)id; }
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
    if (RLGL.currentEncoder)
        [RLGL.currentEncoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:offset vertexCount:count];
}

void rlDrawVertexArrayElements(int offset, int count, const void *buffer) {
    (void)buffer; (void)offset; (void)count;
}

void rlDrawVertexArrayInstanced(int offset, int count, int instances) {
    if (RLGL.currentEncoder)
        [RLGL.currentEncoder drawPrimitives:MTLPrimitiveTypeTriangle
            vertexStart:offset vertexCount:count instanceCount:instances];
}

void rlDrawVertexArrayElementsInstanced(int offset, int count, const void *buffer, int instances) {
    (void)offset; (void)count; (void)buffer; (void)instances;
}

//----------------------------------------------------------------------------------
// Shader management
//----------------------------------------------------------------------------------
unsigned int rlCompileShader(const char *shaderCode, int type) { (void)shaderCode; (void)type; return 0; }

unsigned int rlLoadShaderCode(const char *vsCode, const char *fsCode) {
    (void)vsCode; (void)fsCode;
    if (RLGL.shaderCount >= RL_MTL_MAX_SHADERS) return 0;
    RLGL.shaderCount++;
    return RLGL.shaderCount;
}

unsigned int rlLoadShaderProgram(unsigned int vShaderId, unsigned int fShaderId) { (void)vShaderId; (void)fShaderId; return 0; }
void rlUnloadShaderProgram(unsigned int id) { (void)id; }
int rlGetLocationUniform(unsigned int shaderId, const char *uniformName) { (void)shaderId; (void)uniformName; return -1; }
int rlGetLocationAttrib(unsigned int shaderId, const char *attribName) { (void)shaderId; (void)attribName; return -1; }
void rlSetUniform(int locIndex, const void *value, int uniformType, int count) { (void)locIndex; (void)value; (void)uniformType; (void)count; }
void rlSetUniformMatrix(int locIndex, Matrix mat) { (void)locIndex; (void)mat; }
void rlSetUniformMatrices(int locIndex, const Matrix *mat, int count) { (void)locIndex; (void)mat; (void)count; }
void rlSetUniformSampler(int locIndex, unsigned int textureId) { (void)locIndex; (void)textureId; }

void rlSetShader(unsigned int id, int *locs) {
    if (RLGL.State.currentShaderId != id) {
        rlDrawRenderBatch(RLGL.currentBatch);
        RLGL.State.currentShaderId = id;
        RLGL.State.currentShaderLocs = locs;
    }
}

unsigned int rlLoadComputeShaderProgram(unsigned int shaderId) { (void)shaderId; return 0; }
void rlComputeShaderDispatch(unsigned int groupX, unsigned int groupY, unsigned int groupZ) { (void)groupX; (void)groupY; (void)groupZ; }

unsigned int rlLoadShaderBuffer(unsigned int size, const void *data, int usageHint) { (void)size; (void)data; (void)usageHint; return 0; }
void rlUnloadShaderBuffer(unsigned int ssboId) { (void)ssboId; }
void rlUpdateShaderBuffer(unsigned int id, const void *data, unsigned int dataSize, unsigned int offset) { (void)id; (void)data; (void)dataSize; (void)offset; }
void rlBindShaderBuffer(unsigned int id, unsigned int index) { (void)id; (void)index; }
void rlReadShaderBuffer(unsigned int id, void *dest, unsigned int count, unsigned int offset) { (void)id; (void)dest; (void)count; (void)offset; }
void rlCopyShaderBuffer(unsigned int destId, unsigned int srcId, unsigned int destOffset, unsigned int srcOffset, unsigned int count) { (void)destId; (void)srcId; (void)destOffset; (void)srcOffset; (void)count; }
unsigned int rlGetShaderBufferSize(unsigned int id) { (void)id; return 0; }
void rlBindImageTexture(unsigned int id, unsigned int index, int format, bool readonly) { (void)id; (void)index; (void)format; (void)readonly; }

//----------------------------------------------------------------------------------
// Matrix state
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
// Draw helpers
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
        rlNormal3f(0,0,-1); rlTexCoord2f(0,0); rlVertex3f(-1,-1,-1);
        rlNormal3f(0,0,-1); rlTexCoord2f(1,1); rlVertex3f(1,1,-1);
        rlNormal3f(0,0,-1); rlTexCoord2f(1,0); rlVertex3f(1,-1,-1);
        rlNormal3f(0,0,-1); rlTexCoord2f(1,1); rlVertex3f(1,1,-1);
        rlNormal3f(0,0,-1); rlTexCoord2f(0,0); rlVertex3f(-1,-1,-1);
        rlNormal3f(0,0,-1); rlTexCoord2f(0,1); rlVertex3f(-1,1,-1);
        rlNormal3f(0,0,1); rlVertex3f(-1,-1,1); rlNormal3f(0,0,1); rlVertex3f(1,-1,1); rlNormal3f(0,0,1); rlVertex3f(1,1,1);
        rlNormal3f(0,0,1); rlVertex3f(1,1,1); rlNormal3f(0,0,1); rlVertex3f(-1,1,1); rlNormal3f(0,0,1); rlVertex3f(-1,-1,1);
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
// Default shader
//----------------------------------------------------------------------------------
static void rlLoadShaderDefault(void) {
    RLGL.State.defaultShaderLocs = (int *)RL_CALLOC(RL_MAX_SHADER_LOCATIONS, sizeof(int));
    for (int i = 0; i < RL_MAX_SHADER_LOCATIONS; i++) RLGL.State.defaultShaderLocs[i] = -1;
    RLGL.State.defaultShaderId = rlLoadShaderCode(NULL, NULL);
    if (RLGL.State.defaultShaderId > 0) {
        RLGL.State.defaultShaderLocs[RL_SHADER_LOC_VERTEX_POSITION] = 0;
        RLGL.State.defaultShaderLocs[RL_SHADER_LOC_VERTEX_TEXCOORD01] = 1;
        RLGL.State.defaultShaderLocs[RL_SHADER_LOC_VERTEX_COLOR] = 3;
        RLGL.State.defaultShaderLocs[RL_SHADER_LOC_MATRIX_MVP] = 0;
        RLGL.State.defaultShaderLocs[RL_SHADER_LOC_COLOR_DIFFUSE] = 1;
        RLGL.State.defaultShaderLocs[RL_SHADER_LOC_MAP_DIFFUSE] = 0;
    }
}

static void rlUnloadShaderDefault(void) {
    if (RLGL.State.defaultShaderId > 0) rlUnloadShaderProgram(RLGL.State.defaultShaderId);
    RL_FREE(RLGL.State.defaultShaderLocs);
}

//----------------------------------------------------------------------------------
// Math helpers
//----------------------------------------------------------------------------------
static rl_float16 rlMatrixToFloatV(Matrix mat) {
    rl_float16 r = {0};
    r.v[0]=mat.m0; r.v[1]=mat.m1; r.v[2]=mat.m2; r.v[3]=mat.m3;
    r.v[4]=mat.m4; r.v[5]=mat.m5; r.v[6]=mat.m6; r.v[7]=mat.m7;
    r.v[8]=mat.m8; r.v[9]=mat.m9; r.v[10]=mat.m10; r.v[11]=mat.m11;
    r.v[12]=mat.m12; r.v[13]=mat.m13; r.v[14]=mat.m14; r.v[15]=mat.m15;
    return r;
}

static Matrix rlMatrixIdentity(void) { Matrix r = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1}; return r; }

static Matrix rlMatrixMultiply(Matrix left, Matrix right) {
    Matrix r = {0};
    r.m0=left.m0*right.m0+left.m1*right.m4+left.m2*right.m8+left.m3*right.m12;
    r.m1=left.m0*right.m1+left.m1*right.m5+left.m2*right.m9+left.m3*right.m13;
    r.m2=left.m0*right.m2+left.m1*right.m6+left.m2*right.m10+left.m3*right.m14;
    r.m3=left.m0*right.m3+left.m1*right.m7+left.m2*right.m11+left.m3*right.m15;
    r.m4=left.m4*right.m0+left.m5*right.m4+left.m6*right.m8+left.m7*right.m12;
    r.m5=left.m4*right.m1+left.m5*right.m5+left.m6*right.m9+left.m7*right.m13;
    r.m6=left.m4*right.m2+left.m5*right.m6+left.m6*right.m10+left.m7*right.m14;
    r.m7=left.m4*right.m3+left.m5*right.m7+left.m6*right.m11+left.m7*right.m15;
    r.m8=left.m8*right.m0+left.m9*right.m4+left.m10*right.m8+left.m11*right.m12;
    r.m9=left.m8*right.m1+left.m9*right.m5+left.m10*right.m9+left.m11*right.m13;
    r.m10=left.m8*right.m2+left.m9*right.m6+left.m10*right.m10+left.m11*right.m14;
    r.m11=left.m8*right.m3+left.m9*right.m7+left.m10*right.m11+left.m11*right.m15;
    r.m12=left.m12*right.m0+left.m13*right.m4+left.m14*right.m8+left.m15*right.m12;
    r.m13=left.m12*right.m1+left.m13*right.m5+left.m14*right.m9+left.m15*right.m13;
    r.m14=left.m12*right.m2+left.m13*right.m6+left.m14*right.m10+left.m15*right.m14;
    r.m15=left.m12*right.m3+left.m13*right.m7+left.m14*right.m11+left.m15*right.m15;
    return r;
}

static Matrix rlMatrixTranspose(Matrix mat) {
    Matrix r = {0};
    r.m0=mat.m0; r.m1=mat.m4; r.m2=mat.m8; r.m3=mat.m12;
    r.m4=mat.m1; r.m5=mat.m5; r.m6=mat.m9; r.m7=mat.m13;
    r.m8=mat.m2; r.m9=mat.m6; r.m10=mat.m10; r.m11=mat.m14;
    r.m12=mat.m3; r.m13=mat.m7; r.m14=mat.m11; r.m15=mat.m15;
    return r;
}

static Matrix rlMatrixInvert(Matrix mat) {
    Matrix r = {0};
    float a00=mat.m0,a01=mat.m1,a02=mat.m2,a03=mat.m3;
    float a10=mat.m4,a11=mat.m5,a12=mat.m6,a13=mat.m7;
    float a20=mat.m8,a21=mat.m9,a22=mat.m10,a23=mat.m11;
    float a30=mat.m12,a31=mat.m13,a32=mat.m14,a33=mat.m15;
    float b00=a00*a11-a01*a10,b01=a00*a12-a02*a10,b02=a00*a13-a03*a10;
    float b03=a01*a12-a02*a11,b04=a01*a13-a03*a11,b05=a02*a13-a03*a12;
    float b06=a20*a31-a21*a30,b07=a20*a32-a22*a30,b08=a20*a33-a23*a30;
    float b09=a21*a32-a22*a31,b10=a21*a33-a23*a31,b11=a22*a33-a23*a32;
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

static int rlGetPixelDataSize(int width, int height, int format) {
    int bpp = 0;
    switch (format) {
        case 1: bpp=8; break; case 2: case 3: case 5: case 6: bpp=16; break;
        case 7: bpp=32; break; case 4: bpp=24; break; case 8: bpp=32; break;
        case 9: bpp=32*3; break; case 10: bpp=32*4; break;
        case 11: bpp=16; break; case 12: bpp=16*3; break; case 13: bpp=16*4; break;
        default: break;
    }
    return (int)((double)bpp/8.0*width*height);
}

void rlSetHeadlessRenderViewport(int width, int height) { (void)width; (void)height; }
void rlUnsetHeadlessRenderViewport(void) { }

#endif // GRAPHICS_API_METAL
