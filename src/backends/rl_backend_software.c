/**********************************************************************************************
*
*   rl_backend_software - Pure CPU software rendering backend for rlgl
*
*   This implements all rlgl functions using CPU-based rasterization.
*   No GPU API (OpenGL, DirectX, Vulkan, Metal) is required.
*
*   Features:
*     - RGBA8 color buffer + float depth buffer
*     - Triangle rasterization with perspective-correct attribute interpolation
*     - Nearest-neighbor texture sampling
*     - Depth testing (less-equal)
*     - Alpha blending (src-alpha, one-minus-src-alpha)
*     - Scissor test
*     - Backface culling
*     - Line rasterization (Bresenham)
*     - Multi-texture support (CPU-side pixel storage)
*     - Framebuffer objects (render-to-texture)
*
*   Limitations:
*     - No GPU shaders (fixed-function pipeline: texColor * vertexColor * diffuseColor)
*     - No compute shaders / SSBOs
*     - No cubemap textures
*     - No instanced rendering
*     - Performance is limited by CPU speed
*
*   CONFIGURATION:
*       #define GRAPHICS_API_SOFTWARE    Select this backend
*
*   LICENSE: zlib/libpng (same as raylib)
*
**********************************************************************************************/

#include "raylib.h"
#include "rlgl.h"
#include "utils.h"
#include "rl_backend.h"

#if defined(GRAPHICS_API_SOFTWARE)

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

//----------------------------------------------------------------------------------
// Defines and Macros
//----------------------------------------------------------------------------------
#ifndef RL_SW_MAX_TEXTURES
    #define RL_SW_MAX_TEXTURES       4096
#endif
#ifndef RL_SW_MAX_SHADERS
    #define RL_SW_MAX_SHADERS        256
#endif
#ifndef RL_SW_MAX_BUFFERS
    #define RL_SW_MAX_BUFFERS        4096
#endif
#ifndef RL_SW_MAX_FRAMEBUFFERS
    #define RL_SW_MAX_FRAMEBUFFERS   256
#endif

// Minimum/maximum helpers
#define SW_MIN(a, b) ((a) < (b) ? (a) : (b))
#define SW_MAX(a, b) ((a) > (b) ? (a) : (b))
#define SW_CLAMP(x, lo, hi) SW_MAX((lo), SW_MIN((hi), (x)))

//----------------------------------------------------------------------------------
// Types and Structures
//----------------------------------------------------------------------------------

// Tracked CPU-side texture
typedef struct rlSWTexture {
    unsigned char *pixels;              // RGBA8 pixel data (always stored as RGBA8 internally)
    int width, height, mipmaps, format;
    bool isRenderTarget;
    bool isDepth;
} rlSWTexture;

// Tracked software shader (minimal: just an ID, fixed-function pipeline)
typedef struct rlSWShader {
    bool active;
} rlSWShader;

// Tracked buffer (CPU-side)
typedef struct rlSWBuffer {
    unsigned char *data;
    unsigned int size;
    bool isIndex;
} rlSWBuffer;

// Tracked framebuffer
typedef struct rlSWFramebuffer {
    unsigned int colorTextureId;
    unsigned int depthTextureId;
    unsigned char *colorBuffer;         // Points to texture pixels or NULL
    float *depthBuffer;                 // Separate depth buffer
    int width, height;
} rlSWFramebuffer;

// Internal vertex after transformation (clip space / screen space)
typedef struct rlSWVertex {
    float x, y, z, w;                  // Clip-space position
    float sx, sy, sz;                  // Screen-space position
    float u, v;                        // Texture coordinates
    float r, g, b, a;                  // Vertex color [0..1]
    float invW;                        // 1/w for perspective correction
} rlSWVertex;

//----------------------------------------------------------------------------------
// Global Variables Definition
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
        int scissorRect[4];            // x, y, width, height
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

    // Software framebuffer (main render target)
    unsigned char *colorBuffer;         // RGBA8, width*height*4
    float *depthBuffer;                 // float, width*height
    int fbWidth, fbHeight;

    // Viewport
    int viewportX, viewportY, viewportW, viewportH;

    // Resource tracking
    rlSWTexture textures[RL_SW_MAX_TEXTURES];
    unsigned int textureCount;

    rlSWShader shaders[RL_SW_MAX_SHADERS];
    unsigned int shaderCount;

    rlSWBuffer buffers[RL_SW_MAX_BUFFERS];
    unsigned int bufferCount;

    rlSWFramebuffer framebuffers[RL_SW_MAX_FRAMEBUFFERS];
    unsigned int framebufferCount;

    // Active FBO (0 = default framebuffer)
    unsigned int activeFBO;

    rlRenderBatch defaultBatch;
    rlRenderBatch *currentBatch;
} rlglData;

static rlglData RLGL = { 0 };

//----------------------------------------------------------------------------------
// Forward declarations
//----------------------------------------------------------------------------------
static void rlLoadShaderDefault(void);
static void rlUnloadShaderDefault(void);

// Rasterization helpers
static void swRasterizeTriangle(rlSWVertex v0, rlSWVertex v1, rlSWVertex v2, unsigned int texId);
static void swRasterizeLine(rlSWVertex v0, rlSWVertex v1);
static void swSampleTexture(unsigned int texId, float u, float v, unsigned char *outRGBA);
static void swBlendPixel(unsigned char *dst, const unsigned char *src);

// Get active color/depth buffer
static unsigned char *swGetColorBuffer(void);
static float *swGetDepthBuffer(void);
static int swGetFBWidth(void);
static int swGetFBHeight(void);

// Auxiliar math types and functions
typedef struct rl_float16 { float v[16]; } rl_float16;
static rl_float16 rlMatrixToFloatV(Matrix mat);
static Matrix rlMatrixIdentity(void);
static Matrix rlMatrixMultiply(Matrix left, Matrix right);
static Matrix rlMatrixTranspose(Matrix mat);
static Matrix rlMatrixInvert(Matrix mat);

//----------------------------------------------------------------------------------
// Active framebuffer accessors (handles FBO redirection)
//----------------------------------------------------------------------------------
static unsigned char *swGetColorBuffer(void)
{
    if (RLGL.activeFBO > 0 && RLGL.activeFBO <= RLGL.framebufferCount) {
        unsigned int idx = RLGL.activeFBO - 1;
        if (RLGL.framebuffers[idx].colorBuffer) return RLGL.framebuffers[idx].colorBuffer;
    }
    return RLGL.colorBuffer;
}

static float *swGetDepthBuffer(void)
{
    if (RLGL.activeFBO > 0 && RLGL.activeFBO <= RLGL.framebufferCount) {
        unsigned int idx = RLGL.activeFBO - 1;
        if (RLGL.framebuffers[idx].depthBuffer) return RLGL.framebuffers[idx].depthBuffer;
    }
    return RLGL.depthBuffer;
}

static int swGetFBWidth(void)
{
    if (RLGL.activeFBO > 0 && RLGL.activeFBO <= RLGL.framebufferCount) {
        return RLGL.framebuffers[RLGL.activeFBO - 1].width;
    }
    return RLGL.fbWidth;
}

static int swGetFBHeight(void)
{
    if (RLGL.activeFBO > 0 && RLGL.activeFBO <= RLGL.framebufferCount) {
        return RLGL.framebuffers[RLGL.activeFBO - 1].height;
    }
    return RLGL.fbHeight;
}

//----------------------------------------------------------------------------------
// Texture sampling (nearest neighbor, RGBA8)
//----------------------------------------------------------------------------------
static void swSampleTexture(unsigned int texId, float u, float v, unsigned char *outRGBA)
{
    if (texId == 0 || texId > RLGL.textureCount || !RLGL.textures[texId - 1].pixels) {
        outRGBA[0] = 255; outRGBA[1] = 255; outRGBA[2] = 255; outRGBA[3] = 255;
        return;
    }
    const rlSWTexture *tex = &RLGL.textures[texId - 1];

    // Wrap to [0,1]
    u = u - floorf(u);
    v = v - floorf(v);

    int tx = (int)(u * tex->width);
    int ty = (int)(v * tex->height);
    if (tx >= tex->width) tx = tex->width - 1;
    if (ty >= tex->height) ty = tex->height - 1;
    if (tx < 0) tx = 0;
    if (ty < 0) ty = 0;

    int offset = (ty * tex->width + tx) * 4;
    outRGBA[0] = tex->pixels[offset + 0];
    outRGBA[1] = tex->pixels[offset + 1];
    outRGBA[2] = tex->pixels[offset + 2];
    outRGBA[3] = tex->pixels[offset + 3];
}

//----------------------------------------------------------------------------------
// Alpha blending (src-alpha, one-minus-src-alpha)
//----------------------------------------------------------------------------------
static void swBlendPixel(unsigned char *dst, const unsigned char *src)
{
    if (!RLGL.State.colorBlendEnabled || src[3] == 255) {
        dst[0] = src[0]; dst[1] = src[1]; dst[2] = src[2]; dst[3] = src[3];
        return;
    }
    if (src[3] == 0) return;

    int sa = src[3];
    int da = 255 - sa;
    dst[0] = (unsigned char)((src[0] * sa + dst[0] * da) / 255);
    dst[1] = (unsigned char)((src[1] * sa + dst[1] * da) / 255);
    dst[2] = (unsigned char)((src[2] * sa + dst[2] * da) / 255);
    dst[3] = (unsigned char)SW_MIN(255, sa + (dst[3] * da) / 255);
}

//----------------------------------------------------------------------------------
// Triangle rasterizer (barycentric method, perspective-correct interpolation)
//----------------------------------------------------------------------------------
static void swRasterizeTriangle(rlSWVertex v0, rlSWVertex v1, rlSWVertex v2, unsigned int texId)
{
    unsigned char *colorBuf = swGetColorBuffer();
    float *depthBuf = swGetDepthBuffer();
    int fbW = swGetFBWidth();
    int fbH = swGetFBHeight();
    if (!colorBuf || fbW <= 0 || fbH <= 0) return;

    // Backface culling (screen-space cross product)
    if (RLGL.State.backfaceCullingEnabled) {
        float cross = (v1.sx - v0.sx) * (v2.sy - v0.sy) - (v1.sy - v0.sy) * (v2.sx - v0.sx);
        if (cross <= 0.0f) return; // CW winding = backface (raylib convention)
    }

    // Bounding box
    int minX = (int)floorf(SW_MIN(SW_MIN(v0.sx, v1.sx), v2.sx));
    int maxX = (int)ceilf(SW_MAX(SW_MAX(v0.sx, v1.sx), v2.sx));
    int minY = (int)floorf(SW_MIN(SW_MIN(v0.sy, v1.sy), v2.sy));
    int maxY = (int)ceilf(SW_MAX(SW_MAX(v0.sy, v1.sy), v2.sy));

    // Clip to viewport
    int vpX = RLGL.viewportX;
    int vpY = RLGL.viewportY;
    int vpX2 = vpX + RLGL.viewportW;
    int vpY2 = vpY + RLGL.viewportH;

    minX = SW_MAX(minX, vpX);
    minY = SW_MAX(minY, vpY);
    maxX = SW_MIN(maxX, vpX2 - 1);
    maxY = SW_MIN(maxY, vpY2 - 1);

    // Clip to framebuffer
    minX = SW_MAX(minX, 0);
    minY = SW_MAX(minY, 0);
    maxX = SW_MIN(maxX, fbW - 1);
    maxY = SW_MIN(maxY, fbH - 1);

    // Scissor test
    if (RLGL.State.scissorTest) {
        minX = SW_MAX(minX, RLGL.State.scissorRect[0]);
        minY = SW_MAX(minY, RLGL.State.scissorRect[1]);
        maxX = SW_MIN(maxX, RLGL.State.scissorRect[0] + RLGL.State.scissorRect[2] - 1);
        maxY = SW_MIN(maxY, RLGL.State.scissorRect[1] + RLGL.State.scissorRect[3] - 1);
    }

    if (minX > maxX || minY > maxY) return;

    // Edge function denominator
    float denom = (v1.sy - v2.sy) * (v0.sx - v2.sx) + (v2.sx - v1.sx) * (v0.sy - v2.sy);
    if (fabsf(denom) < 1e-8f) return; // Degenerate triangle
    float invDenom = 1.0f / denom;

    // Pre-compute perspective-correct attribute values (attr/w and 1/w)
    float u0w = v0.u * v0.invW, u1w = v1.u * v1.invW, u2w = v2.u * v2.invW;
    float v0w = v0.v * v0.invW, v1w = v1.v * v1.invW, v2w = v2.v * v2.invW;
    float r0w = v0.r * v0.invW, r1w = v1.r * v1.invW, r2w = v2.r * v2.invW;
    float g0w = v0.g * v0.invW, g1w = v1.g * v1.invW, g2w = v2.g * v2.invW;
    float b0w = v0.b * v0.invW, b1w = v1.b * v1.invW, b2w = v2.b * v2.invW;
    float a0w = v0.a * v0.invW, a1w = v1.a * v1.invW, a2w = v2.a * v2.invW;

    // Rasterize
    for (int py = minY; py <= maxY; py++) {
        for (int px = minX; px <= maxX; px++) {
            float pxf = (float)px + 0.5f;
            float pyf = (float)py + 0.5f;

            // Barycentric coordinates
            float w0 = ((v1.sy - v2.sy) * (pxf - v2.sx) + (v2.sx - v1.sx) * (pyf - v2.sy)) * invDenom;
            float w1 = ((v2.sy - v0.sy) * (pxf - v2.sx) + (v0.sx - v2.sx) * (pyf - v2.sy)) * invDenom;
            float w2 = 1.0f - w0 - w1;

            // Inside test
            if (w0 < 0.0f || w1 < 0.0f || w2 < 0.0f) continue;

            // Depth interpolation (linear in screen space for z)
            float z = w0 * v0.sz + w1 * v1.sz + w2 * v2.sz;

            // Depth test
            int pixIdx = py * fbW + px;
            if (RLGL.State.depthTestEnabled && depthBuf) {
                if (z > depthBuf[pixIdx]) continue;
            }

            // Perspective-correct interpolation
            float invWInterp = w0 * v0.invW + w1 * v1.invW + w2 * v2.invW;
            float wInterp = (fabsf(invWInterp) > 1e-8f) ? (1.0f / invWInterp) : 1.0f;

            float tu = (w0 * u0w + w1 * u1w + w2 * u2w) * wInterp;
            float tv = (w0 * v0w + w1 * v1w + w2 * v2w) * wInterp;
            float cr = (w0 * r0w + w1 * r1w + w2 * r2w) * wInterp;
            float cg = (w0 * g0w + w1 * g1w + w2 * g2w) * wInterp;
            float cb = (w0 * b0w + w1 * b1w + w2 * b2w) * wInterp;
            float ca = (w0 * a0w + w1 * a1w + w2 * a2w) * wInterp;

            // Sample texture
            unsigned char texColor[4];
            swSampleTexture(texId, tu, tv, texColor);

            // Final color = texel * vertex_color (fixed-function pipeline)
            unsigned char finalColor[4];
            finalColor[0] = (unsigned char)SW_CLAMP((int)(texColor[0] * cr), 0, 255);
            finalColor[1] = (unsigned char)SW_CLAMP((int)(texColor[1] * cg), 0, 255);
            finalColor[2] = (unsigned char)SW_CLAMP((int)(texColor[2] * cb), 0, 255);
            finalColor[3] = (unsigned char)SW_CLAMP((int)(texColor[3] * ca), 0, 255);

            // Blend and write
            int colorIdx = pixIdx * 4;
            swBlendPixel(&colorBuf[colorIdx], finalColor);

            // Write depth
            if (RLGL.State.depthWriteEnabled && depthBuf) {
                depthBuf[pixIdx] = z;
            }
        }
    }
}

//----------------------------------------------------------------------------------
// Line rasterizer (Bresenham)
//----------------------------------------------------------------------------------
static void swRasterizeLine(rlSWVertex v0, rlSWVertex v1)
{
    unsigned char *colorBuf = swGetColorBuffer();
    float *depthBuf = swGetDepthBuffer();
    int fbW = swGetFBWidth();
    int fbH = swGetFBHeight();
    if (!colorBuf || fbW <= 0 || fbH <= 0) return;

    int x0 = (int)v0.sx, y0 = (int)v0.sy;
    int x1 = (int)v1.sx, y1 = (int)v1.sy;

    int dx = abs(x1 - x0);
    int dy = abs(y1 - y0);
    int sx = (x0 < x1) ? 1 : -1;
    int sy = (y0 < y1) ? 1 : -1;
    int err = dx - dy;
    int steps = dx + dy;
    if (steps == 0) steps = 1;

    for (int s = 0; s <= steps; s++) {
        if (x0 >= 0 && x0 < fbW && y0 >= 0 && y0 < fbH) {
            bool inViewport = (x0 >= RLGL.viewportX && x0 < RLGL.viewportX + RLGL.viewportW &&
                               y0 >= RLGL.viewportY && y0 < RLGL.viewportY + RLGL.viewportH);
            if (RLGL.State.scissorTest) {
                inViewport = inViewport &&
                    (x0 >= RLGL.State.scissorRect[0]) &&
                    (x0 < RLGL.State.scissorRect[0] + RLGL.State.scissorRect[2]) &&
                    (y0 >= RLGL.State.scissorRect[1]) &&
                    (y0 < RLGL.State.scissorRect[1] + RLGL.State.scissorRect[3]);
            }
            if (inViewport) {
                // Interpolate color
                float t = (steps > 0) ? (float)s / (float)steps : 0.0f;
                unsigned char finalColor[4];
                finalColor[0] = (unsigned char)SW_CLAMP((int)((v0.r + t * (v1.r - v0.r)) * 255.0f), 0, 255);
                finalColor[1] = (unsigned char)SW_CLAMP((int)((v0.g + t * (v1.g - v0.g)) * 255.0f), 0, 255);
                finalColor[2] = (unsigned char)SW_CLAMP((int)((v0.b + t * (v1.b - v0.b)) * 255.0f), 0, 255);
                finalColor[3] = (unsigned char)SW_CLAMP((int)((v0.a + t * (v1.a - v0.a)) * 255.0f), 0, 255);

                int pixIdx = y0 * fbW + x0;
                float z = v0.sz + t * (v1.sz - v0.sz);

                if (RLGL.State.depthTestEnabled && depthBuf) {
                    if (z > depthBuf[pixIdx]) goto next;
                }

                swBlendPixel(&colorBuf[pixIdx * 4], finalColor);
                if (RLGL.State.depthWriteEnabled && depthBuf) depthBuf[pixIdx] = z;
            }
        }
next:
        if (x0 == x1 && y0 == y1) break;
        int e2 = 2 * err;
        if (e2 > -dy) { err -= dy; x0 += sx; }
        if (e2 < dx)  { err += dx; y0 += sy; }
    }
}

//----------------------------------------------------------------------------------
// Transform a batch vertex to screen space
//----------------------------------------------------------------------------------
static rlSWVertex swTransformVertex(float px, float py, float pz,
                                     float tu, float tv,
                                     unsigned char cr, unsigned char cg, unsigned char cb, unsigned char ca,
                                     const Matrix *mvp)
{
    rlSWVertex v;

    // MVP transform
    v.x = mvp->m0*px + mvp->m4*py + mvp->m8*pz  + mvp->m12;
    v.y = mvp->m1*px + mvp->m5*py + mvp->m9*pz  + mvp->m13;
    v.z = mvp->m2*px + mvp->m6*py + mvp->m10*pz + mvp->m14;
    v.w = mvp->m3*px + mvp->m7*py + mvp->m11*pz + mvp->m15;

    // Perspective divide
    if (fabsf(v.w) > 1e-8f) {
        v.invW = 1.0f / v.w;
        float ndcX = v.x * v.invW;
        float ndcY = v.y * v.invW;
        float ndcZ = v.z * v.invW;

        // Viewport transform (NDC to screen coordinates)
        // NDC Y is up, screen Y is down
        v.sx = (ndcX * 0.5f + 0.5f) * RLGL.viewportW + RLGL.viewportX;
        v.sy = (1.0f - (ndcY * 0.5f + 0.5f)) * RLGL.viewportH + RLGL.viewportY;
        v.sz = ndcZ * 0.5f + 0.5f;  // Map [-1,1] to [0,1]
    } else {
        v.invW = 1.0f;
        v.sx = 0; v.sy = 0; v.sz = 0;
    }

    v.u = tu;
    v.v = tv;
    v.r = cr / 255.0f;
    v.g = cg / 255.0f;
    v.b = cb / 255.0f;
    v.a = ca / 255.0f;

    return v;
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
    RLGL.viewportX = x;
    RLGL.viewportY = y;
    RLGL.viewportW = width;
    RLGL.viewportH = height;
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
    if (id > 0 && id <= RLGL.framebufferCount) {
        RLGL.activeFBO = id;
    } else {
        RLGL.activeFBO = 0;
    }
}
void rlDisableFramebuffer(void) { RLGL.activeFBO = 0; }
unsigned int rlGetActiveFramebuffer(void) { return RLGL.activeFBO; }

//----------------------------------------------------------------------------------
// Render state management
//----------------------------------------------------------------------------------
void rlEnableColorBlend(void) { RLGL.State.colorBlendEnabled = true; }
void rlDisableColorBlend(void) { RLGL.State.colorBlendEnabled = false; }
void rlEnableDepthTest(void) { RLGL.State.depthTestEnabled = true; }
void rlDisableDepthTest(void) { RLGL.State.depthTestEnabled = false; }
void rlEnableDepthMask(void) { RLGL.State.depthWriteEnabled = true; }
void rlDisableDepthMask(void) { RLGL.State.depthWriteEnabled = false; }
void rlEnableBackfaceCulling(void) { RLGL.State.backfaceCullingEnabled = true; }
void rlDisableBackfaceCulling(void) { RLGL.State.backfaceCullingEnabled = false; }
void rlColorMask(bool r, bool g, bool b, bool a) { (void)r; (void)g; (void)b; (void)a; }
void rlSetCullFace(int mode) { (void)mode; }
void rlEnableScissorTest(void) { RLGL.State.scissorTest = true; }
void rlDisableScissorTest(void) { RLGL.State.scissorTest = false; }
void rlScissor(int x, int y, int width, int height) {
    RLGL.State.scissorRect[0] = x;
    RLGL.State.scissorRect[1] = y;
    RLGL.State.scissorRect[2] = width;
    RLGL.State.scissorRect[3] = height;
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

void rlClearScreenBuffers(void)
{
    unsigned char *colorBuf = swGetColorBuffer();
    float *depthBuf = swGetDepthBuffer();
    int w = swGetFBWidth();
    int h = swGetFBHeight();

    if (colorBuf && w > 0 && h > 0) {
        unsigned char cr = (unsigned char)(RLGL.State.clearColor[0] * 255.0f);
        unsigned char cg = (unsigned char)(RLGL.State.clearColor[1] * 255.0f);
        unsigned char cb = (unsigned char)(RLGL.State.clearColor[2] * 255.0f);
        unsigned char ca = (unsigned char)(RLGL.State.clearColor[3] * 255.0f);

        int pixelCount = w * h;
        for (int i = 0; i < pixelCount; i++) {
            colorBuf[4*i + 0] = cr;
            colorBuf[4*i + 1] = cg;
            colorBuf[4*i + 2] = cb;
            colorBuf[4*i + 3] = ca;
        }
    }

    if (depthBuf && w > 0 && h > 0) {
        int pixelCount = w * h;
        for (int i = 0; i < pixelCount; i++) {
            depthBuf[i] = 1.0f; // Far plane
        }
    }
}

void rlCheckErrors(void) { }
void rlSetBlendMode(int mode) { RLGL.State.currentBlendMode = mode; }
void rlSetBlendFactors(int glSrcFactor, int glDstFactor, int glEquation) { (void)glSrcFactor; (void)glDstFactor; (void)glEquation; }
void rlSetBlendFactorsSeparate(int glSrcRGB, int glDstRGB, int glSrcAlpha, int glDstAlpha, int glEqRGB, int glEqAlpha) { (void)glSrcRGB; (void)glDstRGB; (void)glSrcAlpha; (void)glDstAlpha; (void)glEqRGB; (void)glEqAlpha; }

//----------------------------------------------------------------------------------
// rlgl initialization and cleanup
//----------------------------------------------------------------------------------
void rlglInit(int width, int height, bool headless)
{
    (void)headless;
    memset(&RLGL, 0, sizeof(rlglData));

    // Allocate main framebuffer
    RLGL.fbWidth = width;
    RLGL.fbHeight = height;
    RLGL.colorBuffer = (unsigned char *)RL_CALLOC(width * height * 4, sizeof(unsigned char));
    RLGL.depthBuffer = (float *)RL_MALLOC(width * height * sizeof(float));
    for (int i = 0; i < width * height; i++) RLGL.depthBuffer[i] = 1.0f;

    // Default viewport
    RLGL.viewportX = 0;
    RLGL.viewportY = 0;
    RLGL.viewportW = width;
    RLGL.viewportH = height;

    // Create default 1x1 white texture
    unsigned char whitePixel[4] = { 255, 255, 255, 255 };
    unsigned int idx = RLGL.textureCount;
    RLGL.textures[idx].pixels = (unsigned char *)RL_MALLOC(4);
    memcpy(RLGL.textures[idx].pixels, whitePixel, 4);
    RLGL.textures[idx].width = 1;
    RLGL.textures[idx].height = 1;
    RLGL.textures[idx].mipmaps = 1;
    RLGL.textures[idx].format = 7; // RGBA8
    RLGL.textureCount++;
    RLGL.State.defaultTextureId = idx + 1;

    // Load default "shader" (fixed-function in software)
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

    RLGL.State.framebufferWidth = width;
    RLGL.State.framebufferHeight = height;

    TRACELOG(RL_LOG_INFO, "RLGL: Software rendering backend initialized successfully (%dx%d)", width, height);
}

void rlglClose(void)
{
    rlUnloadRenderBatch(RLGL.defaultBatch);
    rlUnloadShaderDefault();

    if (RLGL.State.defaultTextureId > 0) rlUnloadTexture(RLGL.State.defaultTextureId);

    // Release all tracked textures
    for (unsigned int i = 0; i < RLGL.textureCount; i++) {
        if (RLGL.textures[i].pixels) { RL_FREE(RLGL.textures[i].pixels); RLGL.textures[i].pixels = NULL; }
    }

    // Release all tracked buffers
    for (unsigned int i = 0; i < RLGL.bufferCount; i++) {
        if (RLGL.buffers[i].data) { RL_FREE(RLGL.buffers[i].data); RLGL.buffers[i].data = NULL; }
    }

    // Release all tracked framebuffers
    for (unsigned int i = 0; i < RLGL.framebufferCount; i++) {
        if (RLGL.framebuffers[i].depthBuffer) { RL_FREE(RLGL.framebuffers[i].depthBuffer); RLGL.framebuffers[i].depthBuffer = NULL; }
    }

    // Release main framebuffer
    if (RLGL.colorBuffer) { RL_FREE(RLGL.colorBuffer); RLGL.colorBuffer = NULL; }
    if (RLGL.depthBuffer) { RL_FREE(RLGL.depthBuffer); RLGL.depthBuffer = NULL; }

    TRACELOG(RL_LOG_INFO, "RLGL: Software rendering backend closed successfully");
}

void rlLoadExtensions(void *loader)
{
    (void)loader;
    // Software renderer capabilities
    RLGL.ExtSupported.computeShader = false;
    RLGL.ExtSupported.ssbo = false;
    RLGL.ExtSupported.texNPOT = true;
    RLGL.ExtSupported.texCompDXT = false;
    RLGL.ExtSupported.texFloat32 = true;
    RLGL.ExtSupported.texFloat16 = true;
    RLGL.ExtSupported.texDepth = true;
    RLGL.ExtSupported.maxAnisotropyLevel = 1.0f;
    RLGL.ExtSupported.maxDepthBits = 32;

    TRACELOG(RL_LOG_INFO, "SOFTWARE: Renderer capabilities loaded (CPU-based rasterization)");
}

int rlGetVersion(void) { return RL_SOFTWARE; }

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
        batch.vertexBuffer[i].indices = (unsigned int *)RL_MALLOC(indexCount * sizeof(unsigned int));
        for (int q = 0, k = 0; q < bufferElements; q++, k += 6) {
            batch.vertexBuffer[i].indices[k+0] = (unsigned int)(4*q);
            batch.vertexBuffer[i].indices[k+1] = (unsigned int)(4*q+1);
            batch.vertexBuffer[i].indices[k+2] = (unsigned int)(4*q+2);
            batch.vertexBuffer[i].indices[k+3] = (unsigned int)(4*q);
            batch.vertexBuffer[i].indices[k+4] = (unsigned int)(4*q+2);
            batch.vertexBuffer[i].indices[k+5] = (unsigned int)(4*q+3);
        }

        batch.vertexBuffer[i].vCounter = 0;
        batch.vertexBuffer[i].tcCounter = 0;
        batch.vertexBuffer[i].ncCounter = 0;
        batch.vertexBuffer[i].cCounter = 0;
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
// Draw render batch (software rasterization)
//----------------------------------------------------------------------------------
void rlDrawRenderBatch(rlRenderBatch *batch)
{
    if (batch->vertexBuffer[batch->currentBuffer].vCounter == 0) return;

    int vCount = batch->vertexBuffer[batch->currentBuffer].vCounter;
    float *vertices  = batch->vertexBuffer[batch->currentBuffer].vertices;
    float *texcoords = batch->vertexBuffer[batch->currentBuffer].texcoords;
    unsigned char *colors = batch->vertexBuffer[batch->currentBuffer].colors;
    unsigned int *indices = batch->vertexBuffer[batch->currentBuffer].indices;

    // Compute MVP matrix
    Matrix matMVP = rlMatrixMultiply(RLGL.State.modelview, RLGL.State.projection);

    // Process draw calls
    int vertexOffset = 0;
    for (int i = 0; i < batch->drawCounter; i++) {
        int drawVertexCount = batch->draws[i].vertexCount;
        if (drawVertexCount == 0) { vertexOffset += batch->draws[i].vertexAlignment; continue; }

        unsigned int texId = batch->draws[i].textureId;
        int mode = batch->draws[i].mode;

        if (mode == RL_QUADS) {
            // Quads use index buffer (6 indices per quad, 4 vertices per quad)
            int quadCount = drawVertexCount / 4;
            int indexOffset = (vertexOffset / 4) * 6;

            for (int q = 0; q < quadCount; q++) {
                int baseIdx = indexOffset + q * 6;

                // Two triangles per quad: (0,1,2) and (0,2,3)
                for (int t = 0; t < 2; t++) {
                    int i0 = indices[baseIdx + t*3 + 0];
                    int i1 = indices[baseIdx + t*3 + 1];
                    int i2 = indices[baseIdx + t*3 + 2];

                    rlSWVertex sv0 = swTransformVertex(
                        vertices[3*i0], vertices[3*i0+1], vertices[3*i0+2],
                        texcoords[2*i0], texcoords[2*i0+1],
                        colors[4*i0], colors[4*i0+1], colors[4*i0+2], colors[4*i0+3],
                        &matMVP);
                    rlSWVertex sv1 = swTransformVertex(
                        vertices[3*i1], vertices[3*i1+1], vertices[3*i1+2],
                        texcoords[2*i1], texcoords[2*i1+1],
                        colors[4*i1], colors[4*i1+1], colors[4*i1+2], colors[4*i1+3],
                        &matMVP);
                    rlSWVertex sv2 = swTransformVertex(
                        vertices[3*i2], vertices[3*i2+1], vertices[3*i2+2],
                        texcoords[2*i2], texcoords[2*i2+1],
                        colors[4*i2], colors[4*i2+1], colors[4*i2+2], colors[4*i2+3],
                        &matMVP);

                    // Simple near-plane clip: skip if any vertex is behind camera
                    if (sv0.w <= 0.0f || sv1.w <= 0.0f || sv2.w <= 0.0f) continue;

                    swRasterizeTriangle(sv0, sv1, sv2, texId);
                }
            }
        } else if (mode == RL_TRIANGLES) {
            int triCount = drawVertexCount / 3;
            for (int t = 0; t < triCount; t++) {
                int vi = vertexOffset + t * 3;

                rlSWVertex sv0 = swTransformVertex(
                    vertices[3*vi], vertices[3*vi+1], vertices[3*vi+2],
                    texcoords[2*vi], texcoords[2*vi+1],
                    colors[4*vi], colors[4*vi+1], colors[4*vi+2], colors[4*vi+3],
                    &matMVP);
                rlSWVertex sv1 = swTransformVertex(
                    vertices[3*(vi+1)], vertices[3*(vi+1)+1], vertices[3*(vi+1)+2],
                    texcoords[2*(vi+1)], texcoords[2*(vi+1)+1],
                    colors[4*(vi+1)], colors[4*(vi+1)+1], colors[4*(vi+1)+2], colors[4*(vi+1)+3],
                    &matMVP);
                rlSWVertex sv2 = swTransformVertex(
                    vertices[3*(vi+2)], vertices[3*(vi+2)+1], vertices[3*(vi+2)+2],
                    texcoords[2*(vi+2)], texcoords[2*(vi+2)+1],
                    colors[4*(vi+2)], colors[4*(vi+2)+1], colors[4*(vi+2)+2], colors[4*(vi+2)+3],
                    &matMVP);

                if (sv0.w <= 0.0f || sv1.w <= 0.0f || sv2.w <= 0.0f) continue;

                swRasterizeTriangle(sv0, sv1, sv2, texId);
            }
        } else if (mode == RL_LINES) {
            int lineCount = drawVertexCount / 2;
            for (int l = 0; l < lineCount; l++) {
                int vi = vertexOffset + l * 2;

                rlSWVertex sv0 = swTransformVertex(
                    vertices[3*vi], vertices[3*vi+1], vertices[3*vi+2],
                    texcoords[2*vi], texcoords[2*vi+1],
                    colors[4*vi], colors[4*vi+1], colors[4*vi+2], colors[4*vi+3],
                    &matMVP);
                rlSWVertex sv1 = swTransformVertex(
                    vertices[3*(vi+1)], vertices[3*(vi+1)+1], vertices[3*(vi+1)+2],
                    texcoords[2*(vi+1)], texcoords[2*(vi+1)+1],
                    colors[4*(vi+1)], colors[4*(vi+1)+1], colors[4*(vi+1)+2], colors[4*(vi+1)+3],
                    &matMVP);

                if (sv0.w <= 0.0f || sv1.w <= 0.0f) continue;

                swRasterizeLine(sv0, sv1);
            }
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
    if (RLGL.textureCount >= RL_SW_MAX_TEXTURES) {
        TRACELOG(RL_LOG_WARNING, "TEXTURE: Max software texture limit reached (%d)", RL_SW_MAX_TEXTURES);
        return 0;
    }

    int pixelCount = width * height;
    unsigned char *rgba = (unsigned char *)RL_CALLOC(pixelCount * 4, sizeof(unsigned char));

    // Convert source data to RGBA8
    if (data != NULL) {
        const unsigned char *src = (const unsigned char *)data;
        switch (format) {
            case 1: // GRAYSCALE
                for (int i = 0; i < pixelCount; i++) {
                    rgba[4*i+0] = src[i]; rgba[4*i+1] = src[i]; rgba[4*i+2] = src[i]; rgba[4*i+3] = 255;
                }
                break;
            case 2: // GRAY_ALPHA
                for (int i = 0; i < pixelCount; i++) {
                    rgba[4*i+0] = src[2*i]; rgba[4*i+1] = src[2*i]; rgba[4*i+2] = src[2*i]; rgba[4*i+3] = src[2*i+1];
                }
                break;
            case 4: // R8G8B8
                for (int i = 0; i < pixelCount; i++) {
                    rgba[4*i+0] = src[3*i+0]; rgba[4*i+1] = src[3*i+1]; rgba[4*i+2] = src[3*i+2]; rgba[4*i+3] = 255;
                }
                break;
            case 7: // R8G8B8A8
                memcpy(rgba, data, pixelCount * 4);
                break;
            case 3: // R5G6B5
                for (int i = 0; i < pixelCount; i++) {
                    unsigned short px = ((const unsigned short *)data)[i];
                    rgba[4*i+0] = (unsigned char)(((px >> 11) & 0x1F) * 255 / 31);
                    rgba[4*i+1] = (unsigned char)(((px >> 5) & 0x3F) * 255 / 63);
                    rgba[4*i+2] = (unsigned char)((px & 0x1F) * 255 / 31);
                    rgba[4*i+3] = 255;
                }
                break;
            case 5: // R5G5B5A1
                for (int i = 0; i < pixelCount; i++) {
                    unsigned short px = ((const unsigned short *)data)[i];
                    rgba[4*i+0] = (unsigned char)(((px >> 11) & 0x1F) * 255 / 31);
                    rgba[4*i+1] = (unsigned char)(((px >> 6) & 0x1F) * 255 / 31);
                    rgba[4*i+2] = (unsigned char)(((px >> 1) & 0x1F) * 255 / 31);
                    rgba[4*i+3] = (px & 0x01) ? 255 : 0;
                }
                break;
            case 6: // R4G4B4A4
                for (int i = 0; i < pixelCount; i++) {
                    unsigned short px = ((const unsigned short *)data)[i];
                    rgba[4*i+0] = (unsigned char)(((px >> 12) & 0x0F) * 17);
                    rgba[4*i+1] = (unsigned char)(((px >> 8) & 0x0F) * 17);
                    rgba[4*i+2] = (unsigned char)(((px >> 4) & 0x0F) * 17);
                    rgba[4*i+3] = (unsigned char)((px & 0x0F) * 17);
                }
                break;
            case 8: // R32 (float)
                for (int i = 0; i < pixelCount; i++) {
                    float val = ((const float *)data)[i];
                    unsigned char v = (unsigned char)SW_CLAMP((int)(val * 255.0f), 0, 255);
                    rgba[4*i+0] = v; rgba[4*i+1] = v; rgba[4*i+2] = v; rgba[4*i+3] = 255;
                }
                break;
            case 9: // R32G32B32 (float)
                for (int i = 0; i < pixelCount; i++) {
                    const float *fp = &((const float *)data)[3*i];
                    rgba[4*i+0] = (unsigned char)SW_CLAMP((int)(fp[0] * 255.0f), 0, 255);
                    rgba[4*i+1] = (unsigned char)SW_CLAMP((int)(fp[1] * 255.0f), 0, 255);
                    rgba[4*i+2] = (unsigned char)SW_CLAMP((int)(fp[2] * 255.0f), 0, 255);
                    rgba[4*i+3] = 255;
                }
                break;
            case 10: // R32G32B32A32 (float)
                for (int i = 0; i < pixelCount; i++) {
                    const float *fp = &((const float *)data)[4*i];
                    rgba[4*i+0] = (unsigned char)SW_CLAMP((int)(fp[0] * 255.0f), 0, 255);
                    rgba[4*i+1] = (unsigned char)SW_CLAMP((int)(fp[1] * 255.0f), 0, 255);
                    rgba[4*i+2] = (unsigned char)SW_CLAMP((int)(fp[2] * 255.0f), 0, 255);
                    rgba[4*i+3] = (unsigned char)SW_CLAMP((int)(fp[3] * 255.0f), 0, 255);
                }
                break;
            default:
                // For compressed and other formats, fill with white
                memset(rgba, 255, pixelCount * 4);
                TRACELOG(RL_LOG_WARNING, "TEXTURE: Format %d not directly supported by software renderer, using white", format);
                break;
        }
    }

    unsigned int idx = RLGL.textureCount;
    RLGL.textures[idx].pixels = rgba;
    RLGL.textures[idx].width = width;
    RLGL.textures[idx].height = height;
    RLGL.textures[idx].mipmaps = (mipmapCount > 0) ? mipmapCount : 1;
    RLGL.textures[idx].format = format;
    RLGL.textures[idx].isRenderTarget = false;
    RLGL.textures[idx].isDepth = false;
    RLGL.textureCount++;

    unsigned int id = idx + 1;
    TRACELOG(RL_LOG_INFO, "TEXTURE: [ID %i] Software texture loaded (%ix%i, format: %i)", id, width, height, format);
    return id;
}

unsigned int rlLoadTextureDepth(int width, int height, bool useRenderBuffer)
{
    (void)useRenderBuffer;
    if (RLGL.textureCount >= RL_SW_MAX_TEXTURES) return 0;

    unsigned int idx = RLGL.textureCount;
    RLGL.textures[idx].pixels = NULL;
    RLGL.textures[idx].width = width;
    RLGL.textures[idx].height = height;
    RLGL.textures[idx].mipmaps = 1;
    RLGL.textures[idx].format = 0;
    RLGL.textures[idx].isRenderTarget = false;
    RLGL.textures[idx].isDepth = true;
    RLGL.textureCount++;

    return idx + 1;
}

unsigned int rlLoadTextureCubemap(const void *data, int size, int format, int mipmapCount)
{
    (void)data; (void)size; (void)format; (void)mipmapCount;
    TRACELOG(RL_LOG_WARNING, "TEXTURE: Cubemap textures not supported in software renderer");
    return 0;
}

void rlUpdateTexture(unsigned int id, int offsetX, int offsetY, int width, int height, int format, const void *pixels)
{
    if (id == 0 || id > RLGL.textureCount || !pixels) return;
    unsigned int idx = id - 1;
    if (!RLGL.textures[idx].pixels) return;

    int texW = RLGL.textures[idx].width;

    // Convert and copy region
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int dstIdx = ((offsetY + y) * texW + (offsetX + x)) * 4;
            int srcIdx = (y * width + x);

            switch (format) {
                case 7: // RGBA8
                    memcpy(&RLGL.textures[idx].pixels[dstIdx], &((const unsigned char *)pixels)[srcIdx * 4], 4);
                    break;
                case 4: // RGB8
                    RLGL.textures[idx].pixels[dstIdx + 0] = ((const unsigned char *)pixels)[srcIdx * 3 + 0];
                    RLGL.textures[idx].pixels[dstIdx + 1] = ((const unsigned char *)pixels)[srcIdx * 3 + 1];
                    RLGL.textures[idx].pixels[dstIdx + 2] = ((const unsigned char *)pixels)[srcIdx * 3 + 2];
                    RLGL.textures[idx].pixels[dstIdx + 3] = 255;
                    break;
                case 1: // GRAYSCALE
                {
                    unsigned char v = ((const unsigned char *)pixels)[srcIdx];
                    RLGL.textures[idx].pixels[dstIdx+0] = v;
                    RLGL.textures[idx].pixels[dstIdx+1] = v;
                    RLGL.textures[idx].pixels[dstIdx+2] = v;
                    RLGL.textures[idx].pixels[dstIdx+3] = 255;
                } break;
                case 2: // GRAY_ALPHA
                {
                    unsigned char v = ((const unsigned char *)pixels)[srcIdx * 2];
                    unsigned char a = ((const unsigned char *)pixels)[srcIdx * 2 + 1];
                    RLGL.textures[idx].pixels[dstIdx+0] = v;
                    RLGL.textures[idx].pixels[dstIdx+1] = v;
                    RLGL.textures[idx].pixels[dstIdx+2] = v;
                    RLGL.textures[idx].pixels[dstIdx+3] = a;
                } break;
                default:
                    // For unsupported formats, write white
                    RLGL.textures[idx].pixels[dstIdx+0] = 255;
                    RLGL.textures[idx].pixels[dstIdx+1] = 255;
                    RLGL.textures[idx].pixels[dstIdx+2] = 255;
                    RLGL.textures[idx].pixels[dstIdx+3] = 255;
                    break;
            }
        }
    }
}

void rlUpdateTextureRec(unsigned int id, Rectangle rec, int format, const void *pixels)
{
    rlUpdateTexture(id, (int)rec.x, (int)rec.y, (int)rec.width, (int)rec.height, format, pixels);
}

void rlUnloadTexture(unsigned int id)
{
    if (id == 0 || id > RLGL.textureCount) return;
    unsigned int idx = id - 1;
    if (RLGL.textures[idx].pixels) { RL_FREE(RLGL.textures[idx].pixels); RLGL.textures[idx].pixels = NULL; }
}

void rlGenTextureMipmaps(unsigned int id, int width, int height, int format, int *mipmaps) { (void)id; (void)width; (void)height; (void)format; (void)mipmaps; }

void *rlReadTexturePixels(unsigned int id, int width, int height, int format)
{
    (void)format;
    if (id == 0 || id > RLGL.textureCount) return NULL;
    unsigned int idx = id - 1;
    if (!RLGL.textures[idx].pixels) return NULL;

    int dataSize = width * height * 4; // Always RGBA8
    void *pixels = RL_MALLOC(dataSize);
    memcpy(pixels, RLGL.textures[idx].pixels, dataSize);
    return pixels;
}

unsigned char *rlReadScreenPixels(int width, int height)
{
    unsigned char *screenData = (unsigned char *)RL_CALLOC(width * height * 4, sizeof(unsigned char));

    if (RLGL.colorBuffer) {
        int copyW = SW_MIN(width, RLGL.fbWidth);
        int copyH = SW_MIN(height, RLGL.fbHeight);

        // Copy with Y-flip (OpenGL convention: bottom-to-top, screen: top-to-bottom)
        for (int y = 0; y < copyH; y++) {
            int srcRow = (copyH - 1 - y);
            memcpy(&screenData[y * width * 4], &RLGL.colorBuffer[srcRow * RLGL.fbWidth * 4], copyW * 4);
        }
    }

    return screenData;
}

void rlGetGlTextureFormats(int format, unsigned int *glInternalFormat, unsigned int *glFormat, unsigned int *glType)
{
    // Software renderer doesn't use GL formats, but provide reasonable defaults
    if (glInternalFormat) *glInternalFormat = 0;
    if (glFormat) *glFormat = 0;
    if (glType) *glType = 0;
}

const char *rlGetPixelFormatName(unsigned int format)
{
    switch (format) {
        case 1: return "GRAYSCALE"; break;
        case 2: return "GRAY_ALPHA"; break;
        case 3: return "R5G6B5"; break;
        case 4: return "R8G8B8"; break;
        case 5: return "R5G5B5A1"; break;
        case 6: return "R4G4B4A4"; break;
        case 7: return "R8G8B8A8"; break;
        case 8: return "R32"; break;
        case 9: return "R32G32B32"; break;
        case 10: return "R32G32B32A32"; break;
        case 11: return "R16"; break;
        case 12: return "R16G16B16"; break;
        case 13: return "R16G16B16A16"; break;
        case 14: return "DXT1_RGB"; break;
        case 15: return "DXT1_RGBA"; break;
        case 16: return "DXT3_RGBA"; break;
        case 17: return "DXT5_RGBA"; break;
        case 18: return "ETC1_RGB"; break;
        case 19: return "ETC2_RGB"; break;
        case 20: return "ETC2_EAC_RGBA"; break;
        case 21: return "PVRT_RGB"; break;
        case 22: return "PVRT_RGBA"; break;
        case 23: return "ASTC_4x4_RGBA"; break;
        case 24: return "ASTC_8x8_RGBA"; break;
        default: return "UNKNOWN"; break;
    }
}

//----------------------------------------------------------------------------------
// Framebuffer management
//----------------------------------------------------------------------------------
unsigned int rlLoadFramebuffer(void)
{
    if (RLGL.framebufferCount >= RL_SW_MAX_FRAMEBUFFERS) return 0;
    unsigned int idx = RLGL.framebufferCount;
    memset(&RLGL.framebuffers[idx], 0, sizeof(rlSWFramebuffer));
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
        RLGL.framebuffers[fboIdx].colorBuffer = RLGL.textures[texIdx].pixels;
        RLGL.framebuffers[fboIdx].width = RLGL.textures[texIdx].width;
        RLGL.framebuffers[fboIdx].height = RLGL.textures[texIdx].height;
        RLGL.textures[texIdx].isRenderTarget = true;
    } else if (attachType == RL_ATTACHMENT_DEPTH) {
        RLGL.framebuffers[fboIdx].depthTextureId = texId;
        // Allocate depth buffer for this FBO
        int w = RLGL.framebuffers[fboIdx].width;
        int h = RLGL.framebuffers[fboIdx].height;
        if (w > 0 && h > 0 && !RLGL.framebuffers[fboIdx].depthBuffer) {
            RLGL.framebuffers[fboIdx].depthBuffer = (float *)RL_MALLOC(w * h * sizeof(float));
            for (int i = 0; i < w * h; i++) RLGL.framebuffers[fboIdx].depthBuffer[i] = 1.0f;
        }
    }
}

bool rlFramebufferComplete(unsigned int id) {
    if (id == 0 || id > RLGL.framebufferCount) return false;
    unsigned int idx = id - 1;
    return (RLGL.framebuffers[idx].colorBuffer != NULL);
}

void rlUnloadFramebuffer(unsigned int id) {
    if (id == 0 || id > RLGL.framebufferCount) return;
    unsigned int idx = id - 1;
    if (RLGL.framebuffers[idx].depthBuffer) {
        RL_FREE(RLGL.framebuffers[idx].depthBuffer);
        RLGL.framebuffers[idx].depthBuffer = NULL;
    }
    // Color buffer is owned by the texture, not the framebuffer
}

void rlActiveDrawBuffers(int count) { (void)count; }
void rlBlitFramebuffer(int srcX, int srcY, int srcWidth, int srcHeight, int dstX, int dstY, int dstWidth, int dstHeight, int bufferMask)
{
    (void)srcX; (void)srcY; (void)srcWidth; (void)srcHeight;
    (void)dstX; (void)dstY; (void)dstWidth; (void)dstHeight;
    (void)bufferMask;
}
void rlBindFramebuffer(unsigned int target, unsigned int framebuffer) { (void)target; rlEnableFramebuffer(framebuffer); }

//----------------------------------------------------------------------------------
// Vertex buffer management (user VBOs — CPU-side)
//----------------------------------------------------------------------------------
unsigned int rlLoadVertexArray(void) { return 0; }
void rlSetVertexAttribute(unsigned int index, int compSize, int type, bool normalized, int stride, int offset) { (void)index; (void)compSize; (void)type; (void)normalized; (void)stride; (void)offset; }
void rlSetVertexAttributeDivisor(unsigned int index, int divisor) { (void)index; (void)divisor; }
void rlSetVertexAttributeDefault(int locIndex, const void *value, int attribType, int count) { (void)locIndex; (void)value; (void)attribType; (void)count; }

unsigned int rlLoadVertexBuffer(const void *buffer, int size, bool dynamic)
{
    (void)dynamic;
    if (RLGL.bufferCount >= RL_SW_MAX_BUFFERS) return 0;
    unsigned int idx = RLGL.bufferCount;

    RLGL.buffers[idx].data = (unsigned char *)RL_MALLOC(size);
    if (buffer) memcpy(RLGL.buffers[idx].data, buffer, size);
    else memset(RLGL.buffers[idx].data, 0, size);
    RLGL.buffers[idx].size = size;
    RLGL.buffers[idx].isIndex = false;
    RLGL.bufferCount++;
    return idx + 1;
}

unsigned int rlLoadVertexBufferElement(const void *buffer, int size, bool dynamic)
{
    (void)dynamic;
    if (RLGL.bufferCount >= RL_SW_MAX_BUFFERS) return 0;
    unsigned int idx = RLGL.bufferCount;

    RLGL.buffers[idx].data = (unsigned char *)RL_MALLOC(size);
    if (buffer) memcpy(RLGL.buffers[idx].data, buffer, size);
    else memset(RLGL.buffers[idx].data, 0, size);
    RLGL.buffers[idx].size = size;
    RLGL.buffers[idx].isIndex = true;
    RLGL.bufferCount++;
    return idx + 1;
}

void rlUpdateVertexBuffer(unsigned int bufferId, const void *data, int dataSize, int offset)
{
    if (bufferId == 0 || bufferId > RLGL.bufferCount || !data) return;
    unsigned int idx = bufferId - 1;
    if (RLGL.buffers[idx].data && (offset + dataSize) <= (int)RLGL.buffers[idx].size) {
        memcpy(RLGL.buffers[idx].data + offset, data, dataSize);
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
    if (RLGL.buffers[idx].data) { RL_FREE(RLGL.buffers[idx].data); RLGL.buffers[idx].data = NULL; }
}

void rlBindVertexArray(unsigned int vaoId) { (void)vaoId; }
void rlEnableVertexArray(unsigned int vaoId) { (void)vaoId; }
void rlDisableVertexArray(void) { }
void rlEnableVertexBuffer(unsigned int id) { (void)id; }
void rlDisableVertexBuffer(void) { }
void rlEnableVertexBufferElement(unsigned int id) { (void)id; }
void rlDisableVertexBufferElement(void) { }
void rlEnableVertexAttribute(unsigned int index) { (void)index; }
void rlDisableVertexAttribute(unsigned int index) { (void)index; }

void rlDrawVertexArray(int offset, int count) { (void)offset; (void)count; }
void rlDrawVertexArrayElements(int offset, int count, const void *buffer) { (void)offset; (void)count; (void)buffer; }
void rlDrawVertexArrayInstanced(int offset, int count, int instances) { (void)offset; (void)count; (void)instances; }
void rlDrawVertexArrayElementsInstanced(int offset, int count, const void *buffer, int instances) { (void)offset; (void)count; (void)buffer; (void)instances; }

//----------------------------------------------------------------------------------
// Shader management (fixed-function, stubs)
//----------------------------------------------------------------------------------
unsigned int rlCompileShader(const char *shaderCode, int type)
{
    (void)shaderCode; (void)type;
    // Software renderer uses fixed-function pipeline, no real shader compilation
    return 0;
}

unsigned int rlLoadShaderCode(const char *vsCode, const char *fsCode)
{
    (void)vsCode; (void)fsCode;
    if (RLGL.shaderCount >= RL_SW_MAX_SHADERS) return 0;

    unsigned int idx = RLGL.shaderCount;
    RLGL.shaders[idx].active = true;
    RLGL.shaderCount++;

    unsigned int id = idx + 1;
    TRACELOG(RL_LOG_INFO, "SHADER: [ID %i] Software fixed-function shader loaded", id);
    return id;
}

unsigned int rlLoadShaderProgram(unsigned int vShaderId, unsigned int fShaderId) {
    (void)vShaderId; (void)fShaderId;
    return 0;
}

void rlUnloadShaderProgram(unsigned int id) {
    if (id == 0 || id > RLGL.shaderCount) return;
    RLGL.shaders[id - 1].active = false;
}

int rlGetLocationUniform(unsigned int shaderId, const char *uniformName) { (void)shaderId; (void)uniformName; return -1; }
int rlGetLocationAttrib(unsigned int shaderId, const char *attribName) { (void)shaderId; (void)attribName; return -1; }
void rlSetUniform(int locIndex, const void *value, int uniformType, int count) { (void)locIndex; (void)value; (void)uniformType; (void)count; }
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
unsigned int rlLoadComputeShaderProgram(unsigned int shaderId) { (void)shaderId; return 0; }
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
        rlNormal3f(0,0,-1); rlTexCoord2f(0,0); rlVertex3f(-1,-1,-1);
        rlNormal3f(0,0,-1); rlTexCoord2f(1,1); rlVertex3f(1,1,-1);
        rlNormal3f(0,0,-1); rlTexCoord2f(1,0); rlVertex3f(1,-1,-1);
        rlNormal3f(0,0,-1); rlTexCoord2f(1,1); rlVertex3f(1,1,-1);
        rlNormal3f(0,0,-1); rlTexCoord2f(0,0); rlVertex3f(-1,-1,-1);
        rlNormal3f(0,0,-1); rlTexCoord2f(0,1); rlVertex3f(-1,1,-1);
        rlNormal3f(0,0,1); rlTexCoord2f(0,0); rlVertex3f(-1,-1,1);
        rlNormal3f(0,0,1); rlTexCoord2f(1,0); rlVertex3f(1,-1,1);
        rlNormal3f(0,0,1); rlTexCoord2f(1,1); rlVertex3f(1,1,1);
        rlNormal3f(0,0,1); rlTexCoord2f(1,1); rlVertex3f(1,1,1);
        rlNormal3f(0,0,1); rlTexCoord2f(0,1); rlVertex3f(-1,1,1);
        rlNormal3f(0,0,1); rlTexCoord2f(0,0); rlVertex3f(-1,-1,1);
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
// Default shader loading (fixed-function)
//----------------------------------------------------------------------------------
static void rlLoadShaderDefault(void)
{
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

static void rlUnloadShaderDefault(void)
{
    if (RLGL.State.defaultShaderId > 0) rlUnloadShaderProgram(RLGL.State.defaultShaderId);
    RL_FREE(RLGL.State.defaultShaderLocs);
}

//----------------------------------------------------------------------------------
// Clip planes (module-level variables, same pattern as rlgl.h)
//----------------------------------------------------------------------------------
void rlSetClipPlanes(double nearPlane, double farPlane)
{
    // These are defined as module-level variables in rlgl.h
    // For external backends we just store them locally
    (void)nearPlane; (void)farPlane;
}

double rlGetCullDistanceNear(void) { return RL_CULL_DISTANCE_NEAR; }
double rlGetCullDistanceFar(void) { return RL_CULL_DISTANCE_FAR; }

//----------------------------------------------------------------------------------
// State pointer functions (GL 1.1 only, stubs here)
//----------------------------------------------------------------------------------
void rlEnableStatePointer(int vertexAttribType, void *buffer) { (void)vertexAttribType; (void)buffer; }
void rlDisableStatePointer(int vertexAttribType) { (void)vertexAttribType; }

//----------------------------------------------------------------------------------
// rlSetTexture (batch texture management)
//----------------------------------------------------------------------------------
void rlSetTexture(unsigned int id)
{
    if (id == 0) {
        if (RLGL.currentBatch->vertexBuffer[RLGL.currentBatch->currentBuffer].vCounter >=
            RLGL.currentBatch->vertexBuffer[RLGL.currentBatch->currentBuffer].elementCount * 4) {
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
                        ((RLGL.currentBatch->draws[RLGL.currentBatch->drawCounter - 1].vertexCount < 4) ? 1 :
                        (4 - (RLGL.currentBatch->draws[RLGL.currentBatch->drawCounter - 1].vertexCount % 4)));
                else
                    RLGL.currentBatch->draws[RLGL.currentBatch->drawCounter - 1].vertexAlignment = 0;

                if (!rlCheckRenderBatchLimit(RLGL.currentBatch->draws[RLGL.currentBatch->drawCounter - 1].vertexAlignment)) {
                    RLGL.currentBatch->vertexBuffer[RLGL.currentBatch->currentBuffer].vCounter +=
                        RLGL.currentBatch->draws[RLGL.currentBatch->drawCounter - 1].vertexAlignment;
                    RLGL.currentBatch->drawCounter++;
                }
            }

            if (RLGL.currentBatch->drawCounter >= RL_DEFAULT_BATCH_DRAWCALLS) rlDrawRenderBatch(RLGL.currentBatch);

            RLGL.currentBatch->draws[RLGL.currentBatch->drawCounter - 1].textureId = id;
            RLGL.currentBatch->draws[RLGL.currentBatch->drawCounter - 1].vertexCount = 0;
        }
    }
}

//----------------------------------------------------------------------------------
// Pixel data size helper
//----------------------------------------------------------------------------------
static int rlGetPixelDataSize(int width, int height, int format)
{
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
    return (int)((double)bpp/8.0*width*height);
}

//----------------------------------------------------------------------------------
// Headless render viewport stubs
//----------------------------------------------------------------------------------
void rlSetHeadlessRenderViewport(int width, int height) { (void)width; (void)height; }
void rlUnsetHeadlessRenderViewport(void) { }

//----------------------------------------------------------------------------------
// Software framebuffer access (for platform layer to blit to screen)
//----------------------------------------------------------------------------------
unsigned char *rlGetSoftwareFramebuffer(void) { return RLGL.colorBuffer; }
int rlGetSoftwareFramebufferWidth(void) { return RLGL.fbWidth; }
int rlGetSoftwareFramebufferHeight(void) { return RLGL.fbHeight; }

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

#endif // GRAPHICS_API_SOFTWARE
