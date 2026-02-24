/**********************************************************************************************
*
*   rl_unified_shader - Implementation
*
*   Generates native shader source (GLSL / HLSL) from a portable rlUnifiedShaderDesc
*   and compiles it via the active backend's rlLoadShaderCode().
*
*   Supported backends:
*     - GLSL Modern  (OpenGL 3.3, 4.3, ES 3.0):  in/out, texture(), layout(location=N)
*     - GLSL Legacy  (OpenGL 2.1, ES 2.0):        attribute/varying, texture2D(), gl_FragColor
*     - HLSL Modern  (D3D10, D3D11, D3D12):       cbuffer, Texture2D.Sample, SV_POSITION/SV_TARGET
*     - HLSL Legacy  (D3D9):                      register(c), tex2D, POSITION/COLOR0
*
*   LICENSE: zlib/libpng (same as raylib)
*
**********************************************************************************************/

#include "rl_unified_shader.h"
#include "rl_backend.h"
#include "utils.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <ctype.h>
#include <stdarg.h>

//----------------------------------------------------------------------------------
// Backend group detection (compile-time)
//----------------------------------------------------------------------------------
#define USL_GLSL_MODERN     1       // OpenGL 3.3, 4.3, ES 3.0
#define USL_GLSL_LEGACY     2       // OpenGL 2.1, ES 2.0
#define USL_HLSL_MODERN     3       // D3D10, D3D11, D3D12 (SM 4.0/5.0)
#define USL_HLSL_LEGACY     4       // D3D9 (SM 3.0)
#define USL_NONE            0       // Unsupported (GL 1.1, Vulkan, Metal)

#if defined(GRAPHICS_API_OPENGL_33) || defined(GRAPHICS_API_OPENGL_43) || defined(GRAPHICS_API_OPENGL_ES3)
    #define USL_BACKEND_GROUP   USL_GLSL_MODERN
#elif defined(GRAPHICS_API_OPENGL_21) || defined(GRAPHICS_API_OPENGL_ES2)
    #define USL_BACKEND_GROUP   USL_GLSL_LEGACY
#elif defined(GRAPHICS_API_DIRECT3D10) || defined(GRAPHICS_API_DIRECT3D11) || defined(GRAPHICS_API_DIRECT3D12)
    #define USL_BACKEND_GROUP   USL_HLSL_MODERN
#elif defined(GRAPHICS_API_DIRECT3D9)
    #define USL_BACKEND_GROUP   USL_HLSL_LEGACY
#else
    #define USL_BACKEND_GROUP   USL_NONE
#endif

// GLSL version string
#if defined(GRAPHICS_API_OPENGL_43)
    #define USL_GLSL_VERSION_STR    "#version 430\n"
#elif defined(GRAPHICS_API_OPENGL_33)
    #define USL_GLSL_VERSION_STR    "#version 330\n"
#elif defined(GRAPHICS_API_OPENGL_ES3)
    #define USL_GLSL_VERSION_STR    "#version 300 es\nprecision mediump float;\n"
#elif defined(GRAPHICS_API_OPENGL_ES2)
    #define USL_GLSL_VERSION_STR    "#version 100\nprecision mediump float;\n"
#elif defined(GRAPHICS_API_OPENGL_21)
    #define USL_GLSL_VERSION_STR    "#version 120\n"
#else
    #define USL_GLSL_VERSION_STR    ""
#endif

// HLSL shader model targets
#if defined(GRAPHICS_API_DIRECT3D9)
    #define USL_VS_TARGET   "vs_3_0"
    #define USL_PS_TARGET   "ps_3_0"
#elif defined(GRAPHICS_API_DIRECT3D10)
    #define USL_VS_TARGET   "vs_4_0"
    #define USL_PS_TARGET   "ps_4_0"
#else
    #define USL_VS_TARGET   "vs_5_0"
    #define USL_PS_TARGET   "ps_5_0"
#endif

//----------------------------------------------------------------------------------
// StringBuilder - dynamic string building
//----------------------------------------------------------------------------------
typedef struct {
    char *data;
    int len;
    int cap;
} StringBuilder;

static void sbInit(StringBuilder *sb)
{
    sb->cap = 2048;
    sb->data = (char *)RL_MALLOC(sb->cap);
    sb->data[0] = '\0';
    sb->len = 0;
}

static void sbEnsure(StringBuilder *sb, int extra)
{
    while (sb->len + extra + 1 > sb->cap) {
        sb->cap *= 2;
        sb->data = (char *)RL_REALLOC(sb->data, sb->cap);
    }
}

static void sbAppend(StringBuilder *sb, const char *str)
{
    int slen = (int)strlen(str);
    sbEnsure(sb, slen);
    memcpy(sb->data + sb->len, str, slen);
    sb->len += slen;
    sb->data[sb->len] = '\0';
}

static void sbAppendN(StringBuilder *sb, const char *str, int n)
{
    sbEnsure(sb, n);
    memcpy(sb->data + sb->len, str, n);
    sb->len += n;
    sb->data[sb->len] = '\0';
}

static void sbAppendChar(StringBuilder *sb, char c)
{
    sbEnsure(sb, 1);
    sb->data[sb->len++] = c;
    sb->data[sb->len] = '\0';
}

static void sbPrintf(StringBuilder *sb, const char *fmt, ...)
{
    char buf[1024];
    va_list args;
    va_start(args, fmt);
    int n = vsnprintf(buf, sizeof(buf), fmt, args);
    va_end(args);
    if (n > 0) sbAppendN(sb, buf, n);
}

static char *sbFinish(StringBuilder *sb)
{
    char *result = sb->data;
    sb->data = NULL;
    sb->len = 0;
    sb->cap = 0;
    return result;
}

static void sbFree(StringBuilder *sb)
{
    if (sb->data) RL_FREE(sb->data);
    sb->data = NULL;
    sb->len = 0;
    sb->cap = 0;
}

//----------------------------------------------------------------------------------
// Utility functions
//----------------------------------------------------------------------------------
static bool uslIsWordChar(char c)
{
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
           (c >= '0' && c <= '9') || c == '_';
}

// Check if 'word' matches at position 'pos' in 'src' as a complete word (word boundaries on both sides)
static bool uslMatchWord(const char *src, int pos, const char *word)
{
    int wlen = (int)strlen(word);
    if (strncmp(src + pos, word, wlen) != 0) return false;
    if (pos > 0 && uslIsWordChar(src[pos - 1])) return false;
    if (uslIsWordChar(src[pos + wlen])) return false;
    return true;
}

// Check if 'prefix' matches at position 'pos' with left word boundary (but no right boundary check)
static bool uslMatchPrefix(const char *src, int pos, const char *prefix)
{
    int plen = (int)strlen(prefix);
    if (strncmp(src + pos, prefix, plen) != 0) return false;
    if (pos > 0 && uslIsWordChar(src[pos - 1])) return false;
    return true;
}

// Extract comma-separated arguments from a function call.
// src[startPos] is the first character after the opening '('.
// Returns number of arguments found. Sets *endPos to position after closing ')'.
static int uslExtractArgs(const char *src, int startPos,
                           int argStarts[], int argLens[], int maxArgs, int *endPos)
{
    int count = 0;
    int depth = 0;
    int i = startPos;
    int argStart = startPos;

    // Skip leading whitespace
    while (src[i] == ' ' || src[i] == '\t') i++;
    argStart = i;

    while (src[i]) {
        if (src[i] == '(') {
            depth++;
        } else if (src[i] == ')') {
            if (depth == 0) {
                // End of function call
                if (i > argStart && count < maxArgs) {
                    argStarts[count] = argStart;
                    // Trim trailing whitespace
                    int end = i;
                    while (end > argStart && (src[end - 1] == ' ' || src[end - 1] == '\t')) end--;
                    argLens[count] = end - argStart;
                    count++;
                }
                *endPos = i + 1;
                return count;
            }
            depth--;
        } else if (src[i] == ',' && depth == 0) {
            if (count < maxArgs) {
                argStarts[count] = argStart;
                int end = i;
                while (end > argStart && (src[end - 1] == ' ' || src[end - 1] == '\t')) end--;
                argLens[count] = end - argStart;
                count++;
            }
            i++;
            while (src[i] == ' ' || src[i] == '\t') i++;
            argStart = i;
            continue;
        }
        i++;
    }

    // Unbalanced parens — return what we have
    *endPos = i;
    return count;
}

//----------------------------------------------------------------------------------
// Type name mapping
//----------------------------------------------------------------------------------
static const char *uslGLSLTypeName(int type)
{
    switch (type) {
        case RL_STYPE_FLOAT: return "float";
        case RL_STYPE_VEC2:  return "vec2";
        case RL_STYPE_VEC3:  return "vec3";
        case RL_STYPE_VEC4:  return "vec4";
        case RL_STYPE_INT:   return "int";
        case RL_STYPE_MAT4:  return "mat4";
        case RL_STYPE_BOOL:  return "bool";
        case RL_STYPE_MAT3:  return "mat3";
        default: return "float";
    }
}

static const char *uslHLSLTypeName(int type)
{
    switch (type) {
        case RL_STYPE_FLOAT: return "float";
        case RL_STYPE_VEC2:  return "float2";
        case RL_STYPE_VEC3:  return "float3";
        case RL_STYPE_VEC4:  return "float4";
        case RL_STYPE_INT:   return "int";
        case RL_STYPE_MAT4:  return "float4x4";
        case RL_STYPE_BOOL:  return "bool";
        case RL_STYPE_MAT3:  return "float3x3";
        default: return "float";
    }
}

// Map attribute location to standard raylib GLSL attribute name
static void uslGetGLSLInputName(int location, char *out, int maxLen)
{
    static const char *standardNames[] = {
        "vertexPosition",       // 0
        "vertexTexCoord",       // 1
        "vertexNormal",         // 2
        "vertexColor",          // 3
        "vertexTangent",        // 4
        "vertexTexCoord2",      // 5
    };
    if (location >= 0 && location < 6) {
        snprintf(out, maxLen, "%s", standardNames[location]);
    } else {
        snprintf(out, maxLen, "vertexAttrib%d", location);
    }
}

// Map attribute location to HLSL input semantic
static void uslGetHLSLInputSemantic(int location, char *out, int maxLen)
{
    switch (location) {
        case 0: snprintf(out, maxLen, "POSITION");  break;
        case 1: snprintf(out, maxLen, "TEXCOORD0"); break;
        case 2: snprintf(out, maxLen, "NORMAL");    break;
        case 3: snprintf(out, maxLen, "COLOR0");    break;
        case 4: snprintf(out, maxLen, "TANGENT");   break;
        case 5: snprintf(out, maxLen, "TEXCOORD1"); break;
        default: snprintf(out, maxLen, "TEXCOORD%d", location); break;
    }
}

//----------------------------------------------------------------------------------
// Body code transpiler
//
// Transforms portable body code to native shader syntax.
// Handles: type keywords, function calls, input/output field mapping.
//
// Parameters:
//   body     - Portable shader body source
//   isHLSL   - true for D3D backends, false for OpenGL
//   isLegacy - true for D3D9 / GLES2 / GL2.1
//   isVertex - true for vertex shader, false for fragment
//   desc     - The shader descriptor (for input/varying name lookups)
//----------------------------------------------------------------------------------
static char *uslTranspileBody(const char *body, bool isHLSL, bool isLegacy,
                               bool isVertex, const rlUnifiedShaderDesc *desc)
{
    StringBuilder sb;
    sbInit(&sb);
    int len = (int)strlen(body);
    int i = 0;

    while (i < len) {
        // ---- sampleCubemap(texName, dir) ----
        if (uslMatchPrefix(body, i, "sampleCubemap(")) {
            int openParen = i + 13; // position of '('
            i = openParen + 1;      // skip past '('
            int argS[2], argL[2], endPos;
            int argc = uslExtractArgs(body, i, argS, argL, 2, &endPos);
            if (argc == 2) {
                char texName[64] = {0};
                snprintf(texName, sizeof(texName), "%.*s", argL[0], body + argS[0]);

                char dirRaw[512] = {0};
                snprintf(dirRaw, sizeof(dirRaw), "%.*s", argL[1], body + argS[1]);
                char *dirTranspiled = uslTranspileBody(dirRaw, isHLSL, isLegacy, isVertex, desc);

                if (isHLSL && !isLegacy) {
                    // HLSL Modern: texName.Sample(sampler_texName, dir)
                    sbPrintf(&sb, "%s.Sample(sampler_%s, %s)", texName, texName, dirTranspiled);
                } else if (isHLSL && isLegacy) {
                    // HLSL Legacy (D3D9): texCUBE(texName, dir)
                    sbPrintf(&sb, "texCUBE(%s, %s)", texName, dirTranspiled);
                } else if (!isHLSL && !isLegacy) {
                    // GLSL Modern: texture(texName, dir)
                    sbPrintf(&sb, "texture(%s, %s)", texName, dirTranspiled);
                } else {
                    // GLSL Legacy: textureCube(texName, dir)
                    sbPrintf(&sb, "textureCube(%s, %s)", texName, dirTranspiled);
                }

                RL_FREE(dirTranspiled);
            } else {
                sbAppend(&sb, "sampleCubemap(");
            }
            i = endPos;
            continue;
        }

        // ---- sampleTexture(texName, uv) ----
        if (uslMatchPrefix(body, i, "sampleTexture(")) {
            int openParen = i + 13; // position of '('
            i = openParen + 1;      // skip past '('
            int argS[2], argL[2], endPos;
            int argc = uslExtractArgs(body, i, argS, argL, 2, &endPos);
            if (argc == 2) {
                char texName[64] = {0};
                snprintf(texName, sizeof(texName), "%.*s", argL[0], body + argS[0]);

                // Recursively transpile the UV expression (may contain input. references etc.)
                char uvRaw[512] = {0};
                snprintf(uvRaw, sizeof(uvRaw), "%.*s", argL[1], body + argS[1]);
                char *uvTranspiled = uslTranspileBody(uvRaw, isHLSL, isLegacy, isVertex, desc);

                if (isHLSL && !isLegacy) {
                    // HLSL Modern: texture0.Sample(sampler_texture0, uv)
                    sbPrintf(&sb, "%s.Sample(sampler_%s, %s)", texName, texName, uvTranspiled);
                } else if (isHLSL && isLegacy) {
                    // HLSL Legacy (D3D9): tex2D(texture0, uv)
                    sbPrintf(&sb, "tex2D(%s, %s)", texName, uvTranspiled);
                } else if (!isHLSL && !isLegacy) {
                    // GLSL Modern: texture(texture0, uv)
                    sbPrintf(&sb, "texture(%s, %s)", texName, uvTranspiled);
                } else {
                    // GLSL Legacy: texture2D(texture0, uv)
                    sbPrintf(&sb, "texture2D(%s, %s)", texName, uvTranspiled);
                }

                RL_FREE(uvTranspiled);
            } else {
                // Malformed call — pass through
                sbAppend(&sb, "sampleTexture(");
            }
            i = endPos;
            continue;
        }

        // ---- mul(a, b) → GLSL: ((a) * (b)), HLSL: unchanged ----
        if (!isHLSL && uslMatchWord(body, i, "mul") && body[i + 3] == '(') {
            i += 4; // skip "mul("
            int argS[2], argL[2], endPos;
            int argc = uslExtractArgs(body, i, argS, argL, 2, &endPos);
            if (argc == 2) {
                char a[512] = {0}, b[512] = {0};
                snprintf(a, sizeof(a), "%.*s", argL[0], body + argS[0]);
                snprintf(b, sizeof(b), "%.*s", argL[1], body + argS[1]);
                // Recursively transpile arguments
                char *aT = uslTranspileBody(a, isHLSL, isLegacy, isVertex, desc);
                char *bT = uslTranspileBody(b, isHLSL, isLegacy, isVertex, desc);
                sbPrintf(&sb, "((%s) * (%s))", aT, bT);
                RL_FREE(aT);
                RL_FREE(bT);
            } else {
                sbAppend(&sb, "mul(");
            }
            i = endPos;
            continue;
        }

        // ---- GLSL function name mapping: atan2→atan ----
        if (!isHLSL) {
            if (uslMatchWord(body, i, "atan2") && body[i + 5] == '(') {
                sbAppend(&sb, "atan"); i += 5; continue;
            }
        }

        // ---- HLSL type replacements: vec4→float4, vec3→float3, vec2→float2, mat4→float4x4 ----
        if (isHLSL) {
            if (uslMatchWord(body, i, "vec4"))  { sbAppend(&sb, "float4");   i += 4; continue; }
            if (uslMatchWord(body, i, "vec3"))  { sbAppend(&sb, "float3");   i += 4; continue; }
            if (uslMatchWord(body, i, "vec2"))  { sbAppend(&sb, "float2");   i += 4; continue; }
            if (uslMatchWord(body, i, "mat4"))  { sbAppend(&sb, "float4x4"); i += 4; continue; }
            if (uslMatchWord(body, i, "mat3"))  { sbAppend(&sb, "float3x3"); i += 4; continue; }

            // GLSL → HLSL function name mapping
            if (uslMatchWord(body, i, "mix") && body[i + 3] == '(') {
                sbAppend(&sb, "lerp"); i += 3; continue;
            }
            if (uslMatchWord(body, i, "fract") && body[i + 5] == '(') {
                sbAppend(&sb, "frac"); i += 5; continue;
            }
        }

        // ---- GLSL input. and output. mapping ----
        if (!isHLSL) {
            // input.fieldName → attribute name (VS) or varying name (FS)
            if (uslMatchPrefix(body, i, "input.")) {
                int fieldStart = i + 6;
                int fieldEnd = fieldStart;
                while (fieldEnd < len && uslIsWordChar(body[fieldEnd])) fieldEnd++;
                int fieldLen = fieldEnd - fieldStart;

                bool found = false;
                if (isVertex) {
                    // Map to GLSL attribute name by looking up location
                    for (int j = 0; j < desc->inputCount; j++) {
                        if ((int)strlen(desc->inputs[j].name) == fieldLen &&
                            strncmp(body + fieldStart, desc->inputs[j].name, fieldLen) == 0) {
                            char attrName[128];
                            uslGetGLSLInputName(desc->inputs[j].location, attrName, sizeof(attrName));
                            sbAppend(&sb, attrName);
                            found = true;
                            break;
                        }
                    }
                } else {
                    // In fragment shader, input.X maps to varying name
                    for (int j = 0; j < desc->varyingCount; j++) {
                        if ((int)strlen(desc->varyings[j].name) == fieldLen &&
                            strncmp(body + fieldStart, desc->varyings[j].name, fieldLen) == 0) {
                            sbAppend(&sb, desc->varyings[j].name);
                            found = true;
                            break;
                        }
                    }
                }
                if (!found) {
                    // Not found in descriptor — pass through unchanged
                    sbAppendN(&sb, body + i, fieldEnd - i);
                }
                i = fieldEnd;
                continue;
            }

            // output.fieldName → gl_Position (VS), varying name (VS), finalColor/gl_FragColor (FS)
            if (uslMatchPrefix(body, i, "output.")) {
                int fieldStart = i + 7;
                int fieldEnd = fieldStart;
                while (fieldEnd < len && uslIsWordChar(body[fieldEnd])) fieldEnd++;
                int fieldLen = fieldEnd - fieldStart;

                if (isVertex && fieldLen == 8 && strncmp(body + fieldStart, "position", 8) == 0) {
                    sbAppend(&sb, "gl_Position");
                } else if (!isVertex && fieldLen == 5 && strncmp(body + fieldStart, "color", 5) == 0) {
                    sbAppend(&sb, isLegacy ? "gl_FragColor" : "finalColor");
                } else if (isVertex) {
                    // Varying output — use the varying name directly
                    sbAppendN(&sb, body + fieldStart, fieldLen);
                } else {
                    // Unknown FS output — pass through with output. prefix
                    sbAppendN(&sb, body + i, fieldEnd - i);
                }
                i = fieldEnd;
                continue;
            }
        }

        // ---- Default: copy character as-is ----
        sbAppendChar(&sb, body[i]);
        i++;
    }

    return sbFinish(&sb);
}

//----------------------------------------------------------------------------------
// GLSL source generators
//----------------------------------------------------------------------------------
#if (USL_BACKEND_GROUP == USL_GLSL_MODERN) || (USL_BACKEND_GROUP == USL_GLSL_LEGACY)

static char *uslGenerateGLSLVertex(const rlUnifiedShaderDesc *desc,
                                    bool isLegacy)
{
    StringBuilder sb;
    sbInit(&sb);

    // Version string
    sbAppend(&sb, USL_GLSL_VERSION_STR);
    sbAppend(&sb, "\n");

    // Uniforms
    for (int i = 0; i < desc->vsUniformCount; i++) {
        sbPrintf(&sb, "uniform %s %s;\n", uslGLSLTypeName(desc->vsUniforms[i].type),
                 desc->vsUniforms[i].name);
    }

    // Vertex attribute inputs
    for (int i = 0; i < desc->inputCount; i++) {
        char name[128];
        uslGetGLSLInputName(desc->inputs[i].location, name, sizeof(name));
        if (isLegacy) {
            sbPrintf(&sb, "attribute %s %s;\n", uslGLSLTypeName(desc->inputs[i].type), name);
        } else {
            sbPrintf(&sb, "layout(location = %d) in %s %s;\n",
                     desc->inputs[i].location, uslGLSLTypeName(desc->inputs[i].type), name);
        }
    }

    // Varyings (outputs from VS)
    for (int i = 0; i < desc->varyingCount; i++) {
        if (isLegacy) {
            sbPrintf(&sb, "varying %s %s;\n", uslGLSLTypeName(desc->varyings[i].type),
                     desc->varyings[i].name);
        } else {
            sbPrintf(&sb, "out %s %s;\n", uslGLSLTypeName(desc->varyings[i].type),
                     desc->varyings[i].name);
        }
    }

    // Preamble (struct definitions, helper functions, etc.)
    if (desc->vsPreamble && desc->vsPreamble[0]) {
        sbAppend(&sb, "\n");
        char *preamble = uslTranspileBody(desc->vsPreamble, false, isLegacy, true, desc);
        sbAppend(&sb, preamble);
        sbAppend(&sb, "\n");
        RL_FREE(preamble);
    }

    // Main function
    sbAppend(&sb, "\nvoid main() {\n");

    // Transpile and insert body code
    char *body = uslTranspileBody(desc->vertexBody, false, isLegacy, true, desc);
    // Indent each line
    const char *line = body;
    while (*line) {
        const char *eol = strchr(line, '\n');
        if (eol) {
            sbAppend(&sb, "    ");
            sbAppendN(&sb, line, (int)(eol - line + 1));
            line = eol + 1;
        } else {
            if (*line) {
                sbAppend(&sb, "    ");
                sbAppend(&sb, line);
                sbAppend(&sb, "\n");
            }
            break;
        }
    }
    RL_FREE(body);

    sbAppend(&sb, "}\n");
    return sbFinish(&sb);
}

static char *uslGenerateGLSLFragment(const rlUnifiedShaderDesc *desc,
                                      bool isLegacy)
{
    StringBuilder sb;
    sbInit(&sb);

    // Version string
    sbAppend(&sb, USL_GLSL_VERSION_STR);
    sbAppend(&sb, "\n");

    // Uniforms
    for (int i = 0; i < desc->fsUniformCount; i++) {
        sbPrintf(&sb, "uniform %s %s;\n", uslGLSLTypeName(desc->fsUniforms[i].type),
                 desc->fsUniforms[i].name);
    }

    // Textures
    for (int i = 0; i < desc->textureCount; i++) {
        if (desc->textures[i].type == RL_STEX_CUBE) {
            sbPrintf(&sb, "uniform samplerCube %s;\n", desc->textures[i].name);
        } else {
            sbPrintf(&sb, "uniform sampler2D %s;\n", desc->textures[i].name);
        }
    }

    // Varyings (inputs to FS)
    for (int i = 0; i < desc->varyingCount; i++) {
        if (isLegacy) {
            sbPrintf(&sb, "varying %s %s;\n", uslGLSLTypeName(desc->varyings[i].type),
                     desc->varyings[i].name);
        } else {
            sbPrintf(&sb, "in %s %s;\n", uslGLSLTypeName(desc->varyings[i].type),
                     desc->varyings[i].name);
        }
    }

    // Output declaration (modern GLSL only)
    if (!isLegacy) {
        sbAppend(&sb, "out vec4 finalColor;\n");
    }

    // Preamble (struct definitions, helper functions, etc.)
    if (desc->fsPreamble && desc->fsPreamble[0]) {
        sbAppend(&sb, "\n");
        char *preamble = uslTranspileBody(desc->fsPreamble, false, isLegacy, false, desc);
        sbAppend(&sb, preamble);
        sbAppend(&sb, "\n");
        RL_FREE(preamble);
    }

    // Main function
    sbAppend(&sb, "\nvoid main() {\n");

    char *body = uslTranspileBody(desc->fragmentBody, false, isLegacy, false, desc);
    const char *line = body;
    while (*line) {
        const char *eol = strchr(line, '\n');
        if (eol) {
            sbAppend(&sb, "    ");
            sbAppendN(&sb, line, (int)(eol - line + 1));
            line = eol + 1;
        } else {
            if (*line) {
                sbAppend(&sb, "    ");
                sbAppend(&sb, line);
                sbAppend(&sb, "\n");
            }
            break;
        }
    }
    RL_FREE(body);

    sbAppend(&sb, "}\n");
    return sbFinish(&sb);
}

#endif // GLSL backends

//----------------------------------------------------------------------------------
// HLSL source generators
//----------------------------------------------------------------------------------
#if (USL_BACKEND_GROUP == USL_HLSL_MODERN) || (USL_BACKEND_GROUP == USL_HLSL_LEGACY)

static char *uslGenerateHLSLVertex(const rlUnifiedShaderDesc *desc,
                                    bool isLegacy)
{
    StringBuilder sb;
    sbInit(&sb);

    // Uniform declarations
    for (int i = 0; i < desc->vsUniformCount; i++) {
        const char *typeName = uslHLSLTypeName(desc->vsUniforms[i].type);
        if (isLegacy) {
            // D3D9: bare uniform with constant register
            sbPrintf(&sb, "%s %s : register(c%d);\n",
                     typeName, desc->vsUniforms[i].name, desc->vsUniforms[i].binding);
        } else {
            // D3D10/11/12: cbuffer wrapping
            sbPrintf(&sb, "cbuffer cb_%s : register(b%d) { %s %s; };\n",
                     desc->vsUniforms[i].name, desc->vsUniforms[i].binding,
                     typeName, desc->vsUniforms[i].name);
        }
    }

    // VSInput struct
    sbAppend(&sb, "struct VSInput {\n");
    for (int i = 0; i < desc->inputCount; i++) {
        char semantic[32];
        uslGetHLSLInputSemantic(desc->inputs[i].location, semantic, sizeof(semantic));
        sbPrintf(&sb, "    %s %s : %s;\n",
                 uslHLSLTypeName(desc->inputs[i].type), desc->inputs[i].name, semantic);
    }
    sbAppend(&sb, "};\n");

    // VSOutput struct (position + varyings)
    sbAppend(&sb, "struct VSOutput {\n");
    sbPrintf(&sb, "    float4 position : %s;\n", isLegacy ? "POSITION" : "SV_POSITION");
    for (int i = 0; i < desc->varyingCount; i++) {
        sbPrintf(&sb, "    %s %s : TEXCOORD%d;\n",
                 uslHLSLTypeName(desc->varyings[i].type), desc->varyings[i].name, i);
    }
    sbAppend(&sb, "};\n");
    // Preamble (struct definitions, helper functions, etc.)
    if (desc->vsPreamble && desc->vsPreamble[0]) {
        sbAppend(&sb, "\n");
        char *preamble = uslTranspileBody(desc->vsPreamble, true, isLegacy, true, desc);
        sbAppend(&sb, preamble);
        sbAppend(&sb, "\n");
        RL_FREE(preamble);
    }
    // Entry point
    sbAppend(&sb, "VSOutput VSMain(VSInput input) {\n");
    sbAppend(&sb, "    VSOutput output = (VSOutput)0;\n");

    // Transpile and insert body code
    char *body = uslTranspileBody(desc->vertexBody, true, isLegacy, true, desc);
    const char *line = body;
    while (*line) {
        const char *eol = strchr(line, '\n');
        if (eol) {
            sbAppend(&sb, "    ");
            sbAppendN(&sb, line, (int)(eol - line + 1));
            line = eol + 1;
        } else {
            if (*line) {
                sbAppend(&sb, "    ");
                sbAppend(&sb, line);
                sbAppend(&sb, "\n");
            }
            break;
        }
    }
    RL_FREE(body);

    sbAppend(&sb, "    return output;\n");
    sbAppend(&sb, "}\n");
    return sbFinish(&sb);
}

static char *uslGenerateHLSLFragment(const rlUnifiedShaderDesc *desc,
                                      bool isLegacy)
{
    StringBuilder sb;
    sbInit(&sb);

    // Uniform declarations
    for (int i = 0; i < desc->fsUniformCount; i++) {
        const char *typeName = uslHLSLTypeName(desc->fsUniforms[i].type);
        if (isLegacy) {
            sbPrintf(&sb, "%s %s : register(c%d);\n",
                     typeName, desc->fsUniforms[i].name, desc->fsUniforms[i].binding);
        } else {
            sbPrintf(&sb, "cbuffer cb_%s : register(b%d) { %s %s; };\n",
                     desc->fsUniforms[i].name, desc->fsUniforms[i].binding,
                     typeName, desc->fsUniforms[i].name);
        }
    }

    // Texture and sampler declarations
    for (int i = 0; i < desc->textureCount; i++) {
        if (isLegacy) {
            // D3D9: combined sampler
            if (desc->textures[i].type == RL_STEX_CUBE) {
                sbPrintf(&sb, "samplerCUBE %s : register(s%d);\n",
                         desc->textures[i].name, desc->textures[i].slot);
            } else {
                sbPrintf(&sb, "sampler2D %s : register(s%d);\n",
                         desc->textures[i].name, desc->textures[i].slot);
            }
        } else {
            // D3D10/11/12: separate Texture + SamplerState
            if (desc->textures[i].type == RL_STEX_CUBE) {
                sbPrintf(&sb, "TextureCube %s : register(t%d);\n",
                         desc->textures[i].name, desc->textures[i].slot);
            } else {
                sbPrintf(&sb, "Texture2D %s : register(t%d);\n",
                         desc->textures[i].name, desc->textures[i].slot);
            }
            sbPrintf(&sb, "SamplerState sampler_%s : register(s%d);\n",
                     desc->textures[i].name, desc->textures[i].slot);
        }
    }

    // PSInput struct (matches VSOutput)
    sbAppend(&sb, "struct PSInput {\n");
    sbPrintf(&sb, "    float4 position : %s;\n", isLegacy ? "POSITION" : "SV_POSITION");
    for (int i = 0; i < desc->varyingCount; i++) {
        sbPrintf(&sb, "    %s %s : TEXCOORD%d;\n",
                 uslHLSLTypeName(desc->varyings[i].type), desc->varyings[i].name, i);
    }
    sbAppend(&sb, "};\n");

    // PSOutput struct
    sbAppend(&sb, "struct PSOutput {\n");
    sbPrintf(&sb, "    float4 color : %s;\n", isLegacy ? "COLOR0" : "SV_TARGET");
    sbAppend(&sb, "};\n");
    // Preamble (struct definitions, helper functions, etc.)
    if (desc->fsPreamble && desc->fsPreamble[0]) {
        sbAppend(&sb, "\n");
        char *preamble = uslTranspileBody(desc->fsPreamble, true, isLegacy, false, desc);
        sbAppend(&sb, preamble);
        sbAppend(&sb, "\n");
        RL_FREE(preamble);
    }
    // Entry point
    sbAppend(&sb, "PSOutput PSMain(PSInput input) {\n");
    sbAppend(&sb, "    PSOutput output = (PSOutput)0;\n");

    // Transpile and insert body code
    char *body = uslTranspileBody(desc->fragmentBody, true, isLegacy, false, desc);
    const char *line = body;
    while (*line) {
        const char *eol = strchr(line, '\n');
        if (eol) {
            sbAppend(&sb, "    ");
            sbAppendN(&sb, line, (int)(eol - line + 1));
            line = eol + 1;
        } else {
            if (*line) {
                sbAppend(&sb, "    ");
                sbAppend(&sb, line);
                sbAppend(&sb, "\n");
            }
            break;
        }
    }
    RL_FREE(body);

    sbAppend(&sb, "    return output;\n");
    sbAppend(&sb, "}\n");
    return sbFinish(&sb);
}

#endif // HLSL backends

//----------------------------------------------------------------------------------
// Public API implementation
//----------------------------------------------------------------------------------

char *rlGenerateVertexShaderSource(const rlUnifiedShaderDesc *desc)
{
    if (!desc || !desc->vertexBody) return NULL;

#if (USL_BACKEND_GROUP == USL_GLSL_MODERN)
    return uslGenerateGLSLVertex(desc, false);
#elif (USL_BACKEND_GROUP == USL_GLSL_LEGACY)
    return uslGenerateGLSLVertex(desc, true);
#elif (USL_BACKEND_GROUP == USL_HLSL_MODERN)
    return uslGenerateHLSLVertex(desc, false);
#elif (USL_BACKEND_GROUP == USL_HLSL_LEGACY)
    return uslGenerateHLSLVertex(desc, true);
#else
    TRACELOG(RL_LOG_WARNING, "SHADER: Unified shaders not supported for the current backend");
    return NULL;
#endif
}

char *rlGenerateFragmentShaderSource(const rlUnifiedShaderDesc *desc)
{
    if (!desc || !desc->fragmentBody) return NULL;

#if (USL_BACKEND_GROUP == USL_GLSL_MODERN)
    return uslGenerateGLSLFragment(desc, false);
#elif (USL_BACKEND_GROUP == USL_GLSL_LEGACY)
    return uslGenerateGLSLFragment(desc, true);
#elif (USL_BACKEND_GROUP == USL_HLSL_MODERN)
    return uslGenerateHLSLFragment(desc, false);
#elif (USL_BACKEND_GROUP == USL_HLSL_LEGACY)
    return uslGenerateHLSLFragment(desc, true);
#else
    TRACELOG(RL_LOG_WARNING, "SHADER: Unified shaders not supported for the current backend");
    return NULL;
#endif
}

unsigned int rlLoadUnifiedShader(const rlUnifiedShaderDesc *desc)
{
    if (!desc) return 0;

    char *vsSource = rlGenerateVertexShaderSource(desc);
    char *fsSource = rlGenerateFragmentShaderSource(desc);

    if (!vsSource || !fsSource) {
        TRACELOG(RL_LOG_WARNING, "SHADER: Failed to generate unified shader source");
        if (vsSource) RL_FREE(vsSource);
        if (fsSource) RL_FREE(fsSource);
        return 0;
    }

    TRACELOG(RL_LOG_INFO, "SHADER: Unified shader source generated for backend group %d", USL_BACKEND_GROUP);

    // Compile using the backend's existing rlLoadShaderCode
    unsigned int shaderId = rlLoadShaderCode(vsSource, fsSource);

    RL_FREE(vsSource);
    RL_FREE(fsSource);

    if (shaderId > 0) {
        TRACELOG(RL_LOG_INFO, "SHADER: [ID %i] Unified shader loaded successfully", shaderId);
    } else {
        TRACELOG(RL_LOG_WARNING, "SHADER: Unified shader compilation failed");
    }

    return shaderId;
}

void rlFreeShaderSource(char *source)
{
    if (source) RL_FREE(source);
}

rlUnifiedShaderDesc rlGetDefaultUnifiedShaderDesc(void)
{
    rlUnifiedShaderDesc desc = { 0 };

    // Standard raylib vertex inputs
    desc.inputs[0] = (rlShaderInput){ "position", RL_STYPE_VEC3, 0 };
    desc.inputs[1] = (rlShaderInput){ "texcoord", RL_STYPE_VEC2, 1 };
    desc.inputs[2] = (rlShaderInput){ "normal",   RL_STYPE_VEC3, 2 };
    desc.inputs[3] = (rlShaderInput){ "color",    RL_STYPE_VEC4, 3 };
    desc.inputCount = 4;

    // VS -> FS varyings
    desc.varyings[0] = (rlShaderVarying){ "fragTexCoord", RL_STYPE_VEC2 };
    desc.varyings[1] = (rlShaderVarying){ "fragColor",    RL_STYPE_VEC4 };
    desc.varyingCount = 2;

    // Vertex shader uniforms
    desc.vsUniforms[0] = (rlShaderUniform){ "mvp", RL_STYPE_MAT4, 0 };
    desc.vsUniformCount = 1;

    // Fragment shader uniforms
    desc.fsUniforms[0] = (rlShaderUniform){ "colDiffuse", RL_STYPE_VEC4, 0 };
    desc.fsUniformCount = 1;

    // Textures
    desc.textures[0] = (rlShaderTexture){ "texture0", 0 };
    desc.textureCount = 1;

    // Portable body code
    desc.vertexBody =
        "output.position = mul(mvp, vec4(input.position, 1.0));\n"
        "output.fragTexCoord = input.texcoord;\n"
        "output.fragColor = input.color;\n";

    desc.fragmentBody =
        "vec4 texColor = sampleTexture(texture0, input.fragTexCoord);\n"
        "output.color = texColor * input.fragColor * colDiffuse;\n";

    return desc;
}
