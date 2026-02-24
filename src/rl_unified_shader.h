/**********************************************************************************************
*
*   rl_unified_shader - Cross-backend shader abstraction for rlgl
*
*   Write shaders ONCE using a portable syntax and have them compile on any backend:
*   OpenGL (all versions), Direct3D 9/10/11/12, Vulkan (stub), Metal (stub).
*
*   PORTABLE SHADER BODY SYNTAX:
*
*     Types:      vec2, vec3, vec4, mat4, float, int   (GLSL-style, auto-mapped to HLSL equivalents)
*     Multiply:   mul(matrix, vector)                  (HLSL-native, auto-converted to (m * v) for GLSL)
*     Sampling:   sampleTexture(texName, uv)           (auto-mapped per backend)
*                 sampleCubemap(texName, dir)           (cubemap sampling, auto-mapped per backend)
*     Functions:  mix(a, b, t)                          (GLSL mix / HLSL lerp)
*                 fract(x)                              (GLSL fract / HLSL frac)
*     Inputs:     input.fieldName                      (VS: vertex attribute, FS: varying)
*     Outputs:    output.position                      (VS clip-space position)
*                 output.<varyingName>                  (VS: pass data to FS)
*                 output.color                         (FS: final pixel color)
*
*   USAGE EXAMPLE:
*
*     rlUnifiedShaderDesc desc = { 0 };
*
*     desc.inputs[0] = (rlShaderInput){ "position", RL_STYPE_VEC3, 0 };
*     desc.inputs[1] = (rlShaderInput){ "texcoord", RL_STYPE_VEC2, 1 };
*     desc.inputs[2] = (rlShaderInput){ "normal",   RL_STYPE_VEC3, 2 };
*     desc.inputs[3] = (rlShaderInput){ "color",    RL_STYPE_VEC4, 3 };
*     desc.inputCount = 4;
*
*     desc.varyings[0] = (rlShaderVarying){ "fragTexCoord", RL_STYPE_VEC2 };
*     desc.varyings[1] = (rlShaderVarying){ "fragColor",    RL_STYPE_VEC4 };
*     desc.varyingCount = 2;
*
*     desc.vsUniforms[0] = (rlShaderUniform){ "mvp", RL_STYPE_MAT4, 0 };
*     desc.vsUniformCount = 1;
*
*     desc.fsUniforms[0] = (rlShaderUniform){ "colDiffuse", RL_STYPE_VEC4, 0 };
*     desc.fsUniformCount = 1;
*
*     desc.textures[0] = (rlShaderTexture){ "texture0", 0 };
*     desc.textureCount = 1;
*
*     desc.vertexBody =
*         "output.position = mul(mvp, vec4(input.position, 1.0));\n"
*         "output.fragTexCoord = input.texcoord;\n"
*         "output.fragColor = input.color;\n";
*
*     desc.fragmentBody =
*         "vec4 texColor = sampleTexture(texture0, input.fragTexCoord);\n"
*         "output.color = texColor * input.fragColor * colDiffuse;\n";
*
*     unsigned int shaderId = rlLoadUnifiedShader(&desc);
*
*   LICENSE: zlib/libpng (same as raylib)
*
**********************************************************************************************/

#ifndef RL_UNIFIED_SHADER_H
#define RL_UNIFIED_SHADER_H

#include "rlgl.h"

//----------------------------------------------------------------------------------
// Defines
//----------------------------------------------------------------------------------
#ifndef RL_MAX_SHADER_INPUTS
    #define RL_MAX_SHADER_INPUTS    8
#endif
#ifndef RL_MAX_SHADER_VARYINGS
    #define RL_MAX_SHADER_VARYINGS  8
#endif
#ifndef RL_MAX_SHADER_UNIFORMS
    #define RL_MAX_SHADER_UNIFORMS  16
#endif
#ifndef RL_MAX_SHADER_TEXTURES
    #define RL_MAX_SHADER_TEXTURES  8
#endif

//----------------------------------------------------------------------------------
// Types - Portable shader data types
//----------------------------------------------------------------------------------
typedef enum {
    RL_STYPE_FLOAT = 0,     // float / float
    RL_STYPE_VEC2,          // vec2  / float2
    RL_STYPE_VEC3,          // vec3  / float3
    RL_STYPE_VEC4,          // vec4  / float4
    RL_STYPE_INT,           // int   / int
    RL_STYPE_MAT4,          // mat4  / float4x4
    RL_STYPE_BOOL,          // bool  / bool
    RL_STYPE_MAT3,          // mat3  / float3x3
} rlShaderDataType;

// Texture sampler types
typedef enum {
    RL_STEX_2D = 0,         // sampler2D / Texture2D
    RL_STEX_CUBE,           // samplerCube / TextureCube
} rlShaderTexType;

//----------------------------------------------------------------------------------
// Descriptor structures
//----------------------------------------------------------------------------------

// Vertex attribute input descriptor
typedef struct rlShaderInput {
    const char *name;           // Field name used in body code as input.<name>
    int type;                   // rlShaderDataType (RL_STYPE_VEC3, etc.)
    int location;               // Vertex attribute location (0=position, 1=texcoord, 2=normal, 3=color, ...)
} rlShaderInput;

// Varying descriptor (VS output / FS input)
typedef struct rlShaderVarying {
    const char *name;           // Field name used as output.<name> in VS, input.<name> in FS
    int type;                   // rlShaderDataType
} rlShaderVarying;

// Uniform (constant) descriptor
typedef struct rlShaderUniform {
    const char *name;           // Uniform name used directly in body code
    int type;                   // rlShaderDataType
    int binding;                // Register/binding index (HLSL: cbuffer register, D3D9: constant register)
} rlShaderUniform;

// Texture descriptor
typedef struct rlShaderTexture {
    const char *name;           // Texture variable name used in sampleTexture(name, uv) or sampleCubemap(name, dir)
    int slot;                   // Texture slot index (register t/s for HLSL, sampler unit for GLSL)
    int type;                   // rlShaderTexType (RL_STEX_2D or RL_STEX_CUBE), default 0 = 2D
} rlShaderTexture;

// Complete unified shader descriptor
typedef struct rlUnifiedShaderDesc {
    // Vertex attribute inputs
    rlShaderInput inputs[RL_MAX_SHADER_INPUTS];
    int inputCount;

    // VS->FS varyings
    rlShaderVarying varyings[RL_MAX_SHADER_VARYINGS];
    int varyingCount;

    // Vertex shader uniforms (constants)
    rlShaderUniform vsUniforms[RL_MAX_SHADER_UNIFORMS];
    int vsUniformCount;

    // Fragment shader uniforms (constants)
    rlShaderUniform fsUniforms[RL_MAX_SHADER_UNIFORMS];
    int fsUniformCount;

    // Textures (used in fragment shader)
    rlShaderTexture textures[RL_MAX_SHADER_TEXTURES];
    int textureCount;

    // Portable preamble code (inserted before main, after declarations)
    // Type names (vec4, mat4, etc.) are auto-transpiled. Use for struct definitions,
    // #defines, helper functions, and complex uniform declarations (e.g. struct arrays).
    const char *vsPreamble;         // Extra code before vertex shader main()
    const char *fsPreamble;         // Extra code before fragment shader main()

    // Portable shader body code
    const char *vertexBody;         // Code inside vertex shader main()
    const char *fragmentBody;       // Code inside fragment shader main()
} rlUnifiedShaderDesc;

//----------------------------------------------------------------------------------
// Functions
//----------------------------------------------------------------------------------
#if defined(__cplusplus)
extern "C" {
#endif

// Load a unified shader: generates native source for the active backend, compiles, and returns shader ID.
// Returns 0 on failure, or the compiled shader ID on success.
RLAPI unsigned int rlLoadUnifiedShader(const rlUnifiedShaderDesc *desc);

// Generate native vertex shader source string for the active backend.
// Caller must free the returned string with rlFreeShaderSource().
RLAPI char *rlGenerateVertexShaderSource(const rlUnifiedShaderDesc *desc);

// Generate native fragment shader source string for the active backend.
// Caller must free the returned string with rlFreeShaderSource().
RLAPI char *rlGenerateFragmentShaderSource(const rlUnifiedShaderDesc *desc);

// Free a shader source string returned by rlGenerateVertexShaderSource/rlGenerateFragmentShaderSource.
RLAPI void rlFreeShaderSource(char *source);

// Get a pre-filled descriptor for the standard raylib default shader.
// Useful as a reference or starting point for custom shaders.
RLAPI rlUnifiedShaderDesc rlGetDefaultUnifiedShaderDesc(void);

#if defined(__cplusplus)
}
#endif

#endif // RL_UNIFIED_SHADER_H
