/**********************************************************************************************
*
*   rl_backend - Multi-backend render abstraction layer for rlgl
*
*   This header defines the common interface and utilities shared across all rendering backends.
*   Backends: OpenGL (existing), Direct3D 12, Direct3D 11, Direct3D 10, Direct3D 9, Vulkan, Metal, Software
*
*   CONFIGURATION:
*       #define GRAPHICS_API_OPENGL_11 / _21 / _33 / _43 / _ES2 / _ES3  (existing OpenGL)
*       #define GRAPHICS_API_DIRECT3D12       Direct3D 12 rendering backend (preferred over D3D11)
*       #define GRAPHICS_API_DIRECT3D11       Direct3D 11 rendering backend
*       #define GRAPHICS_API_DIRECT3D10       Direct3D 10 rendering backend
*       #define GRAPHICS_API_DIRECT3D9        Direct3D 9 rendering backend
*       #define GRAPHICS_API_VULKAN           Vulkan rendering backend
*       #define GRAPHICS_API_METAL            Metal rendering backend (macOS/iOS)
*       #define GRAPHICS_API_SOFTWARE         Pure CPU software rendering backend (no GPU)
*
*   LICENSE: zlib/libpng (same as raylib)
*
**********************************************************************************************/

#ifndef RL_BACKEND_H
#define RL_BACKEND_H

//----------------------------------------------------------------------------------
// Backend selection validation
//----------------------------------------------------------------------------------

// Count how many non-OpenGL backends are selected
#if defined(GRAPHICS_API_DIRECT3D12)
    #define RL_BACKEND_D3D12  1
#else
    #define RL_BACKEND_D3D12  0
#endif

#if defined(GRAPHICS_API_DIRECT3D11)
    #define RL_BACKEND_D3D11  1
#else
    #define RL_BACKEND_D3D11  0
#endif

#if defined(GRAPHICS_API_DIRECT3D10)
    #define RL_BACKEND_D3D10  1
#else
    #define RL_BACKEND_D3D10  0
#endif

#if defined(GRAPHICS_API_DIRECT3D9)
    #define RL_BACKEND_D3D9   1
#else
    #define RL_BACKEND_D3D9   0
#endif

#if defined(GRAPHICS_API_VULKAN)
    #define RL_BACKEND_VULKAN 1
#else
    #define RL_BACKEND_VULKAN 0
#endif

#if defined(GRAPHICS_API_METAL)
    #define RL_BACKEND_METAL  1
#else
    #define RL_BACKEND_METAL  0
#endif

#if defined(GRAPHICS_API_SOFTWARE)
    #define RL_BACKEND_SOFTWARE 1
#else
    #define RL_BACKEND_SOFTWARE 0
#endif

// Check for OpenGL backend (any version)
#if defined(GRAPHICS_API_OPENGL_11) || defined(GRAPHICS_API_OPENGL_21) || \
    defined(GRAPHICS_API_OPENGL_33) || defined(GRAPHICS_API_OPENGL_43) || \
    defined(GRAPHICS_API_OPENGL_ES2) || defined(GRAPHICS_API_OPENGL_ES3)
    #define RL_BACKEND_OPENGL 1
#else
    #define RL_BACKEND_OPENGL 0
#endif

#define RL_BACKEND_COUNT (RL_BACKEND_OPENGL + RL_BACKEND_D3D12 + RL_BACKEND_D3D11 + RL_BACKEND_D3D10 + RL_BACKEND_D3D9 + RL_BACKEND_VULKAN + RL_BACKEND_METAL + RL_BACKEND_SOFTWARE)

// If no backend is explicitly selected, default to OpenGL 3.3
#if RL_BACKEND_COUNT == 0
    #define GRAPHICS_API_OPENGL_33
    #undef RL_BACKEND_OPENGL
    #define RL_BACKEND_OPENGL 1
#endif

// NOTE: Multiple backends CAN be compiled in simultaneously.
// The active backend is selected at runtime via SetRenderBackend() or auto-detected
// by GetPreferredRenderBackend() during InitWindow().

// Platform validation for non-OpenGL backends
#if defined(GRAPHICS_API_DIRECT3D12) && !defined(_WIN32)
    #error "Direct3D 12 backend is only supported on Windows."
#endif

#if defined(GRAPHICS_API_DIRECT3D11) && !defined(_WIN32)
    #error "Direct3D 11 backend is only supported on Windows."
#endif

#if defined(GRAPHICS_API_DIRECT3D10) && !defined(_WIN32)
    #error "Direct3D 10 backend is only supported on Windows."
#endif

#if defined(GRAPHICS_API_DIRECT3D9) && !defined(_WIN32)
    #error "Direct3D 9 backend is only supported on Windows."
#endif

#if defined(GRAPHICS_API_METAL) && !defined(__APPLE__)
    #error "Metal backend is only supported on macOS/iOS."
#endif

//----------------------------------------------------------------------------------
// Backend version constants
// These values match the rlGlVersion enum in rlgl.h.
// Use the enum values directly (RL_SOFTWARE, RL_DIRECT3D_9, etc.) from rlgl.h
// rather than re-defining them as macros here, to avoid preprocessor conflicts.
//----------------------------------------------------------------------------------

//----------------------------------------------------------------------------------
// Common backend capabilities structure
//----------------------------------------------------------------------------------
typedef struct rlBackendInfo {
    const char *name;               // Backend name string (e.g. "OpenGL 3.3", "Direct3D 11", "Vulkan", "Metal")
    int apiVersion;                 // API version identifier
    bool supportsComputeShaders;    // Compute shader support
    bool supportsSSBO;              // Shader Storage Buffer Object support
    bool supportsInstancing;        // Instanced rendering support
    bool supportsVAO;               // Vertex Array Object support (or equivalent)
    bool supportsTexNPOT;           // Non-power-of-two texture support
    bool supportsTexFloat32;        // 32-bit float texture support
    bool supportsTexFloat16;        // 16-bit float texture support
    bool supportsTexDepth;          // Depth texture support
    bool supportsTexCompDXT;        // DXT compression support
    bool supportsTexCompETC;        // ETC compression support
    bool supportsTexCompASTC;       // ASTC compression support
    bool supportsAnisotropicFilter; // Anisotropic filtering support
    float maxAnisotropyLevel;       // Maximum anisotropy level
    int maxDepthBits;               // Maximum depth buffer bits
    int maxTextureSize;             // Maximum texture dimension
} rlBackendInfo;

//----------------------------------------------------------------------------------
// Performance-optimized alignment macros
//----------------------------------------------------------------------------------
#if defined(_MSC_VER)
    #define RL_ALIGN(n) __declspec(align(n))
#elif defined(__GNUC__) || defined(__clang__)
    #define RL_ALIGN(n) __attribute__((aligned(n)))
#else
    #define RL_ALIGN(n)
#endif

// Cache-line alignment for frequently accessed data
#define RL_CACHE_LINE_SIZE 64
#define RL_CACHE_ALIGN RL_ALIGN(RL_CACHE_LINE_SIZE)

//----------------------------------------------------------------------------------
// Optimized memory pool for small GPU allocations
//----------------------------------------------------------------------------------
#ifndef RL_GPU_POOL_SIZE
    #define RL_GPU_POOL_SIZE 4096   // Default pool size for temporary GPU allocations
#endif

//----------------------------------------------------------------------------------
// Shader cross-compilation defines
// Each backend uses its own shader language; these helpers enable
// backend-agnostic default shader loading
//----------------------------------------------------------------------------------
#if defined(GRAPHICS_API_DIRECT3D12)
    #define RL_SHADER_LANG_HLSL     1
    #define RL_DEFAULT_SHADER_EXT   ".hlsl"
#elif defined(GRAPHICS_API_DIRECT3D11)
    #define RL_SHADER_LANG_HLSL     1
    #define RL_DEFAULT_SHADER_EXT   ".hlsl"
#elif defined(GRAPHICS_API_DIRECT3D10)
    #define RL_SHADER_LANG_HLSL     1
    #define RL_DEFAULT_SHADER_EXT   ".hlsl"
#elif defined(GRAPHICS_API_DIRECT3D9)
    #define RL_SHADER_LANG_HLSL     1
    #define RL_DEFAULT_SHADER_EXT   ".hlsl"
#elif defined(GRAPHICS_API_VULKAN)
    #define RL_SHADER_LANG_SPIRV    1
    #define RL_DEFAULT_SHADER_EXT   ".spv"
#elif defined(GRAPHICS_API_METAL)
    #define RL_SHADER_LANG_MSL      1
    #define RL_DEFAULT_SHADER_EXT   ".metal"
#elif defined(GRAPHICS_API_SOFTWARE)
    #define RL_SHADER_LANG_NONE     1
    #define RL_DEFAULT_SHADER_EXT   ""      // No GPU shaders (fixed-function)
#else
    #define RL_SHADER_LANG_GLSL     1
    #define RL_DEFAULT_SHADER_EXT   ".glsl"
#endif

#endif // RL_BACKEND_H
