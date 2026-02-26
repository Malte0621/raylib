# Set OpenGL_GL_PREFERENCE to new "GLVND" even when legacy library exists and
# cmake is <= 3.10
#
# See https://cmake.org/cmake/help/latest/policy/CMP0072.html for more
# information.
if(POLICY CMP0072)
  cmake_policy(SET CMP0072 NEW)
endif()

if (${PLATFORM} MATCHES "Desktop")
    set(PLATFORM_CPP "PLATFORM_DESKTOP")

    if (APPLE)
        # Need to force OpenGL 3.3 on OS X
        # See: https://github.com/raysan5/raylib/issues/341
        set(GRAPHICS "GRAPHICS_API_OPENGL_33")
        find_library(OPENGL_LIBRARY OpenGL)
        set(LIBS_PRIVATE ${OPENGL_LIBRARY})
        link_libraries("${LIBS_PRIVATE}")
        if (NOT CMAKE_SYSTEM STRLESS "Darwin-18.0.0")
            add_definitions(-DGL_SILENCE_DEPRECATION)
            MESSAGE(AUTHOR_WARNING "OpenGL is deprecated starting with macOS 10.14 (Mojave)!")
        endif ()
    elseif (WIN32)
        add_definitions(-D_CRT_SECURE_NO_WARNINGS)
        find_package(OpenGL QUIET)
        set(LIBS_PRIVATE ${OPENGL_LIBRARIES} winmm)
    elseif (UNIX)
        find_library(pthread NAMES pthread)
        find_package(OpenGL QUIET)
        if ("${OPENGL_LIBRARIES}" STREQUAL "")
            set(OPENGL_LIBRARIES "GL")
        endif ()

        if ("${CMAKE_SYSTEM_NAME}" MATCHES "(Net|Open)BSD")
            find_library(OSS_LIBRARY ossaudio)
        endif ()

        set(LIBS_PRIVATE m pthread ${OPENGL_LIBRARIES} ${OSS_LIBRARY})
    else ()
        find_library(pthread NAMES pthread)
        find_package(OpenGL QUIET)
        if ("${OPENGL_LIBRARIES}" STREQUAL "")
            set(OPENGL_LIBRARIES "GL")
        endif ()

        set(LIBS_PRIVATE m atomic pthread ${OPENGL_LIBRARIES} ${OSS_LIBRARY})

        if ("${CMAKE_SYSTEM_NAME}" MATCHES "(Net|Open)BSD")
            find_library(OSS_LIBRARY ossaudio)
            set(LIBS_PRIVATE m pthread ${OPENGL_LIBRARIES} ${OSS_LIBRARY})
        endif ()

        if (NOT "${CMAKE_SYSTEM_NAME}" MATCHES "(Net|Open)BSD" AND USE_AUDIO)
            set(LIBS_PRIVATE ${LIBS_PRIVATE} dl)
        endif ()
    endif ()

elseif (${PLATFORM} MATCHES "Web")
    set(PLATFORM_CPP "PLATFORM_WEB")
    if(NOT GRAPHICS)
        set(GRAPHICS "GRAPHICS_API_OPENGL_ES2")
    endif()
    set(CMAKE_STATIC_LIBRARY_SUFFIX ".a")

elseif (${PLATFORM} MATCHES "Android")
    set(PLATFORM_CPP "PLATFORM_ANDROID")
    set(GRAPHICS "GRAPHICS_API_OPENGL_ES2")
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
    list(APPEND raylib_sources ${ANDROID_NDK}/sources/android/native_app_glue/android_native_app_glue.c)
    include_directories(${ANDROID_NDK}/sources/android/native_app_glue)
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--exclude-libs,libatomic.a -Wl,--build-id -Wl,-z,noexecstack -Wl,-z,relro -Wl,-z,now -Wl,--warn-shared-textrel -Wl,--fatal-warnings -u ANativeActivity_onCreate -Wl,-undefined,dynamic_lookup")

    find_library(OPENGL_LIBRARY OpenGL)
    set(LIBS_PRIVATE m log android EGL GLESv2 OpenSLES atomic c)

elseif ("${PLATFORM}" MATCHES "DRM")
    set(PLATFORM_CPP "PLATFORM_DRM")
    set(GRAPHICS "GRAPHICS_API_OPENGL_ES2")

    add_definitions(-D_DEFAULT_SOURCE)
    add_definitions(-DEGL_NO_X11)
    add_definitions(-DPLATFORM_DRM)

    find_library(GLESV2 GLESv2)
    find_library(EGL EGL)
    find_library(DRM drm)
    find_library(GBM gbm)

    if (NOT CMAKE_CROSSCOMPILING OR NOT CMAKE_SYSROOT)
        include_directories(/usr/include/libdrm)
    endif ()
    set(LIBS_PRIVATE ${GLESV2} ${EGL} ${DRM} ${GBM} atomic pthread m dl)

elseif ("${PLATFORM}" MATCHES "SDL")
    find_package(SDL2 REQUIRED)
    set(PLATFORM_CPP "PLATFORM_DESKTOP_SDL")
    set(LIBS_PRIVATE SDL2::SDL2)

endif ()

if (NOT ${OPENGL_VERSION} MATCHES "OFF")
    set(SUGGESTED_GRAPHICS "${GRAPHICS}")

    if (${OPENGL_VERSION} MATCHES "4.3")
        set(GRAPHICS "GRAPHICS_API_OPENGL_43")
    elseif (${OPENGL_VERSION} MATCHES "3.3")
        set(GRAPHICS "GRAPHICS_API_OPENGL_33")
    elseif (${OPENGL_VERSION} MATCHES "2.1")
        set(GRAPHICS "GRAPHICS_API_OPENGL_21")
    elseif (${OPENGL_VERSION} MATCHES "1.1")
        set(GRAPHICS "GRAPHICS_API_OPENGL_11")
    elseif (${OPENGL_VERSION} MATCHES "ES 2.0")
        set(GRAPHICS "GRAPHICS_API_OPENGL_ES2")
    elseif (${OPENGL_VERSION} MATCHES "ES 3.0")
        set(GRAPHICS "GRAPHICS_API_OPENGL_ES3")
    endif ()
    if (NOT "${SUGGESTED_GRAPHICS}" STREQUAL "" AND NOT "${SUGGESTED_GRAPHICS}" STREQUAL "${GRAPHICS}")
        message(WARNING "You are overriding the suggested GRAPHICS=${SUGGESTED_GRAPHICS} with ${GRAPHICS}! This may fail.")
    endif ()
endif ()

if (NOT GRAPHICS)
    set(GRAPHICS "GRAPHICS_API_OPENGL_33")
endif ()

# ---------------------------------------------------------------------------
# Graphics backend selection (Auto / explicit / OFF)
# Priority for Auto mode:
#   Windows : DirectX 12 (preferred, full mesh rendering implemented)
#             Also links D3D11, D3D10, D3D9 so all backends are available
#   macOS   : Metal        >  OpenGL 3.3
#   Linux   : Vulkan (if SDK found)  >  OpenGL 3.3
#   Other   : keep current GRAPHICS (OpenGL / ES)
#
# In Auto mode ALL detected backends are compiled in and their libraries
# linked.  GRAPHICS is set to the primary/preferred backend.  The full
# list is stored in GRAPHICS_ALL_BACKENDS for CompileDefinitions.cmake.
# ---------------------------------------------------------------------------

# Centralised Vulkan SDK check — needs both the library AND the headers.
# The NVIDIA driver ships vulkan-1.dll but not vulkan.h; we must skip the
# backend in that case.
set(RL_VULKAN_AVAILABLE FALSE)
find_package(Vulkan QUIET)
if (Vulkan_FOUND)
    if (Vulkan_INCLUDE_DIRS AND EXISTS "${Vulkan_INCLUDE_DIRS}/vulkan/vulkan.h")
        set(RL_VULKAN_AVAILABLE TRUE)
        message(STATUS "Vulkan SDK found (headers: ${Vulkan_INCLUDE_DIRS})")
    else()
        message(STATUS "Vulkan runtime found but SDK headers missing — Vulkan backend disabled")
    endif()
endif()

set(GRAPHICS_ALL_BACKENDS "")   # Will hold *all* backend defines to compile

if (${GRAPHICS_BACKEND} MATCHES "Auto")
    set(_RL_AUTO_BACKEND_CHOSEN FALSE)

    if (WIN32)
        # Detect D3D12 availability
        include(CheckCSourceCompiles)
        set(CMAKE_REQUIRED_LIBRARIES d3d12 dxgi)
        check_c_source_compiles("
            #include <initguid.h>
            #include <d3d12.h>
            int main(void) { D3D12CreateDevice(0, D3D_FEATURE_LEVEL_11_0, &IID_ID3D12Device, 0); return 0; }
        " RL_HAS_D3D12)
        unset(CMAKE_REQUIRED_LIBRARIES)

        # --- Primary backend (determines GRAPHICS) ---
        if (RL_HAS_D3D12)
            set(GRAPHICS "GRAPHICS_API_DIRECT3D12")
            set(_RL_AUTO_BACKEND_CHOSEN TRUE)
            message(STATUS "[Auto] Primary backend: DirectX 12")
        else()
            set(GRAPHICS "GRAPHICS_API_DIRECT3D11")
            set(_RL_AUTO_BACKEND_CHOSEN TRUE)
            message(STATUS "[Auto] Primary backend: DirectX 11 (D3D12 not available)")
        endif()

        # --- Link and enable ALL available Windows backends ---
        # D3D12
        if (RL_HAS_D3D12)
            list(APPEND GRAPHICS_ALL_BACKENDS "GRAPHICS_API_DIRECT3D12")
            set(LIBS_PRIVATE ${LIBS_PRIVATE} d3d12)
        endif()
        # D3D11 (always available on modern Windows)
        list(APPEND GRAPHICS_ALL_BACKENDS "GRAPHICS_API_DIRECT3D11")
        set(LIBS_PRIVATE ${LIBS_PRIVATE} d3d11)
        # D3D10
        list(APPEND GRAPHICS_ALL_BACKENDS "GRAPHICS_API_DIRECT3D10")
        set(LIBS_PRIVATE ${LIBS_PRIVATE} d3d10)
        # D3D9
        list(APPEND GRAPHICS_ALL_BACKENDS "GRAPHICS_API_DIRECT3D9")
        set(LIBS_PRIVATE ${LIBS_PRIVATE} d3d9)
        # Common DirectX libraries (d3dcompiler, dxgi, dxguid)
        set(LIBS_PRIVATE ${LIBS_PRIVATE} d3dcompiler dxgi dxguid)
        # Vulkan (if SDK with headers found)
        if (RL_VULKAN_AVAILABLE)
            list(APPEND GRAPHICS_ALL_BACKENDS "GRAPHICS_API_VULKAN")
            set(LIBS_PRIVATE ${LIBS_PRIVATE} ${Vulkan_LIBRARIES})
        endif()
        # Software (always available, no extra libs)
        list(APPEND GRAPHICS_ALL_BACKENDS "GRAPHICS_API_SOFTWARE")

        message(STATUS "[Auto] All enabled backends: ${GRAPHICS_ALL_BACKENDS}")
    elseif (APPLE)
        find_library(METAL_LIBRARY Metal)
        if (METAL_LIBRARY)
            find_library(METALKIT_LIBRARY MetalKit)
            find_library(QUARTZCORE_LIBRARY QuartzCore)
            set(GRAPHICS "GRAPHICS_API_METAL")
            list(APPEND GRAPHICS_ALL_BACKENDS "GRAPHICS_API_METAL")
            set(LIBS_PRIVATE ${LIBS_PRIVATE} ${METAL_LIBRARY} ${METALKIT_LIBRARY} ${QUARTZCORE_LIBRARY})
            set(_RL_AUTO_BACKEND_CHOSEN TRUE)
            message(STATUS "[Auto] Selected Metal (native Apple API)")
        endif()
        # Software always available
        list(APPEND GRAPHICS_ALL_BACKENDS "GRAPHICS_API_SOFTWARE")
    elseif (UNIX AND NOT ANDROID AND NOT "${PLATFORM}" MATCHES "DRM|Web")
        if (RL_VULKAN_AVAILABLE)
            set(GRAPHICS "GRAPHICS_API_VULKAN")
            list(APPEND GRAPHICS_ALL_BACKENDS "GRAPHICS_API_VULKAN")
            set(LIBS_PRIVATE ${LIBS_PRIVATE} ${Vulkan_LIBRARIES})
            set(_RL_AUTO_BACKEND_CHOSEN TRUE)
            message(STATUS "[Auto] Selected Vulkan (SDK found)")
        endif()
        # Software always available
        list(APPEND GRAPHICS_ALL_BACKENDS "GRAPHICS_API_SOFTWARE")
    endif()

    if (NOT _RL_AUTO_BACKEND_CHOSEN)
        # Check if OpenGL is available, otherwise fall back to software rendering
        find_package(OpenGL QUIET)
        if (OpenGL_FOUND)
            message(STATUS "[Auto] Keeping default OpenGL backend (${GRAPHICS})")
        else()
            set(GRAPHICS "GRAPHICS_API_SOFTWARE")
            set(_RL_AUTO_BACKEND_CHOSEN TRUE)
            message(STATUS "[Auto] No GPU rendering API available, falling back to Software (CPU) renderer")
        endif()
    endif()

elseif (NOT ${GRAPHICS_BACKEND} MATCHES "OFF")
    # Explicit backend selection — sets the *primary* backend but still
    # detects and links ALL available backends so they can be used at runtime.
    if (${GRAPHICS_BACKEND} MATCHES "DirectX12")
        if (NOT WIN32)
            message(FATAL_ERROR "DirectX12 backend is only supported on Windows")
        endif()
        set(GRAPHICS "GRAPHICS_API_DIRECT3D12")
    elseif (${GRAPHICS_BACKEND} MATCHES "DirectX11")
        if (NOT WIN32)
            message(FATAL_ERROR "DirectX11 backend is only supported on Windows")
        endif()
        set(GRAPHICS "GRAPHICS_API_DIRECT3D11")
    elseif (${GRAPHICS_BACKEND} MATCHES "DirectX10")
        if (NOT WIN32)
            message(FATAL_ERROR "DirectX10 backend is only supported on Windows")
        endif()
        set(GRAPHICS "GRAPHICS_API_DIRECT3D10")
    elseif (${GRAPHICS_BACKEND} MATCHES "DirectX9")
        if (NOT WIN32)
            message(FATAL_ERROR "DirectX9 backend is only supported on Windows")
        endif()
        set(GRAPHICS "GRAPHICS_API_DIRECT3D9")
    elseif (${GRAPHICS_BACKEND} MATCHES "Vulkan")
        set(GRAPHICS "GRAPHICS_API_VULKAN")
    elseif (${GRAPHICS_BACKEND} MATCHES "Metal")
        if (NOT APPLE)
            message(FATAL_ERROR "Metal backend is only supported on Apple platforms")
        endif()
        set(GRAPHICS "GRAPHICS_API_METAL")
    elseif (${GRAPHICS_BACKEND} MATCHES "Software")
        set(GRAPHICS "GRAPHICS_API_SOFTWARE")
    endif()
    message(STATUS "Graphics Backend Override: ${GRAPHICS_BACKEND} -> ${GRAPHICS}")

    # Even in explicit mode, detect and link ALL available backends
    if (WIN32)
        include(CheckCSourceCompiles)
        set(CMAKE_REQUIRED_LIBRARIES d3d12 dxgi)
        check_c_source_compiles("
            #include <initguid.h>
            #include <d3d12.h>
            int main(void) { D3D12CreateDevice(0, D3D_FEATURE_LEVEL_11_0, &IID_ID3D12Device, 0); return 0; }
        " RL_HAS_D3D12_EXPLICIT)
        unset(CMAKE_REQUIRED_LIBRARIES)

        if (RL_HAS_D3D12_EXPLICIT)
            list(APPEND GRAPHICS_ALL_BACKENDS "GRAPHICS_API_DIRECT3D12")
            set(LIBS_PRIVATE ${LIBS_PRIVATE} d3d12)
        endif()
        list(APPEND GRAPHICS_ALL_BACKENDS "GRAPHICS_API_DIRECT3D11")
        set(LIBS_PRIVATE ${LIBS_PRIVATE} d3d11)
        list(APPEND GRAPHICS_ALL_BACKENDS "GRAPHICS_API_DIRECT3D10")
        set(LIBS_PRIVATE ${LIBS_PRIVATE} d3d10)
        list(APPEND GRAPHICS_ALL_BACKENDS "GRAPHICS_API_DIRECT3D9")
        set(LIBS_PRIVATE ${LIBS_PRIVATE} d3d9)
        set(LIBS_PRIVATE ${LIBS_PRIVATE} d3dcompiler dxgi dxguid)

        if (RL_VULKAN_AVAILABLE)
            list(APPEND GRAPHICS_ALL_BACKENDS "GRAPHICS_API_VULKAN")
            set(LIBS_PRIVATE ${LIBS_PRIVATE} ${Vulkan_LIBRARIES})
        endif()
        list(APPEND GRAPHICS_ALL_BACKENDS "GRAPHICS_API_SOFTWARE")
    elseif (APPLE)
        find_library(METAL_LIBRARY Metal)
        if (METAL_LIBRARY)
            find_library(METALKIT_LIBRARY MetalKit)
            find_library(QUARTZCORE_LIBRARY QuartzCore)
            list(APPEND GRAPHICS_ALL_BACKENDS "GRAPHICS_API_METAL")
            set(LIBS_PRIVATE ${LIBS_PRIVATE} ${METAL_LIBRARY} ${METALKIT_LIBRARY} ${QUARTZCORE_LIBRARY})
        endif()
        list(APPEND GRAPHICS_ALL_BACKENDS "GRAPHICS_API_SOFTWARE")
    elseif (UNIX AND NOT ANDROID)
        if (RL_VULKAN_AVAILABLE)
            list(APPEND GRAPHICS_ALL_BACKENDS "GRAPHICS_API_VULKAN")
            set(LIBS_PRIVATE ${LIBS_PRIVATE} ${Vulkan_LIBRARIES})
        endif()
        list(APPEND GRAPHICS_ALL_BACKENDS "GRAPHICS_API_SOFTWARE")
    endif()

    # Ensure the primary backend is in the list
    if (NOT "${GRAPHICS}" IN_LIST GRAPHICS_ALL_BACKENDS)
        list(APPEND GRAPHICS_ALL_BACKENDS "${GRAPHICS}")
    endif()
    message(STATUS "[Explicit] All enabled backends: ${GRAPHICS_ALL_BACKENDS}")
endif()

set(LIBS_PRIVATE ${LIBS_PRIVATE} ${OPENAL_LIBRARY})

if (${PLATFORM} MATCHES "Desktop")
    set(LIBS_PRIVATE ${LIBS_PRIVATE} glfw)
endif ()
