# Adding compile definitions
target_compile_definitions("raylib" PUBLIC "${PLATFORM_CPP}")

# Primary/active backend — this is the ONLY GRAPHICS_API_* define set.
# Only one backend can be compiled because all backends implement the same
# public rlgl functions; defining multiple would compile all of them and
# cause duplicate-symbol linker errors.
target_compile_definitions("raylib" PUBLIC "${GRAPHICS}")

# Expose RL_HAS_* availability flags so application code can query which
# backend libraries were linked (even though only one backend is compiled).
# GRAPHICS_ALL_BACKENDS is populated by LibraryConfigurations.cmake.
if (GRAPHICS_ALL_BACKENDS)
    foreach(_RL_BACKEND_DEF ${GRAPHICS_ALL_BACKENDS})
        # Convert GRAPHICS_API_DIRECT3D12 → RL_HAS_DIRECT3D12, etc.
        string(REPLACE "GRAPHICS_API_" "RL_HAS_" _RL_HAS_FLAG "${_RL_BACKEND_DEF}")
        target_compile_definitions("raylib" PUBLIC "${_RL_HAS_FLAG}")
    endforeach()
    message(STATUS "Active backend: ${GRAPHICS}")
    message(STATUS "Available backend libraries: ${GRAPHICS_ALL_BACKENDS}")
endif()

function(define_if target variable)
    if(${${variable}})
        message(STATUS "${variable}=${${variable}}")
        target_compile_definitions(${target} PRIVATE "${variable}")
    endif()
endfunction()

if(${CUSTOMIZE_BUILD})
    target_compile_definitions("raylib" PRIVATE EXTERNAL_CONFIG_FLAGS)

    foreach(FLAG IN LISTS CONFIG_HEADER_FLAGS)
        string(REGEX MATCH "([^=]+)=(.+)" _ ${FLAG})
        define_if("raylib" ${CMAKE_MATCH_1})
    endforeach()

    foreach(VALUE IN LISTS CONFIG_HEADER_VALUES)
        target_compile_definitions("raylib" PRIVATE ${VALUE})
    endforeach()
endif()
