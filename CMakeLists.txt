cmake_minimum_required(VERSION 3.24)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(vkrender2)

if(APPLE)
    enable_language(OBJC)
endif()
add_subdirectory(third_party)

add_subdirectory(src)

add_subdirectory(apps)

option(VKRENDER2_FORCE_COLORED_OUTPUT "Always produce ANSI-colored output (GNU/Clang only)." TRUE)
if (${VKRENDER2_FORCE_COLORED_OUTPUT})
    if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
        add_compile_options(-fdiagnostics-color=always)
    elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
        add_compile_options(-fcolor-diagnostics)
    endif()
endif()

set(VKRENDER2_VALIDATION_LAYERS_ENABLED_DEFAULT ON)
if (CMAKE_BUILD_TYPE STREQUAL "Release")
    set(VKRENDER2_VALIDATION_LAYERS_ENABLED_DEFAULT OFF)
endif()
