cmake_minimum_required(VERSION 3.24)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(vkrender2)
add_subdirectory(third_party)

add_executable(vkrender2
src/main.cpp
src/App.cpp
src/vk2/Device.cpp
src/vk2/Definitions.cpp
src/vk2/DeletionQueue.cpp
src/vk2/Initializers.cpp
src/vk2/Swapchain.cpp
src/vk2/VkCommon.cpp
src/VkRender2.cpp

)

target_compile_options(vkrender2
    PRIVATE
    $<$<OR:$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:GNU>,$<CXX_COMPILER_ID:Clang>>:
    -Wall
    -Wextra
    -pedantic-errors
    -Wno-missing-field-initializers
    -Wno-unused-result
    >
    $<$<CXX_COMPILER_ID:MSVC>:
    /W4
    /WX
    /permissive-
    >
)

option(VKRENDER2_FORCE_COLORED_OUTPUT "Always produce ANSI-colored output (GNU/Clang only)." TRUE)
if (${FORCE_COLORED_OUTPUT})
    if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
        add_compile_options(-fdiagnostics-color=always)
    elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
        add_compile_options(-fcolor-diagnostics)
    endif()
endif()

target_include_directories(vkrender2 PRIVATE
third_party
src
)

target_link_libraries(vkrender2 PRIVATE
glfw
glm
lib_imgui
fastgltf
Tracy::TracyClient
volk::volk
vk-bootstrap::vk-bootstrap
VulkanMemoryAllocator
glslang
)

target_compile_definitions(glm INTERFACE GLM_FORCE_DEPTH_ZERO_TO_ONE VK_NO_PROTOTYPES GLFW_INCLUDE_NONE)
