cmake_minimum_required(VERSION 3.24)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(vkrender2)
add_subdirectory(third_party)

add_executable(vkrender2

src/main.cpp
src/BaseRenderer.cpp
src/StateTracker.cpp
src/Camera.cpp
src/VkRender2.cpp
src/ThreadPool.cpp
src/util/IndexAllocator.cpp
src/util/CVar.cpp

src/vk2/StagingBufferPool.cpp
src/vk2/Fence.cpp
src/vk2/BindlessResourceAllocator.cpp
src/vk2/Buffer.cpp
src/vk2/Definitions.cpp
src/vk2/DeletionQueue.cpp
src/vk2/Descriptors.cpp
src/vk2/Device.cpp
src/vk2/Initializers.cpp
src/vk2/PipelineManager.cpp
src/vk2/SamplerCache.cpp
src/vk2/ShaderCompiler.cpp
src/vk2/Texture.cpp
src/SceneLoader.cpp
src/vk2/Swapchain.cpp
src/vk2/VkCommon.cpp

src/App.cpp
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
resources
src
)

target_link_libraries(vkrender2 PRIVATE
ktx
glfw
glm
lib_imgui
fastgltf
Tracy::TracyClient
volk::volk
vk-bootstrap::vk-bootstrap
VulkanMemoryAllocator
glslang
glslang-default-resource-limits
spirv-reflect-static
BS_thread_pool
stb_image
SPIRV
)

# target_compile_definitions(vkrender2 PRIVATE TRACY_VK_USE_SYMBOL_TABLE)
target_compile_definitions(glm INTERFACE GLM_FORCE_DEPTH_ZERO_TO_ONE VK_NO_PROTOTYPES GLFW_INCLUDE_NONE)
