add_library(renderer

AnimationManager.cpp
StateTracker.cpp
Camera.cpp
VkRender2.cpp
ThreadPool.cpp
util/IndexAllocator.cpp
util/CVar.cpp
util/FileWatcher.cpp
RenderGraph.cpp
techniques/IBL.cpp
techniques/CSM.cpp

Scene.cpp
ResourceManager.cpp
CommandEncoder.cpp
vk2/VkTypes.cpp
vk2/StagingBufferPool.cpp
vk2/Definitions.cpp
vk2/Device.cpp
vk2/Initializers.cpp
vk2/PipelineManager.cpp
vk2/ShaderCompiler.cpp
vk2/Texture.cpp
SceneLoader.cpp
vk2/Swapchain.cpp
vk2/VkCommon.cpp
)

target_link_libraries(renderer PUBLIC
ktx
glfw
glm
lib_imgui
lib_imguizmo
fastgltf
Tracy::TracyClient
volk::volk
vk-bootstrap::vk-bootstrap
VulkanMemoryAllocator
glslang
glslang-default-resource-limits
BS_thread_pool
stb_image
mikktspace
nfd
SPIRV
)

target_compile_options(renderer
    PRIVATE
    $<$<OR:$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:GNU>,$<CXX_COMPILER_ID:Clang>>:
    -Wall
    -Wextra
    -pedantic
    -Wno-missing-field-initializers
    -Wno-unused-result
    >
    $<$<CXX_COMPILER_ID:MSVC>:
    /W4
    /WX
    /permissive-
    >
)

if (${VKRENDER2_VALIDATION_LAYERS_ENABLED_DEFAULT})
    target_compile_definitions(renderer PRIVATE VKRENDER2_VALIDATION_LAYERS_ENABLED_DEFAULT)
endif()

target_include_directories(renderer SYSTEM PUBLIC
    ${CMAKE_SOURCE_DIR}/third_party/vma/include
)

target_include_directories(renderer PUBLIC
${CMAKE_SOURCE_DIR}/third_party
${CMAKE_SOURCE_DIR}/resources
${CMAKE_SOURCE_DIR}/src
)
target_compile_options(renderer PRIVATE "-Wno-unknown-pragmas")
