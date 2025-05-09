option(GLFW_BUILD_TESTS "" OFF)
option(GLFW_BUILD_DOCS "" OFF)
option(GLFW_INSTALL "" OFF)
option(GLFW_BUILD_EXAMPLES "" OFF)


set(VMA_STATIC_VULKAN_FUNCTIONS OFF CACHE BOOL "" FORCE)
set(VMA_DYNAMIC_VULKAN_FUNCTIONS OFF CACHE BOOL "" FORCE)

option(ENABLE_OPT "" OFF)
set(ENABLE_GLSLANG_BINARIES OFF CACHE BOOL "" FORCE)

option(TRACY_ENABLE "Enable profiling" ON)
option(TRACY_ONLY_IPV4 "IPv4 only" ON)


option(KTX_FEATURE_TESTS "" OFF)
option(KTX_FEATURE_VULKAN "" ON)
option(KTX_FEATURE_GL_UPLOAD "" OFF)
option(KTX_FEATURE_VK_UPLOAD "" ON)
option(KTX_FEATURE_WRITE "" OFF)
option(KTX_FEATURE_TOOLS "" OFF)
option(KTX_FEATURE_STATIC_LIBRARY "" ON)

add_subdirectory(glfw)
add_subdirectory(glm)
add_subdirectory(volk)
add_subdirectory(vk-bootstrap)
add_subdirectory(vma)
add_subdirectory(glslang)
add_subdirectory(ktx)
add_subdirectory(tracy)
add_subdirectory(fastgltf)

add_library(lib_imgui
    imgui/imgui.cpp
    imgui/imgui_demo.cpp
    imgui/imgui_draw.cpp
    imgui/imgui_widgets.cpp
    imgui/imgui_tables.cpp
    imgui/misc/cpp/imgui_stdlib.cpp
    imgui/backends/imgui_impl_glfw.cpp
    imgui/backends/imgui_impl_vulkan.cpp)
target_include_directories(lib_imgui PUBLIC
    imgui
    imgui/misc/cpp
    imgui/backends
    ${Vulkan_INCLUDE_DIRS})

target_compile_definitions(lib_imgui PUBLIC IMGUI_IMPL_VULKAN_NO_PROTOTYPES)
target_link_libraries(lib_imgui PRIVATE glfw volk::volk)

add_library(BS_thread_pool INTERFACE)
target_include_directories(BS_thread_pool INTERFACE bshoshany-thread-pool/include)

add_library(stb_image INTERFACE)
target_include_directories(stb_image SYSTEM INTERFACE stb_image)
add_library(mikktspace STATIC MikkTSpace/mikktspace.c)
target_include_directories(mikktspace PUBLIC MikkTSpace)
