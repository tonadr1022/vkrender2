# VkRender2

This is a Vulkan renderer.

![Bistro](screenshots/bistro.png)
![Sponza](screenshots/sponza.png)

## Features

- Render graph for auto barrier placement (TODO multiple queues)
- Fully Bindless/Indirect rendering (1 descriptor set)
- PBR
- Image Based Lighting
- Cascaded Shadow Maps
- glTF loading with KTX2 compressed texture support
- HDR
- Tonemapping
- (for now) MoltenVK compatible, ie using every Vulkan 1.3 extension possible while requiring 1.2

## Building

```bash
cmake --preset Release
cmake --build build/Release
./bin/Release/vkrender2
```

## Running

- I didn't make an effort to really make this user friendly yet (either going to with this renderer, or make a new one (VkRender3??))
- WASD to move around
- R to ascend, F to descend
- Esc to enter/exit movement mode

## Dependencies

- [GLM](https://github.com/g-truc/glm)
- [GLFW](https://github.com/glfw/glfw)
- [ImGui](https://github.com/ocornut/imgui)
- [Vulkan](https://www.lunarg.com/vulkan-sdk/)
- [Volk](https://github.com/zeux/volk.git)
- [Vulkan Memory Allocator](https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator)
- [Glslang](https://github.com/KhronosGroup/glslang.git)
- [Tracy](https://github.com/wolfpld/tracy.git)
- [KTX](https://github.com/KhronosGroup/KTX-Software.git)
- [BS Thread Pool](https://github.com/bshoshany/thread-pool.git)
- [VkBootstrap](https://github.com/charles-lunarg/vk-bootstrap)
- [stb_image](https://github.com/nothings/stb)
- [fastgltf](https://github.com/spnda/fastgltf)
- [MikkTSpace](https://github.com/mmikk/MikkTSpace)

## TODOs (be warned, this repo may die if I work on a render graph and start over lol)

- Culling
- Render graph
- Meshlets (with or without actual mesh shaders)
- Bloom
- Anti-aliasing
- Global illumination (somehow idk we'll get there)
- Point/spot lights

## Resources/Notes

- <https://developer.nvidia.com/blog/vulkan-dos-donts/>
- <https://gpuopen-librariesandsdks.github.io/VulkanMemoryAllocator/html/usage_patterns.html>
- <https://tellusim.com/mesh-shader-emulation/>
- <https://github.com/KhronosGroup/Vulkan-Docs/wiki/Synchronization-Examples#upload-data-from-the-cpu-to-a-vertex-buffer>
- extra pbr: <http://www.codinglabs.net/article_physically_based_rendering.aspx>
- csm example: <https://github.com/walbourn/directx-sdk-samples/blob/main/CascadedShadowMaps11/RenderCascadeScene.hlsl>

### render graph Notes

- adding buffers to the render graph: buffers that need double buffering?
