#ifndef VK2_COMMON_H
#define VK2_COMMON_H

#include "./resources.h.glsl"

struct SceneData {
    mat4 view_proj;
    uvec4 debug_flags;
    vec3 view_pos;
};

VK2_DECLARE_STORAGE_BUFFERS_RO(SceneUniforms){
SceneData data;
} scene_data_buffer[];

#endif
