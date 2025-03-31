#ifndef VK2_COMMON_H
#define VK2_COMMON_H

#include "./resources.h.glsl"

#define AO_ENABLED_BIT 0x1
#define METALLIC_ROUGHNESS_TEX_MASK 3
#define PACKED_OCCLUSION_ROUGHNESS_METALLIC 1
#define DEBUG_MODE_MASK 127

#define DEBUG_MODE_NONE 0
#define DEBUG_MODE_AO_MAP 1
#define DEBUG_MODE_NORMALS 2
#define DEBUG_MODE_CASCADE_LEVELS 3
#define DEBUG_MODE_COUNT 4

#ifndef __cplusplus
struct SceneData {
    mat4 view_proj;
    mat4 view;
    uvec4 debug_flags; // w is debug modes
    vec3 view_pos;
    vec3 light_dir;
    vec3 light_color;
};

VK2_DECLARE_STORAGE_BUFFERS_RO(SceneUniforms){
SceneData data;
} scene_data_buffer[];

#endif

#endif
