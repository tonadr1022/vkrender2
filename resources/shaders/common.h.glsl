#ifndef VK2_COMMON_H
#define VK2_COMMON_H

#include "./resources.h.glsl"

#define AO_ENABLED_BIT (1 << 0)
#define NORMAL_MAPS_ENABLED_BIT (1 << 1)
#define CSM_ENABLED_BIT (1 << 2)
#define IBL_ENABLED_BIT (1 << 3)

#define METALLIC_ROUGHNESS_TEX_MASK 3
#define PACKED_OCCLUSION_ROUGHNESS_METALLIC 1
#define DEBUG_MODE_MASK 127

#define DEBUG_MODE_NONE 0
#define DEBUG_MODE_AO_MAP 1
#define DEBUG_MODE_NORMALS 2
#define DEBUG_MODE_CASCADE_LEVELS 3
#define DEBUG_MODE_SHADOW 4
#define DEBUG_MODE_COUNT 5

#ifndef __cplusplus
struct SceneData {
    mat4 view_proj;
    mat4 view;
    mat4 proj;
    uvec4 debug_flags; // w is debug modes
    vec3 view_pos;
    vec3 light_dir;
    vec3 light_color;
    float ambient_intensity;
};

#ifdef BDA

layout(std430, buffer_reference) readonly buffer SceneDatas {
    SceneData data;
};

#else
VK2_DECLARE_STORAGE_BUFFERS_RO(SceneUniforms){
SceneData data;
} scene_data_buffer[];
#endif

#endif

#endif
