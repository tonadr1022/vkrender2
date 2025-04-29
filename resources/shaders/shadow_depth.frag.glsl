#version 460

layout(location = 0) in vec2 in_uv;
layout(location = 1) flat in uint material_id;

#extension GL_GOOGLE_include_directive : enable

#include "./common.h.glsl"
#include "./shadow_depth_common.h.glsl"
#include "./material.h.glsl"

VK2_DECLARE_SAMPLED_IMAGES(texture2D);

VK2_DECLARE_STORAGE_BUFFERS_RO(MaterialBuffers){
Material mats[];
} materials[];

void main() {
    SceneData scene_data = scene_data_buffer[scene_buffer].data;
    uvec4 debug_flags = scene_data.debug_flags;
    Material material = materials[materials_buffer].mats[nonuniformEXT(material_id)];
    vec4 color = texture(vk2_sampler2D(material.ids.x, sampler_idx), in_uv);
    if ((material.ids2.w & MATERIAL_ALPHA_MODE_MASK_BIT) != 0 && color.a < .5) {
        discard;
    }
}
