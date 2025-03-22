#version 460

#include "../resources.h.glsl"
#include "./basic_common.h.glsl"

layout(location = 0) in vec3 in_normal;
layout(location = 1) in vec2 in_uv;
layout(location = 2) flat in uint material_id;

layout(location = 0) out vec4 out_frag_color;

struct Material {
    uint albedo_id;
    uint normal_id;
};

VK2_DECLARE_SAMPLED_IMAGES(texture2D);

VK2_DECLARE_STORAGE_BUFFERS_RO(MaterialBuffers){
Material mats[];
} materials[];

void main() {
    Material material = materials[materials_buffer].mats[nonuniformEXT(material_id)];
    vec4 color = texture(vk2_sampler2D(material.albedo_id, sampler_idx), in_uv);
    // vec4 normal = texture(vk2_sampler2D(material.normal_id, sampler_idx), in_uv);
    out_frag_color = vec4(color.rgb, 1.);
    // out_frag_color = vec4(normal.rgb, 1.);
}
