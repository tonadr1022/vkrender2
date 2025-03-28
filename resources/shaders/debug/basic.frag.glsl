#version 460

#include "../math.h.glsl"
#include "../common.h.glsl"
#include "../pbr/pbr.h.glsl"
#include "./basic_common.h.glsl"

layout(location = 0) in vec3 in_normal;
layout(location = 1) in vec2 in_uv;
layout(location = 2) in vec3 in_frag_pos;
layout(location = 3) flat in uint material_id;

layout(location = 0) out vec4 out_frag_color;

struct Material {
    vec4 emissive_factors;
    uvec4 ids; // albedo, normal, metal_rough, emissive
    uvec4 ids2; // ao
};

VK2_DECLARE_SAMPLED_IMAGES(texture2D);

VK2_DECLARE_STORAGE_BUFFERS_RO(MaterialBuffers){
Material mats[];
} materials[];

void main() {
    SceneData scene_data = scene_data_buffer[scene_buffer].data;
    uvec4 debug_flags = scene_data.debug_flags;
    Material material = materials[materials_buffer].mats[nonuniformEXT(material_id)];
    vec4 color = texture(vk2_sampler2D(material.ids.x, sampler_idx), in_uv);
    vec3 emissive = texture(vk2_sampler2D(material.ids.w, sampler_idx), in_uv).rgb *
            material.emissive_factors.w * material.emissive_factors.rgb;
    float ao = 1.0;
    // TODO: packed ao metal rough
    if ((debug_flags.x & AO_ENABLED_BIT) != 0) {
        ao = texture(vk2_sampler2D(material.ids2.x, sampler_idx), in_uv).r;
    }
    vec3 metal_rough = texture(vk2_sampler2D(material.ids.z, sampler_idx), in_uv).rgb;
    vec3 V = normalize(scene_data.view_pos - in_frag_pos);
    vec3 N = normalize(in_normal);
    vec3 L = normalize(vec3(0.5, 0.5, 0.0));
    // blue is metalness
    vec3 light_out = color_pbr(N, L, V, vec4(color.rgb, 1.), metal_rough.b, metal_rough.g, vec3(1.));

    out_frag_color = vec4((light_out + color.rgb * .2) * ao + emissive, 1.);
    // if ((debug_flags.x & AO_ENABLED_BIT) == 0) {
    //     out_frag_color = vec4(metal_rough, 1.);
    // }
    // out_frag_color = vec4(normal.rgb, 1.);
}
