#version 460

#include "../math.h.glsl"
#include "../common.h.glsl"
#include "../pbr/pbr.h.glsl"
#include "./basic_common.h.glsl"
#include "../shadows/shadows.h.glsl"

layout(location = 0) in vec3 in_normal;
layout(location = 1) in vec2 in_uv;
layout(location = 2) in vec3 in_frag_pos;
layout(location = 3) flat in uint material_id;

layout(location = 0) out vec4 out_frag_color;

struct Material {
    vec4 emissive_factors;
    uvec4 ids; // albedo, normal, metal_rough, emissive
    uvec4 ids2; // ao, w is flags
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
    if (color.a < .5) {
        discard;
    }
    vec3 emissive = texture(vk2_sampler2D(material.ids.w, sampler_idx), in_uv).rgb *
            material.emissive_factors.w * material.emissive_factors.rgb;
    float ao = 1.0;
    if ((debug_flags.x & AO_ENABLED_BIT) != 0) {
        if ((material.ids2.w & METALLIC_ROUGHNESS_TEX_MASK) == PACKED_OCCLUSION_ROUGHNESS_METALLIC) {
            ao = texture(vk2_sampler2D(material.ids.z, sampler_idx), in_uv).r;
        } else {
            ao = texture(vk2_sampler2D(material.ids2.x, sampler_idx), in_uv).r;
        }
        if ((debug_flags.w & DEBUG_MODE_MASK) == DEBUG_MODE_AO_MAP) {
            out_frag_color = vec4(vec3(ao), 1.);
            return;
        }
    }
    vec3 metal_rough = texture(vk2_sampler2D(material.ids.z, sampler_idx), in_uv).rgb;
    vec3 V = normalize(scene_data.view_pos - in_frag_pos);
    vec3 N = normalize((in_normal));
    if ((debug_flags.w & DEBUG_MODE_MASK) == DEBUG_MODE_NORMALS) {
        out_frag_color = vec4(N, 1.);
        return;
    }

    vec3 halfv = normalize(V + scene_data.light_dir);
    float ndoth = max(dot(N, halfv), 0.0);
    float ndotl = max(dot(N, scene_data.light_dir), 0.0);
    float gloss = metal_rough.b;
    float specular = pow(ndoth, mix(1, 64, gloss)) * gloss;
    float ambient = 0.07;
    // float shadowAmbient = 0.05;
    float sunIntensity = 2.5;
    vec3 outputColor = color.rgb * (ndotl * 2.5 + ambient) * ao + vec3(specular) * color.rgb * sunIntensity + emissive;
    float shadow = calc_shadow(shadow_datas[shadow_buffer_idx].data, scene_data, shadow_img_idx, shadow_sampler_idx, N, in_frag_pos);
    out_frag_color = vec4(tonemap(outputColor), 1.);
    out_frag_color = vec4(vec3(shadow), 1.);
    return;

    vec3 light_out = color_pbr(N, scene_data.light_dir, V, vec4(color.rgb, 1.), metal_rough.b, metal_rough.g, scene_data.light_color);
    outputColor = (light_out + color.rgb * .4) * ao + emissive;

    out_frag_color = vec4(ACESFilm(outputColor), 1.);
    // out_frag_color = vec4(tonemap(outputColor), 1.);
    // out_frag_color = vec4(normal.rgb, 1.);
}
