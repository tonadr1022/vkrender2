#version 460

#extension GL_GOOGLE_include_directive : enable
#include "../math.h.glsl"
#include "../common.h.glsl"
#include "../pbr/pbr.h.glsl"
#include "./basic_common.h.glsl"
#include "../shadows/shadows.h.glsl"

layout(location = 0) in vec3 in_normal;
layout(location = 1) in vec2 in_uv;
layout(location = 2) in vec3 in_frag_pos;
layout(location = 3) in vec3 in_bitangent;
layout(location = 4) in vec3 in_tangent;
layout(location = 5) flat in uint material_id;

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
    // if (material.emissive_factors.w > 1.) {
    //     out_frag_color = vec4(1.);
    //     return;
    // } else {
    //     out_frag_color = vec4(0.);
    //     return;
    // }
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

    vec3 N;
    if ((debug_flags.x & NORMAL_MAPS_ENABLED_BIT) != 0 && material.ids.y != 0) {
        vec3 tex_map_norm = texture(vk2_sampler2D(material.ids.y, sampler_idx), in_uv).rgb * 2.0 - 1.0;
        mat3 in_tbn = mat3(normalize(in_tangent), normalize(in_bitangent), normalize(in_normal));
        N = normalize(in_tbn * tex_map_norm);
    } else {
        N = normalize(in_normal);
    }
    // vec3 N = normalize((in_normal));
    if ((debug_flags.w & DEBUG_MODE_MASK) == DEBUG_MODE_NORMALS) {
        out_frag_color = vec4(N, 1.);
        return;
    }

    vec3 V = normalize(scene_data.view_pos - in_frag_pos);

    float ambient = 0.07;

    // vec3 gammaCorrected = pow(color.rgb, vec3(1.0 / 2.2));
    // out_frag_color = vec4(gammaCorrected, 1.);
    // return;
    if ((debug_flags.w & DEBUG_MODE_MASK) == DEBUG_MODE_CASCADE_LEVELS) {
        out_frag_color = vec4(cascade_debug_color(shadow_datas[shadow_buffer_idx].data, scene_data, in_frag_pos), 1.);
        return;
    }

    float shadow = calc_shadow(shadow_datas[shadow_buffer_idx].data, scene_data, shadow_img_idx, shadow_sampler_idx, N, in_frag_pos);
    if ((debug_flags.w & DEBUG_MODE_MASK) == DEBUG_MODE_SHADOW) {
        out_frag_color = vec4(vec3(shadow), 1.);
        return;
    }
    // vec3 outputColor = color.rgb * (ndotl * min(shadow + shadowAmbient, 1.) * sunIntensity + ambient) * ao + vec3(specular * shadow) * color.rgb * sunIntensity + emissive;
    // out_frag_color = vec4(ACESFilm(outputColor), 1.);
    // out_frag_color = vec4(tonemap(outputColor), 1.);
    // out_frag_color = vec4(vec3(metal_rough.g), 1.);
    // return;
    vec3 light_out = color_pbr(N, scene_data.light_dir, V, vec4(color.rgb, 1.), metal_rough.b, metal_rough.g, scene_data.light_color) * shadow;
    vec3 amb = ambient * color.rgb * ao * scene_data.ambient_intensity;
    vec3 outputColor = light_out * shadow + emissive + amb;
    vec3 tonemapped = ACESFilm(outputColor);
    vec3 gamma_corrected = gamma_correct(tonemapped);
    out_frag_color = vec4(gamma_corrected, 1.);
    // out_frag_color = vec4(ACESFilm(outputColor), 1.);
    // out_frag_color = vec4(color.rgb * shadow, 1.);

    // outputColor = (light_out + color.rgb * .4) * ao + emissive;
    //
    // out_frag_color = vec4(ACESFilm(outputColor), 1.);
    // out_frag_color = vec4(tonemap(outputColor), 1.);
    // out_frag_color = vec4(normal.rgb, 1.);
}
