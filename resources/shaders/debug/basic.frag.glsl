#version 460

#extension GL_GOOGLE_include_directive : enable
#include "../math.h.glsl"
#include "../common.h.glsl"
#include "../pbr/pbr.h.glsl"
#include "./basic_common.h.glsl"
#include "../material.h.glsl"
#include "../shadows/shadows.h.glsl"

layout(location = 0) in vec3 in_normal;
layout(location = 1) in vec2 in_uv;
layout(location = 2) in vec3 in_frag_pos;
layout(location = 3) in vec3 in_bitangent;
layout(location = 4) in vec3 in_tangent;
layout(location = 5) flat in uint material_id;

layout(location = 0) out vec4 out_frag_color;

VK2_DECLARE_SAMPLED_IMAGES(texture2D);
VK2_DECLARE_SAMPLED_IMAGES(textureCube);

VK2_DECLARE_STORAGE_BUFFERS_RO(MaterialBuffers){
Material mats[];
} materials[];

struct Light {
    vec3 L;
    vec3 radiance;
};

// TODO: make a light buffer?
void get_dir_light(out Light l) {
    // TODO: move elsewhere this is cringe!
    SceneData data = scene_data_buffer[scene_buffer].data;
    l.L = -data.light_dir;
    l.radiance = data.light_color;
}

void main() {
    SceneData scene_data = scene_data_buffer[scene_buffer].data;
    uvec4 debug_flags = scene_data.debug_flags;
    Material material = materials[materials_buffer].mats[nonuniformEXT(material_id)];
    vec4 color = material.albedo_factors;
    if (material.ids.x != 0) {
        color *= texture(vk2_sampler2D(material.ids.x, sampler_idx), in_uv);
    }
    if (color.a < .5) {
        discard;
    }
    vec3 emissive = material.emissive_factors.w * material.emissive_factors.rgb;
    if (material.ids.w != 0) {
        emissive *= texture(vk2_sampler2D(material.ids.w, sampler_idx), in_uv).rgb;
    }
    float ao = 1.0;
    if ((debug_flags.x & AO_ENABLED_BIT) != 0) {
        if (material.ids.z != 0 && (material.ids2.w & METALLIC_ROUGHNESS_TEX_MASK) == PACKED_OCCLUSION_ROUGHNESS_METALLIC) {
            ao = texture(vk2_sampler2D(material.ids.z, sampler_idx), in_uv).r;
        } else if (material.ids2.x != 0) {
            ao = texture(vk2_sampler2D(material.ids2.x, sampler_idx), in_uv).r;
        }
        if ((debug_flags.w & DEBUG_MODE_MASK) == DEBUG_MODE_AO_MAP) {
            out_frag_color = vec4(vec3(ao), 1.);
            return;
        }
    }
    float metallic = material.pbr_factors.x;
    float roughness = material.pbr_factors.y;
    if (material.ids.z != 0) {
        vec3 metal_rough = texture(vk2_sampler2D(material.ids.z, sampler_idx), in_uv).rgb;
        metallic = metal_rough.b;
        roughness = metal_rough.g;
    }

    vec3 N;
    if ((debug_flags.x & NORMAL_MAPS_ENABLED_BIT) != 0 && material.ids.y != 0) {
        vec3 tex_map_norm = texture(vk2_sampler2D(material.ids.y, sampler_idx), in_uv).rgb * 2.0 - 1.0;
        mat3 in_tbn = mat3(normalize(in_tangent), normalize(in_bitangent), normalize(in_normal));
        N = normalize(in_tbn * tex_map_norm);
    } else {
        N = normalize(in_normal);
    }
    // TODO: sep pass?
    if (!gl_FrontFacing) {
        N = -N;
    }
    if ((debug_flags.w & DEBUG_MODE_MASK) == DEBUG_MODE_NORMALS) {
        out_frag_color = vec4(N, 1.);
        return;
    }

    vec3 V = normalize(scene_data.view_pos - in_frag_pos);

    if ((debug_flags.w & DEBUG_MODE_MASK) == DEBUG_MODE_CASCADE_LEVELS) {
        out_frag_color = vec4(cascade_debug_color(shadow_datas[shadow_buffer_idx].data, scene_data, in_frag_pos), 1.);
        return;
    }

    float shadow = calc_shadow(shadow_datas[shadow_buffer_idx].data, scene_data, shadow_img_idx, shadow_sampler_idx, N, in_frag_pos);
    if ((debug_flags.w & DEBUG_MODE_MASK) == DEBUG_MODE_SHADOW) {
        out_frag_color = vec4(vec3(shadow), 1.);
        return;
    }

    vec3 light_out = vec3(0.0);

    vec3 albedo = color.rgb;
    vec3 F0 = vec3(0.04);
    F0 = mix(F0, albedo, metallic);
    float NdotV = max(dot(N, V), 0.0);
    // dir light
    {
        Light l;
        get_dir_light(l);
        vec3 L = l.L;
        vec3 radiance = l.radiance;
        vec3 H = normalize(V + L);

        float NDF = DistributionGGX(N, H, roughness);
        float G = GeometrySmith(N, V, L, roughness);
        vec3 F = FresnelSchlick(clamp(dot(H, V), 0.0, 1.0), F0);

        float NdotL = max(dot(N, L), 0.0);
        vec3 numerator = NDF * G * F;
        float denominator = 4.0 * max(NdotV * NdotL, .001);
        vec3 specular = numerator / denominator;
        vec3 kS = F;
        vec3 kD = (vec3(1.0) - kS) * (1.0 - metallic);
        light_out += (kD * albedo + specular) * radiance * NdotL * shadow;
    }

    // IBL ambient
    vec3 ambient = vec3(0.07) * albedo;
    {
        vec3 F = FresnelSchlickRoughness(NdotV, F0, roughness);

        vec3 kS = F;
        vec3 kD = (1.0 - kS) * (1.0 - metallic);
        vec3 irradiance = texture(vk2_samplerCube(irradiance_img_idx, sampler_idx), N).rgb;
        vec3 diffuse = irradiance * albedo;

        const float MaxReflectionLod = 4.;
        vec3 R = reflect(-V, N);
        vec3 prefiltered_color = textureLod(vk2_samplerCube(prefiltered_env_map_idx,
                    linear_clamp_to_edge_sampler_idx), R, roughness * MaxReflectionLod).rgb;
        vec2 env_brdf = texture(vk2_sampler2D(brdf_lut_idx, linear_clamp_to_edge_sampler_idx), vec2(NdotV, roughness)).rg;
        vec3 specular = prefiltered_color * (F * env_brdf.x + env_brdf.y);

        ambient = (kD * diffuse + specular) * ao * scene_data.ambient_intensity;
        vec3 outputColor = light_out + emissive + ambient;
        // outputColor = emissive + ambient;
        out_frag_color = vec4(outputColor, 1.);
    }
}
