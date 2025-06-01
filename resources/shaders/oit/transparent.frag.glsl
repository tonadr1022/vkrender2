#version 460

#extension GL_GOOGLE_include_directive : enable

#include "../resources.h.glsl"
#define IGNORE_SCENE_UNIFORM_DECL
#include "../common.h.glsl"

#include "./transparent_common.h.glsl"
#include "../pbr/pbr.h.glsl"
#define BDA 1
#include "../material.h.glsl"

VK2_DECLARE_STORAGE_IMAGES_FORMAT(uimage2D, r32ui);
VK2_DECLARE_SAMPLED_IMAGES(texture2D);
VK2_DECLARE_SAMPLED_IMAGES(textureCube);

VK2_DECLARE_STORAGE_BUFFERS(AtomicCounter){
uint cnt;
} cnt_bufs[];

struct OITTransparentFragment {
    f16vec4 color;
    float depth;
    uint next;
};

layout(std430, set = 0, binding = BINDLESS_STORAGE_BUFFER_BINDING) buffer SceneUniforms {
    SceneData data;
} scene_data_buffer[];

layout(scalar, set = 0, binding = BINDLESS_STORAGE_BUFFER_BINDING) buffer OITListsBuf {
    OITTransparentFragment frags[];
} oit_lists_bufs[];

layout(std430, buffer_reference) readonly buffer Materials {
    Material mats[];
};

layout(early_fragment_tests) in;
layout(location = 0) in vec3 in_normal;
layout(location = 1) in vec2 in_uv;
layout(location = 2) in vec3 in_frag_pos;
layout(location = 3) in vec3 in_bitangent;
layout(location = 4) in vec3 in_tangent;
layout(location = 5) flat in uint material_id;

void main() {
    SceneData scene_data = scene_data_buffer[pc.scene_buffer].data;
    Material material = Materials(pc.materials_buffer).mats[nonuniformEXT(material_id)];
    vec3 emissive = material.emissive_factors.rgb;
    if (material.ids.w != 0) {
        emissive *= texture(vk2_sampler2D(material.ids.w, pc.sampler_idx), in_uv).rgb;
    }
    vec4 albedo = material.albedo_factors;
    if (material.ids.x != 0) {
        albedo *= texture(vk2_sampler2D(material.ids.x, pc.sampler_idx), in_uv);
    }

    uvec4 debug_flags = scene_data.debug_flags;
    vec3 N;
    if ((debug_flags.x & NORMAL_MAPS_ENABLED_BIT) != 0 && material.ids.y != 0) {
        vec3 tex_map_norm = texture(vk2_sampler2D(material.ids.y, pc.sampler_idx), in_uv).rgb * 2.0 - 1.0;
        mat3 in_tbn = mat3(normalize(in_tangent), normalize(in_bitangent), normalize(in_normal));
        N = normalize(in_tbn * tex_map_norm);
    } else {
        N = normalize(in_normal);
    }
    if (!gl_FrontFacing) {
        N = -N;
    }

    vec2 uv = (vec2(gl_FragCoord) + .5) / vec2(pc.img_size);
    float depth = texture(vk2_sampler2D(pc.depth_img, pc.sampler_idx), uv).r;
    vec4 clip_pos = vec4(uv * 2. - 1., depth, 1.);
    vec4 wpos_pre_divide = scene_data.inverse_view_proj * clip_pos;
    vec3 world_pos = wpos_pre_divide.xyz / wpos_pre_divide.w;

    float metallic = material.pbr_factors.x;
    float roughness = material.pbr_factors.y;
    if (material.ids.z != 0) {
        vec3 metal_rough = texture(vk2_sampler2D(material.ids.z, pc.sampler_idx), in_uv).rgb;
        metallic = metal_rough.b;
        roughness = metal_rough.g;
    }
    vec3 F0 = vec3(0.04);
    F0 = mix(F0, albedo.rgb, metallic);
    vec3 color = emissive;
    vec3 V = normalize(scene_data.view_pos - world_pos);
    float NdotV = max(dot(N, V), 0.0);
    vec3 F = FresnelSchlickRoughness(NdotV, F0, roughness);
    vec3 kS = F;
    vec3 kD = (1.0 - kS) * (1.0 - metallic);
    vec3 irradiance = texture(vk2_samplerCube(pc.irradiance_img_idx, pc.sampler_idx), N).rgb;
    vec3 diffuse = irradiance * albedo.rgb;
    if ((debug_flags.x & IBL_ENABLED_BIT) != 0) {
        const float MaxReflectionLod = 4.;
        vec3 R = reflect(-V, N);
        vec3 prefiltered_color = textureLod(vk2_samplerCube(pc.prefiltered_env_map_idx,
                    pc.linear_clamp_to_edge_sampler_idx), R, roughness * MaxReflectionLod).rgb;
        vec2 env_brdf = texture(vk2_sampler2D(pc.brdf_lut_idx, pc.linear_clamp_to_edge_sampler_idx), vec2(NdotV, roughness)).rg;
        vec3 specular = prefiltered_color * (F * env_brdf.x + env_brdf.y);
        color += (kD * diffuse + specular) * scene_data.ambient_intensity;
    } else {
        color += kD * diffuse * scene_data.ambient_intensity;
    }

    // TODO: clearcoat transmission thickness
    float alpha = albedo.a;
    bool is_transparent = (alpha > 0.01) && (alpha < 0.99);
    // TODO: remove
    is_transparent = true;
    if (is_transparent && !gl_HelperInvocation) {
        uint idx = atomicAdd(cnt_bufs[pc.atomic_counter_buffer].cnt, 1);
        if (idx < pc.max_oit_fragments) {
            uint prev_idx = imageAtomicExchange(vk2_get_storage_img(uimage2D, pc.oit_tex_heads), ivec2(gl_FragCoord.xy), idx);
            OITTransparentFragment frag;
            frag.color = f16vec4(color, alpha);
            frag.depth = gl_FragCoord.z;
            frag.next = prev_idx;
            oit_lists_bufs[pc.oit_lists_buf].frags[idx] = frag;
        }
    }
}
