#version 460

#extension GL_GOOGLE_include_directive : enable
#define BDA 1
#include "./gbuffer_common.h.glsl"
#include "../common.h.glsl"
#include "../math.h.glsl"
#include "../material.h.glsl"

layout(location = 0) in vec3 in_normal;
layout(location = 1) in vec2 in_uv;
layout(location = 2) in vec3 in_frag_pos;
layout(location = 3) in vec3 in_bitangent;
layout(location = 4) in vec3 in_tangent;
layout(location = 5) flat in uint material_id;

layout(location = 0) out vec4 gbuffer_a;
layout(location = 1) out vec4 gbuffer_b;
layout(location = 2) out vec4 gbuffer_c;

VK2_DECLARE_SAMPLED_IMAGES(texture2D);

layout(std430, buffer_reference) readonly buffer Materials {
    Material mats[];
};

void main() {
    SceneData scene_data = SceneDatas(scene_buffer).data;
    uvec4 debug_flags = scene_data.debug_flags;
    Material material = Materials(materials_buffer).mats[nonuniformEXT(material_id)];
    vec3 N;
    if ((debug_flags.x & NORMAL_MAPS_ENABLED_BIT) != 0 && material.ids.y != 0) {
        vec3 tex_map_norm = texture(vk2_sampler2D(material.ids.y, sampler_idx), in_uv).rgb * 2.0 - 1.0;
        mat3 in_tbn = mat3(normalize(in_tangent), normalize(in_bitangent), normalize(in_normal));
        N = normalize(in_tbn * tex_map_norm);
    } else {
        N = normalize(in_normal);
    }
    if (!gl_FrontFacing) {
        N = -N;
    }

    vec4 albedo = material.albedo_factors;
    if (material.ids.x != 0) {
        albedo *= texture(vk2_sampler2D(material.ids.x, sampler_idx), in_uv);
    }

    // https://bgolus.medium.com/anti-aliased-alpha-test-the-esoteric-alpha-to-coverage-8b177335ae4f
    //runAlphaTest(baseColor.a, mat.emissiveFactorAlphaCutoff.w / max(32.0 * fwidth(uv.x), 1.0));
    #ifdef ALPHA_MASK_ENABLED
    if (albedo.a < .5) {
        discard;
    }
    #endif

    vec3 emissive = material.emissive_factors.rgb;
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
    }

    float metallic = material.pbr_factors.x;
    float roughness = material.pbr_factors.y;
    if (material.ids.z != 0) {
        vec3 metal_rough = texture(vk2_sampler2D(material.ids.z, sampler_idx), in_uv).rgb;
        metallic = metal_rough.b;
        roughness = metal_rough.g;
    }

    gbuffer_a = vec4(encode_oct(N) * 0.5 + 0.5, metallic, roughness);
    gbuffer_b = vec4(albedo.rgb, 1.);
    gbuffer_c = vec4(emissive, ao);
}
