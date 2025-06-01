#version 460

#extension GL_GOOGLE_include_directive : enable
#include "../resources.h.glsl"
#include "../shadows/shadows.h.glsl"
#include "./shade_common.h.glsl"
#include "../math.h.glsl"
#include "../pbr/pbr.h.glsl"

layout(local_size_x = 16, local_size_y = 16) in;

VK2_DECLARE_STORAGE_IMAGES(image2D);
VK2_DECLARE_SAMPLED_IMAGES(texture2D);
VK2_DECLARE_SAMPLED_IMAGES(textureCube);

#define STORE(x) imageStore(vk2_get_storage_img(image2D, output_tex), tex_coord, x)

void main() {
    ivec2 tex_coord = ivec2(gl_GlobalInvocationID);
    ivec2 img_size = imageSize(vk2_get_storage_img(image2D, gbuffer_a_tex));
    if (tex_coord.x >= img_size.x || tex_coord.y >= img_size.y) {
        return;
    }
    SceneData scene_data = scene_data_buffer[scene_buffer].data;
    uvec4 debug_flags = scene_data.debug_flags;

    vec2 uv = (vec2(tex_coord) + .5) / vec2(img_size);
    float depth = texture(vk2_sampler2D(depth_img, sampler_idx), uv).r;
    vec4 clip_pos = vec4(uv * 2. - 1., depth, 1.);
    vec4 wpos_pre_divide = inv_view_proj * clip_pos;
    vec3 world_pos = wpos_pre_divide.xyz / wpos_pre_divide.w;

    vec4 gbuffer_a = imageLoad(vk2_get_storage_img(image2D, gbuffer_a_tex), tex_coord);
    vec4 gbuffer_b = imageLoad(vk2_get_storage_img(image2D, gbuffer_b_tex), tex_coord);
    vec4 gbuffer_c = imageLoad(vk2_get_storage_img(image2D, gbuffer_c_tex), tex_coord);
    vec3 N = decode_oct(gbuffer_a.rg * 2. - 1.);
    float metallic = gbuffer_a.b;
    float roughness = gbuffer_a.a;
    vec3 albedo = gbuffer_b.rgb;
    vec3 emissive = gbuffer_c.rgb;
    // vec3 emissive = albedo * (exp2(gbuffer_b.a * 5) - 1);
    float ao = gbuffer_c.a;
    vec3 V = normalize(scene_data.view_pos - world_pos);

    if ((debug_flags.w & DEBUG_MODE_MASK) == DEBUG_MODE_NORMALS) {
        STORE(vec4(N * .5 + .5, 1.));
        return;
    }

    if ((debug_flags.w & DEBUG_MODE_MASK) == DEBUG_MODE_CASCADE_LEVELS) {
        STORE(vec4(cascade_debug_color(shadow_datas[shadow_buffer_idx].data, scene_data, world_pos), 1.));
        return;
    }

    float shadow = 1.0;
    if ((debug_flags.x & CSM_ENABLED_BIT) != 0) {
        shadow = calc_shadow(shadow_datas[shadow_buffer_idx].data, scene_data, shadow_img_idx, shadow_sampler_idx, N, world_pos);
    }
    if ((debug_flags.w & DEBUG_MODE_MASK) == DEBUG_MODE_SHADOW) {
        STORE(vec4(vec3(shadow), 1.));
        return;
    }

    // imageStore(vk2_get_storage_img(image2D, output_tex), tex_coord, vec4(world_pos, 1.));
    imageStore(vk2_get_storage_img(image2D, output_tex), tex_coord, vec4(N * .5 + .5, 1.));

    vec3 light_out = vec3(0.0);
    vec3 F0 = vec3(0.04);
    F0 = mix(F0, albedo, metallic);
    float NdotV = max(dot(N, V), 0.0);
    // dir light
    {
        vec3 L = -scene_data.light_dir;
        vec3 radiance = scene_data.light_color;
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
    {
        vec3 ambient;
        vec3 F = FresnelSchlickRoughness(NdotV, F0, roughness);
        vec3 kS = F;
        vec3 kD = (1.0 - kS) * (1.0 - metallic);
        vec3 irradiance = texture(vk2_samplerCube(irradiance_img_idx, sampler_idx), N).rgb;
        vec3 diffuse = irradiance * albedo;
        if ((debug_flags.x & IBL_ENABLED_BIT) != 0) {
            const float MaxReflectionLod = 4.;
            vec3 R = reflect(-V, N);
            vec3 prefiltered_color = textureLod(vk2_samplerCube(prefiltered_env_map_idx,
                        linear_clamp_to_edge_sampler_idx), R, roughness * MaxReflectionLod).rgb;
            vec2 env_brdf = texture(vk2_sampler2D(brdf_lut_idx, linear_clamp_to_edge_sampler_idx), vec2(NdotV, roughness)).rg;
            vec3 specular = prefiltered_color * (F * env_brdf.x + env_brdf.y);
            ambient = (kD * diffuse + specular) * ao * scene_data.ambient_intensity;
        } else {
            ambient = kD * diffuse * ao * scene_data.ambient_intensity;
            // ambient = albedo* ao * scene_data.ambient_intensity;
        }
        vec3 outputColor = light_out + emissive + ambient;
        STORE(vec4(outputColor, 1.));
    }
}
