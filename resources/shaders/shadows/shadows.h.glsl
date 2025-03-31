#ifndef SHADOWS_H
#define SHADOWS_H

#include "../common.h.glsl"
VK2_DECLARE_SAMPLED_IMAGES(texture2DArray);

struct ShadowData {
    mat4 light_space_matrices[5];
    vec4 biases; // min = x, max = y, pcf scale, z, z_far: w
    vec4 cascade_levels;
    uint setting_bits;
    int cascade_count;
};

#define PCF_BIT 1
VK2_DECLARE_STORAGE_BUFFERS_RO(ShadowUniforms){
ShadowData data;
} shadow_datas[];

float shadow_projection(uint shadow_img_idx, uint shadow_sampler_idx, vec4 shadow_coord, vec2 offset, float bias, int layer) {
    float shadow_factor = 1.0;
    // perspective divide
    shadow_coord = shadow_coord / shadow_coord.w;
    // [-1,1] to [0,1]
    shadow_coord.st = shadow_coord.st * 0.5 + 0.5;
    float curr_depth = shadow_coord.z;
    if (curr_depth > 1.0 || curr_depth < 0.0) return 1.0;
    vec3 sc = vec3(vec2(shadow_coord.st + offset), layer);
    float pcf_depth = texture(vk2_sampler2DArray(shadow_img_idx, shadow_sampler_idx), vec3(shadow_coord.st + offset, layer)).r;
    return (curr_depth - bias) > pcf_depth ? 0.0 : 1.0;
}
float calc_shadow(in ShadowData shadow_ubo, in SceneData data, uint shadow_img_idx, uint shadow_sampler_idx, vec3 normal, vec3 frag_pos) {
    // get the layer of the cascade depth map using view space depth
    vec4 frag_pos_view_space = data.view * vec4(frag_pos, 1.0);
    float depth_val = abs(frag_pos_view_space.z);
    if (depth_val > shadow_ubo.biases.w) {
        return 1.0;
    }
    int layer = shadow_ubo.cascade_count - 1;
    for (int i = 0; i < shadow_ubo.cascade_count - 1; i++) {
        if (depth_val < shadow_ubo.cascade_levels[i]) {
            layer = i;
            break;
        }
    }
    // get the shadow map coordinate from the matrices
    vec4 shadow_coord = shadow_ubo.light_space_matrices[layer] * vec4(frag_pos, 1.0);

    // add bias to prevent shadow acne
    // when normal is closer to 90deg with light dir, increase bias
    const float max_bias = shadow_ubo.biases.y;
    const float min_bias = shadow_ubo.biases.x;
    float bias = max(max_bias * (1.0 - dot(normal, -data.light_dir.xyz)), min_bias);
    const float bias_mod = 0.5;
    // less bias for farther cascade levels
    if (layer == shadow_ubo.cascade_count - 1) {
        // z_far is w
        bias *= 1.0 / (shadow_ubo.biases.w * bias_mod);
    } else {
        bias *= 1.0 / (shadow_ubo.cascade_levels[layer] * bias_mod);
    }
    if ((shadow_ubo.setting_bits & PCF_BIT) != 0) {
        return 0.0;
        // return FilterPCF2(shadow_coord, bias, layer);
    } else {
        return shadow_projection(shadow_img_idx, shadow_sampler_idx, shadow_coord, vec2(0.0, 0.0), bias, layer);
    }
}

#endif
