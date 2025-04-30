#ifndef GBUFFER_SHADE_COMMON_H
#define GBUFFER_SHADE_COMMON_H

#include "../resources.h.glsl"

VK2_DECLARE_ARGUMENTS(DeferredShadePushConstants){
mat4 inv_view_proj;
uint gbuffer_a_tex;
uint gbuffer_b_tex;
uint gbuffer_c_tex;
uint depth_img;
uint output_tex;
uint sampler_idx;
uint scene_buffer;
uint shadow_img_idx;
uint shadow_sampler_idx;
uint shadow_buffer_idx;
uint irradiance_img_idx;
uint brdf_lut_idx;
uint prefiltered_env_map_idx;
uint linear_clamp_to_edge_sampler_idx;
} ;

#endif
