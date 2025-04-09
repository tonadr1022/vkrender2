#ifndef BASIC_COMMON_H
#define BASIC_COMMON_H

#include "../resources.h.glsl"

VK2_DECLARE_ARGUMENTS(BasicPushConstants){
uint scene_buffer;
uint vertex_buffer_idx;
uint instance_buffer;
uint object_data_buffer;
uint materials_buffer;
uint sampler_idx;
uint shadow_buffer_idx;
uint shadow_sampler_idx;
uint shadow_img_idx;
uint irradiance_img_idx;
uint brdf_lut_idx;
uint prefiltered_env_map_idx;
uint linear_clamp_to_edge_sampler_idx;
} ;

#endif // BASIC_COMMON_H
