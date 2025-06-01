
#ifndef TRANSPARENT_COMMON_H
#define TRANSPARENT_COMMON_H

#include "../resources.h.glsl"

VK2_DECLARE_ARGUMENTS(TransparentPushConstants){
u64 instance_buffer;
u64 vertex_buffer;
u64 object_data_buffer;
u64 materials_buffer;
uint scene_buffer;
uint irradiance_img_idx;
uint brdf_lut_idx;
uint prefiltered_env_map_idx;
uint linear_clamp_to_edge_sampler_idx;
uint atomic_counter_buffer;
uint max_oit_fragments;
uint oit_tex_heads;
uint oit_lists_buf;
uint sampler_idx;
ivec2 img_size;
uint depth_img;
} pc;

#endif
