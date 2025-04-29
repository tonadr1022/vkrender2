#ifndef SHADOW_DEPTH_COMMON_H
#define SHADOW_DEPTH_COMMON_H

#include "./resources.h.glsl"

VK2_DECLARE_ARGUMENTS(ShadowDepthPushConstants){
mat4 vp_matrix;
u64 vtx;
u64 instance_buffer;
u64 object_data_buffer;
uint scene_buffer;
uint materials_buffer;
uint sampler_idx;
} ;
#endif
