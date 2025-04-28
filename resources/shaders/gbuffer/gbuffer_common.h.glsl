#ifndef GBUFFER_COMMON_H
#define GBUFFER_COMMON_H

#include "../resources.h.glsl"

VK2_DECLARE_ARGUMENTS(GBufferPushConstants){
u64 vtx;
u64 scene_buffer;
u64 instance_buffer;
u64 object_data_buffer;
u64 materials_buffer;
uint sampler_idx;
} ;

#endif
