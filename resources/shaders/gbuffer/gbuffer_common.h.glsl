#ifndef GBUFFER_COMMON_H
#define GBUFFER_COMMON_H

#include "../resources.h.glsl"

VK2_DECLARE_ARGUMENTS(GBufferPushConstants){
uint scene_buffer;
uint vertex_buffer_idx;
uint instance_buffer;
uint object_data_buffer;
uint materials_buffer;
uint sampler_idx;
} ;

#endif
