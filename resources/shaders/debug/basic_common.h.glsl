#ifndef BASIC_COMMON_H
#define BASIC_COMMON_H

#include "../resources.h.glsl"

VK2_DECLARE_ARGUMENTS(BasicPushConstants){
uint scene_buffer;
uint vertex_buffer_idx;
uint instance_buffer;
uint materials_buffer;
uint material_id_buffer;
uint sampler_idx;
} ;

#endif // BASIC_COMMON_H
