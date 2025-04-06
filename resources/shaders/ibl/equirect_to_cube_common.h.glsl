#ifndef EQUIRECT_TO_CUBE_H
#define EQUIRECT_TO_CUBE_H

#include "../resources.h.glsl"

VK2_DECLARE_ARGUMENTS(EquirectToCubePushConstants){
mat4 view_proj;
uint sampler_idx;
uint tex_idx;
uint vertex_buffer_idx;
} ;

#endif
