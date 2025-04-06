#ifndef EQ_TO_CUBE_COMP_COMMON_H
#define EQ_TO_CUBE_COMP_COMMON_H

VK2_DECLARE_ARGUMENTS(EquirectToCubeComputePushConstants){
uint sampler_idx;
uint tex_idx;
uint out_tex_idx;
} ;

#endif
