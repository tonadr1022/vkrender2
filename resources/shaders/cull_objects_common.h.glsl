#ifndef CULL_OBJECTS_COMMON_H
#define CULL_OBJECTS_COMMON_H
#define FRUSTUM_CULL_ENABLED_BIT (1 << 0)

VK2_DECLARE_ARGUMENTS(CullObjectPushConstants){
vec4 left;
vec4 right;
vec4 bottom;
vec4 top;
vec4 near;
vec4 far;
u64 scene_data;
u32 num_objs;
u32 in_draw_info_buf_idx;
u32 out_draw_cmds_buf_idx;
u32 object_bounds_buf_idx;
u32 flags;
} ;

#endif
