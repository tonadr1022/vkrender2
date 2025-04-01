#version 460

#extension GL_GOOGLE_include_directive : enable

#include "resources.h.glsl"

layout(local_size_x = 256) in;

layout(push_constant) uniform PC {
    uint num_objs;
    uint in_draw_cmds_buf_idx;
    uint out_draw_cmds_buf_idx;
    uint draw_cnt_buf_idx;
    uint object_bounds_buf_idx;
} pc;

struct ObjectBounds {
    mat4 model;
    vec4 sphere_bounds;
    vec4 extents;
};

VK2_DECLARE_STORAGE_BUFFERS_RO(ObjectBoundsBuffer){
ObjectBounds bounds[];
} object_bounds[];

struct DrawCmd {
    uint index_cnt;
    uint instance_cnt;
    uint first_index;
    int vertex_offset;
    uint first_instance;
};

VK2_DECLARE_STORAGE_BUFFERS_RO(DrawCmdsBuffer){
DrawCmd cmds[];
} draw_cmds[];

VK2_DECLARE_STORAGE_BUFFERS_WO(OutDrawCmdsBuffer){
DrawCmd cmds[];
} out_cmds[];

VK2_DECLARE_STORAGE_BUFFERS_WO(DrawCntBuffer){
uint cnt;
} draw_cnts[];

bool is_visible(in ObjectBounds bounds) {
    return true;
}

void main() {
    uint id = gl_GlobalInvocationID.x;
    if (id >= pc.num_objs) {
        return;
    }
    DrawCmd cmd = draw_cmds[pc.in_draw_cmds_buf_idx].cmds[id];
    ObjectBounds bounds = object_bounds[pc.object_bounds_buf_idx].bounds[id];
    // get the object. test its frustum against the view frustum
    if (is_visible(bounds)) {
        uint out_idx = atomicAdd(draw_cnts[pc.draw_cnt_buf_idx].cnt, 1);
        cmd.instance_cnt = 1;
        out_cmds[pc.out_draw_cmds_buf_idx].cmds[out_idx] = cmd;
    }
}
