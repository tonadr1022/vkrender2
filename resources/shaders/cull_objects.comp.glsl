#version 460

#extension GL_GOOGLE_include_directive : enable

#include "resources.h.glsl"
#define BDA 1
#include "./common.h.glsl"
#include "./cull_objects_common.h.glsl"

layout(local_size_x = 256) in;

bool is_visible_frustum(vec3 origin, float radius) {
    if ((flags & FRUSTUM_CULL_ENABLED_BIT) == 0) {
        return true;
    }
    origin = (SceneDatas(scene_data).data.view * vec4(origin, 1.0)).xyz;
    bool visible = true;
    // bool distance_cull = bool(bitfieldExtract(cull_data.enable_bits, DISTANCE_CULL_BIT, 1));
    // if (distance_cull) {
    //     visible = visible && origin.z + radius > cull_data.z_near && origin.z - radius < cull_data.z_far;
    // }
    visible = visible && origin.z * frustum[1] - abs(origin.x) * frustum[0] > -radius;
    visible = visible && origin.z * frustum[3] - abs(origin.y) * frustum[2] > -radius;
    return visible;
}

struct ObjectBounds {
    mat4 model;
    vec4 sphere_bounds;
    vec4 extents;
};

VK2_DECLARE_STORAGE_BUFFERS_RO(ObjectBoundsBuffer){
ObjectBounds bounds[];
} object_bounds[];

struct DrawInfo {
    uvec4 data; // index_cnt, first_index, vertex_offset
    // uint index_cnt;
    // uint first_index;
    // int vertex_offset;
};

struct DrawCmd {
    uint index_cnt;
    uint instance_cnt;
    uint first_index;
    int vertex_offset;
    uint first_instance;
};

VK2_DECLARE_STORAGE_BUFFERS(DrawCmdsBuffer){
DrawInfo cmds[];
} draw_cmds[];

VK2_DECLARE_STORAGE_BUFFERS(OutDrawCmdsBuffer){
uint cnt;
DrawCmd cmds[];
} out_cmds[];

bool is_visible(in ObjectBounds bounds) {
    return is_visible_frustum(bounds.sphere_bounds.xyz, bounds.sphere_bounds.w);
}

void main() {
    uint id = gl_GlobalInvocationID.x;
    if (id >= num_objs) {
        return;
    }
    DrawInfo draw_info = draw_cmds[in_draw_info_buf_idx].cmds[id];
    if (draw_info.data.x == 0) {
        return;
    }
    // get the object. test its frustum against the view frustum
    if (is_visible(object_bounds[object_bounds_buf_idx].bounds[id])) {
        uint out_idx = atomicAdd(out_cmds[out_draw_cmds_buf_idx].cnt, 1);
        DrawCmd cmd;
        cmd.first_instance = id;
        cmd.index_cnt = draw_info.data.x;
        cmd.first_index = draw_info.data.y;
        cmd.vertex_offset = int(draw_info.data.z);
        cmd.instance_cnt = 1;
        out_cmds[out_draw_cmds_buf_idx].cmds[out_idx] = cmd;
    }
}
