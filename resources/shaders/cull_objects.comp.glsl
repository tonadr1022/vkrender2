#version 460

#extension GL_GOOGLE_include_directive : enable

#include "resources.h.glsl"
#define BDA 1
#include "./common.h.glsl"
#include "./cull_objects_common.h.glsl"

layout(local_size_x = 256) in;

// #define MODEL_SPACE_AABB 1

// bool distance_cull = bool(bitfieldExtract(cull_data.enable_bits, DISTANCE_CULL_BIT, 1));
// if (distance_cull) {
//     visible = visible && origin.z + radius > cull_data.z_near && origin.z - radius < cull_data.z_far;
// }

bool is_visible_frustum(vec3 pos, float radius) {
    if ((flags & FRUSTUM_CULL_ENABLED_BIT) == 0) {
        return true;
    }
    return dot(left.xyz, pos) + left.w > -radius
        && dot(right.xyz, pos) + right.w > -radius
        && dot(bottom.xyz, pos) + bottom.w > -radius
        && dot(top.xyz, pos) + top.w > -radius
        && dot(near.xyz, pos) + near.w > -radius
        && dot(far.xyz, pos) + far.w > -radius;
    // visible = visible && origin.z * frustum[1] - abs(origin.x) * frustum[0] > -radius;
    // visible = visible && origin.z * frustum[3] - abs(origin.y) * frustum[2] > -radius;
}

bool get_visibility(in vec4 clip, in vec3 min, in vec3 max) {
    // get the dimensions
    float x0 = min.x * clip.x;
    float x1 = max.x * clip.x;
    float y0 = min.y * clip.y;
    float y1 = max.y * clip.y;
    float z0 = min.z * clip.z + clip.w;
    float z1 = max.z * clip.z + clip.w;
    // Get the 8 points of the aabb in clip space
    float p1 = x0 + y0 + z0;
    float p2 = x1 + y0 + z0;
    float p3 = x1 + y1 + z0;
    float p4 = x0 + y1 + z0;
    float p5 = x0 + y0 + z1;
    float p6 = x1 + y0 + z1;
    float p7 = x1 + y1 + z1;
    float p8 = x0 + y1 + z1;

    if (p1 <= 0 && p2 <= 0 && p3 <= 0 && p4 <= 0 && p5 <= 0 && p6 <= 0 && p7 <= 0 && p8 <= 0) {
        return false;
    }
    return true;
    // // If all the points are in the plane, it's fully visible
    // if (p1 > 0 && p2 > 0 && p3 > 0 && p4 > 0 && p5 > 0 && p6 > 0 && p7 > 0 && p8 > 0) {
    //     return FullyVisible;
    // }
    // // partial vis
    // return true;
}

#ifdef MODEL_SPACE_AABB

// doesn't handle non-uniform scaling
void transform_aabb(in mat4 model, in vec3 aabb_min, in vec3 aabb_max, out vec3 result_min, out vec3 result_max) {
    result_min = vec3(model[3]);
    result_max = result_min;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            float a = model[i][j] * aabb_min[j];
            float b = model[i][j] * aabb_max[j];
            result_min[i] += min(a, b);
            result_max[i] += max(a, b);
        }
    }
}

bool is_visible_frustum_aabb_transform(in mat4 mat, vec3 min, vec3 max) {
    vec3 old_min = min;
    vec3 old_max = max;
    transform_aabb(mat, old_min, old_max, min, max);
    return get_visibility(left, min, max)
        && get_visibility(right, min, max)
        && get_visibility(bottom, min, max)
        && get_visibility(top, min, max)
        && get_visibility(near, min, max)
        && get_visibility(far, min, max);
}
#endif

bool is_visible_frustum_aabb(vec3 min, vec3 max) {
    return get_visibility(left, min, max)
        && get_visibility(right, min, max)
        && get_visibility(bottom, min, max)
        && get_visibility(top, min, max)
        && get_visibility(near, min, max)
        && get_visibility(far, min, max);
}

struct ObjectBounds {
    mat4 model;
    vec4 aabb_min;
    vec4 aabb_max;
};

VK2_DECLARE_STORAGE_BUFFERS_RO(ObjectBoundsBuffer){
ObjectBounds bounds[];
} object_bounds[];

struct DrawInfo {
    uint index_cnt;
    uint first_index;
    uint vertex_offset;
    uint instance_id;
    uint flags; // 0x1 == double sided
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
    if ((flags & FRUSTUM_CULL_ENABLED_BIT) == 0) {
        return true;
    }
    #ifdef MODEL_SPACE_AABB
    return is_visible_frustum_aabb_transform(bounds.model, bounds.aabb_min.xyz, bounds.aabb_max.xyz);
    #else
    return is_visible_frustum_aabb(bounds.aabb_min.xyz, bounds.aabb_max.xyz);
    #endif
    // return is_visible_frustum(bounds.sphere_bounds.xyz, bounds.sphere_bounds.w);
}

void main() {
    uint id = gl_GlobalInvocationID.x;
    if (id >= num_objs) {
        return;
    }
    DrawInfo draw_info = draw_cmds[in_draw_info_buf_idx].cmds[id];
    if (draw_info.index_cnt == 0) {
        return;
    }
    // get the object. test its frustum against the view frustum
    if (is_visible(object_bounds[object_bounds_buf_idx].bounds[draw_info.instance_id])) {
        uint out_idx = atomicAdd(out_cmds[out_draw_cmds_buf_idx].cnt, 1);
        DrawCmd cmd;
        cmd.first_instance = draw_info.instance_id;
        cmd.index_cnt = draw_info.index_cnt;
        cmd.first_index = draw_info.first_index;
        cmd.vertex_offset = int(draw_info.vertex_offset);
        cmd.instance_cnt = 1;
        out_cmds[out_draw_cmds_buf_idx].cmds[out_idx] = cmd;
    }
}
