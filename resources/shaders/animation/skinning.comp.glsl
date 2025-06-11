#version 460

#define MAX_WEIGHTS 4

#extension GL_GOOGLE_include_directive : enable
#include "../resources.h.glsl"

#define VERTEX_UNDEF
#include "../vertex_common.h.glsl"

struct SkinnedVertexData {
    vec4 pos;
    vec4 normal;
    uint bone_id[MAX_WEIGHTS];
    float weights[MAX_WEIGHTS];
};

layout(local_size_x = 16, local_size_y = 1, local_size_z = 1) in;

layout(std430, buffer_reference) readonly buffer BoneMatrices {
    mat4 matrix[];
};

layout(scalar, buffer_reference) readonly buffer SkinnedVertexBuffer {
    SkinnedVertexData skinned_vertices[];
};

layout(std430, buffer_reference) writeonly buffer VertexBuffer {
    Vertex vertices[];
};

layout(std430, buffer_reference) readonly buffer VertexToTask {
    uint task_index[];
};

struct SkinWorkItem {
    uint skinned_vertex_i;
    uint out_vertex_i;
    uint bone_matrix_start_i;
};

layout(std430, buffer_reference) readonly buffer SkinWorkItemsBuffer {
    SkinWorkItem tasks[];
};

// each primitive has an output vertex offset and input vertex offset
// input: skinned
// output: regular vertex offset

layout(push_constant) uniform PC {
    BoneMatrices bone_matrix_buffer;
    VertexBuffer out_vertices_buf;
    SkinnedVertexBuffer skinned_vertex_buffer;
    VertexToTask vtx_to_task_buf;
    SkinWorkItemsBuffer skin_work_items_buf;
    u64 cnt;
} pc;

void main() {
    uint index = gl_GlobalInvocationID.x;
    if (index >= pc.cnt) {
        return;
    }
    uint task_i = pc.vtx_to_task_buf.task_index[index];
    if (task_i == ~0) {
        return;
    }
    SkinWorkItem skin_task = pc.skin_work_items_buf.tasks[task_i];
    SkinnedVertexData in_vtx = pc.skinned_vertex_buffer.skinned_vertices[skin_task.skinned_vertex_i];
    vec4 in_pos = in_vtx.pos;
    vec4 in_norm = in_vtx.normal;
    vec4 pos = vec4(0.);
    vec4 normal = vec4(0.);
    int i = 0;

    // pos: weighted sum of transformed positions
    // normal: weighted inverse transpose of bone matrices
    for (; i != MAX_WEIGHTS; i++) {
        if (in_vtx.bone_id[i] == ~0) {
            break;
        }
        mat4 bone_mat = pc.bone_matrix_buffer.matrix[skin_task.bone_matrix_start_i + in_vtx.bone_id[i]];
        pos += bone_mat * in_pos * in_vtx.weights[i];
        normal += transpose(inverse(bone_mat)) * in_norm * in_vtx.weights[i];
    }
    if (i == 0) {
        pos.xyz = in_pos.xyz;
        normal.xyz = in_norm.xyz;
    }

    pc.out_vertices_buf.vertices[skin_task.out_vertex_i].pos = pos.xyz;
    pc.out_vertices_buf.vertices[skin_task.out_vertex_i].normal = normal.xyz;
}
