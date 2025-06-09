#version 460

#define MAX_WEIGHTS 8

#extension GL_GOOGLE_include_directive : enable
#include "../resources.h.glsl"

#define VERTEX_UNDEF
#include "../vertex_common.h.glsl"

struct SkinnedVertexData {
    vec4 pos;
    vec4 normal;
    u32 bone_id[MAX_WEIGHTS];
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

layout(push_constant) uniform PC {
    BoneMatrices bone_matrix_buffer;
    VertexBuffer out_vertices_buf;
    SkinnedVertexBuffer skinned_vertex_buffer;
    u32 start_out_vertex_index;
    u32 start_skinned_vertex_index;
} pc;

void main() {
    uint index = gl_GlobalInvocationID.x;
    SkinnedVertexData in_vtx = pc.skinned_vertex_buffer.skinned_vertices[pc.start_skinned_vertex_index + index];
    vec4 in_pos = in_vtx.pos;
    vec4 in_norm = in_vtx.normal;
    vec4 pos = vec4(0.);
    vec4 normal = vec4(0.);
    int i = 0;
    // pos: weighted sum of transformed positions
    // normal: inverse transpose of bone matrices
    for (; i != MAX_WEIGHTS; i++) {
        if (in_vtx.bone_id[i] == ~0) {
            break;
        }
        mat4 bone_mat = pc.bone_matrix_buffer.matrix[in_vtx.bone_id[i]];
        pos += bone_mat * in_pos * in_vtx.weights[i];
        normal += transpose(inverse(bone_mat)) * in_norm * in_vtx.weights[i];
    }
    if (i == 0) {
        pos.xyz = in_pos.xyz;
        normal.xyz = in_norm.xyz;
    }

    pc.out_vertices_buf.vertices[pc.start_out_vertex_index + index].pos = pos.xyz;
    pc.out_vertices_buf.vertices[pc.start_out_vertex_index + index].normal = normal.xyz;
}
