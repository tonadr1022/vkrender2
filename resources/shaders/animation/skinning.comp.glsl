#version 460

#extension GL_GOOGLE_include_directive : enable
#include "../resources.h.glsl"

#define VERTEX_UNDEF
#include "../vertex_common.h.glsl"

#define MAX_WEIGHTS 8
struct SkinnedVertexData {
    vec3 pos;
    uint instance_i;
    vec4 normal;
    // uint output_vertex_i;
    u32 bone_id[MAX_WEIGHTS];
    float weights[MAX_WEIGHTS];
};

struct InstanceData {
    uint first_bone_matrix_i;
    uint first_vertex_i;
};

layout(local_size_x = 16, local_size_y = 1, local_size_z = 1) in;

layout(std430, buffer_reference) readonly buffer BoneMatrices {
    mat4 matrix[];
};

layout(scalar, buffer_reference) readonly buffer InstanceDataBuffer {
    InstanceData instances[];
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
    // InstanceDataBuffer instance_data_buf;
} pc;

void main() {
    uint index = gl_GlobalInvocationID.x;
    SkinnedVertexData in_vtx = pc.skinned_vertex_buffer.skinned_vertices[index];
    if (in_vtx.instance_i == ~0u) {
        return;
    }
    // InstanceData instance_data = pc.instance_data_buf.instances[in_vtx.instance_i];
    InstanceData instance_data;
    instance_data.first_bone_matrix_i = 0;
    instance_data.first_vertex_i = 0;
    uint output_vertex_i = index;
    // uint output_vertex_i = in_vtx.output_vertex_i;

    vec4 in_pos = vec4(in_vtx.pos, 0.);
    vec4 in_norm = vec4(in_vtx.normal);
    vec4 pos = vec4(0.);
    vec4 normal = vec4(0.);
    int i = 0;
    // pos: weighted sum of transformed positions
    // normal: inverse transpose of bone matrices
    for (; i != MAX_WEIGHTS; i++) {
        if (in_vtx.bone_id[i] == ~0) {
            break;
        }
        mat4 bone_mat = pc.bone_matrix_buffer.matrix[in_vtx.bone_id[i] + instance_data.first_bone_matrix_i];
        pos += bone_mat * in_pos * in_vtx.weights[i];
        normal += transpose(inverse(bone_mat)) * in_norm * in_vtx.weights[i];
    }
    if (i == 0) {
        pos.xyz = in_pos.xyz;
        normal.xyz = in_norm.xyz;
    }

    pos = in_pos;
    normal = in_norm;
    pc.out_vertices_buf.vertices[output_vertex_i].pos = pos.xyz;
    pc.out_vertices_buf.vertices[output_vertex_i].normal = normal.xyz;
}
