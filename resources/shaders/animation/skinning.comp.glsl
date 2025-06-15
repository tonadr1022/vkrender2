#version 460

#extension GL_GOOGLE_include_directive : enable
#include "../resources.h.glsl"

#define VERTEX_UNDEF
#include "../vertex_common.h.glsl"

#define MAX_WEIGHTS 4
struct SkinnedVertexData {
    vec3 pos;
    uint instance_i;
    vec4 normal;
    vec4 tangent;
    u32 bone_id[MAX_WEIGHTS];
    float weights[MAX_WEIGHTS];
};

struct InstanceData {
    uint first_bone_matrix_i;
    uint first_vertex_i;
};

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

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
    u64 cnt;
    // InstanceDataBuffer instance_data_buf;
} pc;

void main() {
    uint index = gl_GlobalInvocationID.x;
    if (index >= pc.cnt) {
        return;
    }
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

    vec4 in_pos = vec4(in_vtx.pos, 1.);
    vec4 in_norm = vec4(in_vtx.normal.xyz, 0.);
    vec4 in_tangent = vec4(in_vtx.normal.xyz, 0.);
    vec4 pos = vec4(0.);
    vec4 normal = vec4(0.);
    vec4 tangent = vec4(0.);

    bool has_valid_weight = false;
    int i = 0;
    for (; i < MAX_WEIGHTS; ++i) {
        uint bone_index = in_vtx.bone_id[i] + instance_data.first_bone_matrix_i;
        mat4 bone_mat = pc.bone_matrix_buffer.matrix[bone_index];
        pos += bone_mat * in_pos * in_vtx.weights[i];

        mat3 normal_matrix = transpose(inverse(mat3(bone_mat)));
        normal.xyz += normal_matrix * in_norm.xyz * in_vtx.weights[i];
        tangent.xyz += normal_matrix * in_tangent.xyz * in_vtx.weights[i];
    }

    if (i == 0) {
        pos = in_pos;
        normal = in_norm;
        tangent = in_tangent;
    }

    pc.out_vertices_buf.vertices[output_vertex_i].pos = pos.xyz;
    pc.out_vertices_buf.vertices[output_vertex_i].normal.xyz = normal.xyz;
    pc.out_vertices_buf.vertices[output_vertex_i].tangent.xyz = tangent.xyz;
}
