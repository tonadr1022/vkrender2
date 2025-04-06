#version 460

#extension GL_GOOGLE_include_directive : enable

#include "./equirect_to_cube_common.h.glsl"

struct Vertex {
    vec3 pos;
    float uv_x;
    vec3 normal;
    float uv_y;
    vec4 tangent;
};
layout(location = 0) out vec3 local_pos;

VK2_DECLARE_STORAGE_BUFFERS_RO_SCALAR(VertexBuffers){
Vertex vertices[];
} vertex_buffers[];

void main() {
    Vertex v = vertex_buffers[vertex_buffer_idx].vertices[gl_VertexIndex];
    gl_Position = view_proj * vec4(v.pos, 1.);
    local_pos = v.pos;
}
