#version 460

#include "./test.glsl"

#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : require

struct Vertex {
    vec3 pos;
    vec3 normal;
    vec2 uv;
};

layout(std430, buffer_reference) readonly buffer VertexBuffer {
    Vertex vertices[];
};

layout(scalar, push_constant) uniform PC {
    mat4 view_proj;
    VertexBuffer vertex_buffer;
};

layout(location = 0) out vec3 out_normal;
layout(location = 1) out vec2 out_uv;

void main() {
    Vertex v = vertex_buffer.vertices[gl_VertexIndex];
    gl_Position = view_proj * vec4(v.pos, 1.);
    out_normal = normalize(v.normal);
    out_uv = v.uv;
}
