#version 460

#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_nonuniform_qualifier : require

layout(location = 0) out vec3 out_normal;
layout(location = 1) out vec2 out_uv;

struct Vertex {
    vec3 pos;
    float uv_x;
    vec3 normal;
    float uv_y;
};

#define BINDLESS_STORAGE_BUFFER_BINDING 1

struct InstanceData {
    mat4 model;
};

layout(set = 0, binding = BINDLESS_STORAGE_BUFFER_BINDING, std430) readonly buffer InstanceBuffers {
    InstanceData instances[];
} instance_buffers[];

layout(std430, buffer_reference) readonly buffer VertexBuffer {
    Vertex vertices[];
};

layout(scalar, push_constant) uniform PC {
    mat4 view_proj;
    VertexBuffer vertex_buffer;
    uint instance_buffer;
};

void main() {
    mat4 model = instance_buffers[instance_buffer].instances[gl_DrawID].model;
    Vertex v = vertex_buffer.vertices[gl_VertexIndex];
    gl_Position = view_proj * model * vec4(v.pos, 1.);
    out_normal = normalize(v.normal) * .5 + .5;
    // out_uv = v.uv;
    out_uv = vec2(v.uv_x, v.uv_y);
}
