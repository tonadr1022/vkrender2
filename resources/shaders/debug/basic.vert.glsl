#version 460

#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_nonuniform_qualifier : require

#define BINDLESS_STORAGE_BUFFER_BINDING 1

layout(location = 0) out vec3 out_normal;
layout(location = 1) out vec2 out_uv;
layout(location = 2) flat out uint material_id;

struct Vertex {
    vec3 pos;
    float uv_x;
    vec3 normal;
    float uv_y;
};

layout(std430, buffer_reference, buffer_reference_align = 32) buffer VertexBuffer {
    Vertex vertices[];
};

// TODO: more indirection?
struct InstanceData {
    mat4 model;
    uint material_id;
};

layout(set = 0, binding = BINDLESS_STORAGE_BUFFER_BINDING, std430) readonly buffer InstanceBuffers {
    InstanceData instances[];
} instance_buffers[];

layout(scalar, push_constant) uniform PC {
    mat4 view_proj;
    VertexBuffer vertex_buffer;
    uint instance_buffer;
} pc;

void main() {
    InstanceData data = instance_buffers[pc.instance_buffer].instances[gl_InstanceIndex];
    Vertex v = pc.vertex_buffer.vertices[gl_VertexIndex];
    gl_Position = pc.view_proj * data.model * vec4(v.pos, 1.);
    out_normal = normalize(v.normal) * .5 + .5;
    out_uv = vec2(v.uv_x, v.uv_y);
}
