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

layout(set = 0, binding = BINDLESS_STORAGE_BUFFER_BINDING) restrict readonly buffer VertexBuffer {
    Vertex vertices[];
} vertex_buffers[];

// TODO: more indirection?
struct InstanceData {
    mat4 model;
};

layout(set = 0, binding = BINDLESS_STORAGE_BUFFER_BINDING, std430) restrict readonly buffer MaterialIDBuffer {
    uint material_ids[];
} material_ids[];

layout(set = 0, binding = BINDLESS_STORAGE_BUFFER_BINDING, std430) restrict readonly buffer InstanceBuffers {
    InstanceData instances[];
} instance_buffers[];

layout(push_constant) uniform PC {
    mat4 view_proj;
    uint vertex_buffer_idx;
    uint instance_buffer;
    uint materials_buffer;
    uint material_id_buffer;
} pc;

void main() {
    InstanceData data = instance_buffers[pc.instance_buffer].instances[gl_BaseInstance];
    Vertex v = vertex_buffers[pc.vertex_buffer_idx].vertices[gl_VertexIndex];
    gl_Position = pc.view_proj * data.model * vec4(v.pos, 1.);
    out_normal = normalize(v.normal) * .5 + .5;
    out_uv = vec2(v.uv_x, v.uv_y);
    material_id = material_ids[pc.material_id_buffer].material_ids[gl_BaseInstance];
}
