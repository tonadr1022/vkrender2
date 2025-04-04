#version 460
#extension GL_GOOGLE_include_directive : enable

#include "./common.h.glsl"
#include "./shadow_depth_common.h.glsl"

layout(location = 0) out vec2 out_uv;
layout(location = 1) flat out uint material_id;
struct Vertex {
    vec3 pos;
    float uv_x;
    vec3 normal;
    float uv_y;
    vec4 tangent;
};

VK2_DECLARE_STORAGE_BUFFERS_RO_SCALAR(VertexBuffers){
Vertex vertices[];
} vertex_buffers[];

struct InstanceData {
    uint material_id;
    uint instance_id;
};

struct ObjectData {
    mat4 model;
    vec4 sphere_radius;
    vec4 extent;
};

VK2_DECLARE_STORAGE_BUFFERS_RO(ObjectDataBuffer){
InstanceData datas[];
} instance_buffers[];

VK2_DECLARE_STORAGE_BUFFERS_RO(InstanceDataBuffers){
ObjectData datas[];
} object_data_buffers[];

void main() {
    InstanceData instance_data = instance_buffers[instance_buffer].datas[gl_InstanceIndex];
    Vertex v = vertex_buffers[vertex_buffer_idx].vertices[gl_VertexIndex];
    mat4 model = object_data_buffers[object_data_buffer].datas[instance_data.instance_id].model;
    vec4 pos = model * vec4(v.pos, 1.);
    gl_Position = vp_matrix * pos;
    out_uv = vec2(v.uv_x, v.uv_y);
    material_id = instance_data.material_id;
}
