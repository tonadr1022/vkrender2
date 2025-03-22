#version 460

#include "../resources.h.glsl"
#include "./basic_common.h.glsl"

layout(location = 0) out vec3 out_normal;
layout(location = 1) out vec2 out_uv;
layout(location = 2) flat out uint material_id;

struct Vertex {
    vec3 pos;
    float uv_x;
    vec3 normal;
    float uv_y;
};

VK2_DECLARE_STORAGE_BUFFERS_RO(VertexBuffers){
Vertex vertices[];
} vertex_buffers[];

VK2_DECLARE_STORAGE_BUFFERS_RO(MaterialIDBuffer){
uint material_ids[];
} material_ids[];

VK2_DECLARE_STORAGE_BUFFERS_RO(InstanceBuffers){
mat4 instances[];
} instance_buffers[];

void main() {
    mat4 model = instance_buffers[instance_buffer].instances[gl_BaseInstance];
    Vertex v = vertex_buffers[vertex_buffer_idx].vertices[gl_VertexIndex];
    gl_Position = view_proj * model * vec4(v.pos, 1.);
    out_normal = normalize(v.normal) * .5 + .5;
    out_uv = vec2(v.uv_x, v.uv_y);
    material_id = material_ids[material_id_buffer].material_ids[gl_BaseInstance];
}
