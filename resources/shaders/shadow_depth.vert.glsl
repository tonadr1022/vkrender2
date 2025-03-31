#version 460
#extension GL_GOOGLE_include_directive : enable

#include "./common.h.glsl"
#include "./shadow_depth_common.h.glsl"

struct Vertex {
    vec3 pos;
    float uv_x;
    vec3 normal;
    float uv_y;
};

VK2_DECLARE_STORAGE_BUFFERS_RO(VertexBuffers){
Vertex vertices[];
} vertex_buffers[];

VK2_DECLARE_STORAGE_BUFFERS_RO(InstanceBuffers){
mat4 instances[];
} instance_buffers[];

void main() {
    mat4 model = instance_buffers[instance_buffer].instances[gl_BaseInstance];
    Vertex v = vertex_buffers[vertex_buffer_idx].vertices[gl_VertexIndex];
    vec4 pos = model * vec4(v.pos, 1.);
    gl_Position = vp_matrix * pos;
}
