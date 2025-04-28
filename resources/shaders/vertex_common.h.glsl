#ifndef VERTEX_COMMON_H
#define VERTEX_COMMON_H

struct Vertex {
    vec3 pos;
    float uv_x;
    vec3 normal;
    float uv_y;
    vec4 tangent;
};

#ifdef BDA
layout(std430, buffer_reference) readonly buffer Vertices {
    Vertex vertices[];
};
#else

VK2_DECLARE_STORAGE_BUFFERS_RO_SCALAR(VertexBuffers){
Vertex vertices[];
} vertex_buffers[];

Vertex get_vertex(uint vertex_buffer_idx) {
    return vertex_buffers[vertex_buffer_idx].vertices[gl_VertexIndex];
}
Vertex get_vertex(uint vertex_buffer_idx, uint vertex_index) {
    return vertex_buffers[vertex_buffer_idx].vertices[vertex_index];
}
#endif

#endif
