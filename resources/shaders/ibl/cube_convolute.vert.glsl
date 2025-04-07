#version 460

#extension GL_GOOGLE_include_directive : enable
#include "../common.h.glsl"
#include "./cube_convolute_raster_common.h.glsl"
#include "../vertex_common.h.glsl"
// #include "../vertex_common.h.glsl"

layout(location = 0) out vec3 out_uv;

// const vec3 skybox_vertices[36] = vec3[](
//         vec3(-1.0f, 1.0f, -1.0f),
//         vec3(-1.0f, -1.0f, -1.0f),
//         vec3(1.0f, -1.0f, -1.0f),
//         vec3(1.0f, -1.0f, -1.0f),
//         vec3(1.0f, 1.0f, -1.0f),
//         vec3(-1.0f, 1.0f, -1.0f),
//         vec3(-1.0f, -1.0f, 1.0f),
//         vec3(-1.0f, -1.0f, -1.0f),
//         vec3(-1.0f, 1.0f, -1.0f),
//         vec3(-1.0f, 1.0f, -1.0f),
//         vec3(-1.0f, 1.0f, 1.0f),
//         vec3(-1.0f, -1.0f, 1.0f),
//         vec3(1.0f, -1.0f, -1.0f),
//         vec3(1.0f, -1.0f, 1.0f),
//         vec3(1.0f, 1.0f, 1.0f),
//         vec3(1.0f, 1.0f, 1.0f),
//         vec3(1.0f, 1.0f, -1.0f),
//         vec3(1.0f, -1.0f, -1.0f),
//         vec3(-1.0f, -1.0f, 1.0f),
//         vec3(-1.0f, 1.0f, 1.0f),
//         vec3(1.0f, 1.0f, 1.0f),
//         vec3(1.0f, 1.0f, 1.0f),
//         vec3(1.0f, -1.0f, 1.0f),
//         vec3(-1.0f, -1.0f, 1.0f),
//         vec3(-1.0f, 1.0f, -1.0f),
//         vec3(1.0f, 1.0f, -1.0f),
//         vec3(1.0f, 1.0f, 1.0f),
//         vec3(1.0f, 1.0f, 1.0f),
//         vec3(-1.0f, 1.0f, 1.0f),
//         vec3(-1.0f, 1.0f, -1.0f),
//         vec3(-1.0f, -1.0f, -1.0f),
//         vec3(-1.0f, -1.0f, 1.0f),
//         vec3(1.0f, -1.0f, -1.0f),
//         vec3(1.0f, -1.0f, -1.0f),
//         vec3(-1.0f, -1.0f, 1.0f),
//         vec3(1.0f, -1.0f, 1.0)
//     );

void main() {
    Vertex v = vertex_buffers[vertex_buffer_idx].vertices[gl_VertexIndex];
    gl_Position = vp * vec4(v.pos, 1.0);
    // gl_Position.z = 0.;
    out_uv = v.pos;
}
