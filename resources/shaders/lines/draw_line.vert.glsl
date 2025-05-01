#version 460

#extension GL_GOOGLE_include_directive : enable

#define BDA 1
#include "../resources.h.glsl"
#include "../common.h.glsl"
#include "./draw_line_common.h.glsl"

struct Vertex {
    vec4 pos;
    vec4 color;
};

layout(scalar, buffer_reference) readonly buffer Vertices {
    Vertex vertices[];
};

layout(location = 0) out vec4 out_color;

void main() {
    Vertex v = Vertices(vtx).vertices[gl_VertexIndex];
    gl_Position = SceneDatas(scene_buffer).data.view_proj * vec4(v.pos.xyz, 1.);
    out_color = v.color;
}
