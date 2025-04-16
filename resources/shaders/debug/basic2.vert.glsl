#version 460

#extension GL_GOOGLE_include_directive : enable
#include "../common.h.glsl"
#include "../vertex_common.h.glsl"

layout(location = 0) out vec3 out_normal;
layout(push_constant) uniform PC {
    mat4 vp;
    uint vertex_buffer_idx;
};

void main() {
    Vertex v = get_vertex(vertex_buffer_idx, gl_VertexIndex);
    gl_Position = vp * vec4(v.pos, 1.);
    out_normal = normalize(vec3(vec4(v.normal, 0.)));
}
