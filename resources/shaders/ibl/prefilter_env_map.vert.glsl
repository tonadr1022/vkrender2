#version 460

#extension GL_GOOGLE_include_directive : enable
#include "../resources.h.glsl"
#include "./prefilter_env_map_common.h.glsl"
#include "../vertex_common.h.glsl"

layout(location = 0) out vec3 out_uv;

void main() {
    Vertex v = vertex_buffers[vertex_buffer_idx].vertices[gl_VertexIndex];
    gl_Position = vp * vec4(v.pos, 1.0);
    // gl_Position.z = 0.;
    out_uv = v.pos;
}
