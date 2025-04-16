#version 460

#extension GL_GOOGLE_include_directive : enable
#include "../common.h.glsl"

layout(location = 0) in vec3 in_normal;
layout(location = 0) out vec4 out_frag_color;

void main() {
    out_frag_color = vec4(normalize(in_normal) * .5 + .5, 1.);
}
