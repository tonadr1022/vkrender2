#version 460

layout(location = 0) in vec3 in_normal;
layout(location = 0) out vec4 out_frag_color;

void main() {
    vec3 N = normalize(in_normal);
    if (!gl_FrontFacing) {
        N = -N;
    }
    out_frag_color = vec4(N * .5 + .5, 1.);
}
