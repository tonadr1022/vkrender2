#version 460

#extension GL_GOOGLE_include_directive : enable

#include "../resources.h.glsl"

layout(location = 0) in vec2 in_uv;

layout(location = 0) out vec4 out_frag_color;

VK2_DECLARE_SAMPLED_IMAGES(texture2DArray);

layout(push_constant) uniform PC {
    uint tex_idx;
    uint sampler_idx;
    uint array_idx;
} pc;

void main() {
    out_frag_color = vec4(vec3(texture(vk2_sampler2DArray(pc.tex_idx, pc.sampler_idx), vec3(in_uv, pc.array_idx)).r), 1.);
}
