#version 460

#extension GL_GOOGLE_include_directive : enable

#include "../resources.h.glsl"

layout(push_constant) uniform PC {
    vec2 input_resolution;
    vec2 output_resolution;
    uint input_img;
} pc;

VK2_DECLARE_SAMPLED_IMAGES(texture2D);

layout(location = 0) in vec2 in_uv;
layout(location = 0) out vec4 out_frag_color;

void main() {
    // vec2 scaled_uv = in_uv * (pc.input_resolution / pc.output_resolution);
    out_frag_color = texture(vk2_sampler2D(pc.input_img, 1), in_uv);
}
