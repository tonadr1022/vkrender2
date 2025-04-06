#version 460

#extension GL_GOOGLE_include_directive : enable

#include "./equirect_to_cube_common.h.glsl"

VK2_DECLARE_SAMPLED_IMAGES(texture2D);

layout(location = 0) in vec3 pos;
layout(location = 0) out vec4 out_frag_color;

const vec2 invAtan = vec2(0.1591, 0.3183);

vec2 sample_spherical_map(vec3 p) {
    vec2 uv = vec2(atan(p.z, p.x), asin(p.y));
    uv *= invAtan;
    uv += .5;
    return uv;
}

void main() {
    vec2 uv = sample_spherical_map(pos);
    vec4 color = texture(vk2_sampler2D(tex_idx, sampler_idx), uv);
    out_frag_color = vec4(color.rgb, 1.);
}
