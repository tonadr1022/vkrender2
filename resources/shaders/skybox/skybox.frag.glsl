#version 460

#extension GL_GOOGLE_include_directive : enable
#include "../resources.h.glsl"
#include "./skybox_common.h.glsl"

layout(location = 0) in vec3 in_uv;
layout(location = 0) out vec4 out_frag_color;

VK2_DECLARE_SAMPLED_IMAGES(textureCube);

#define REVERSE_DEPTH 1

void main() {
    #ifdef REVERSE_DEPTH
    // TODO: don't do this here
    // gl_FragDepth = 0.0;
    #endif

    vec4 color = texture(vk2_samplerCube(tex_idx, LINEAR_SAMPLER_BINDLESS_IDX), in_uv);
    out_frag_color = vec4(color.rgb, 1.);
}
