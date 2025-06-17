#version 460

#extension GL_GOOGLE_include_directive : enable

#include "../resources.h.glsl"
#include "../math.h.glsl"

layout(location = 0) in vec2 in_uv;
layout(location = 0) out vec4 out_frag_color;

layout(push_constant) uniform PC {
    uint in_tex_idx;
    uint out_tex_idx;
    uint flags;
    uint tonemap_type;
};

#define TONEMAP_BIT 0x1
#define GAMMA_CORRECT_BIT 0x2
#define DISABLED_BIT 0x4

VK2_DECLARE_SAMPLED_IMAGES(texture2D);

void main() {
    vec4 color = texture(vk2_sampler2D(in_tex_idx, 0), in_uv);
    bool disabled = (flags & DISABLED_BIT) != 0;
    if (!disabled && (flags & TONEMAP_BIT) != 0) {
        if (tonemap_type == 0) {
            color.rgb = tonemap(color.rgb);
        } else {
            color.rgb = ACESFilm(color.rgb);
        }
    }
    if (!disabled && (flags & GAMMA_CORRECT_BIT) != 0) {
        color.rgb = gamma_correct(color.rgb);
    }
    out_frag_color = color;
}
