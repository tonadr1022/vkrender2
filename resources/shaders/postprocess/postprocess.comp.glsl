#version 460

#extension GL_GOOGLE_include_directive : enable

#include "../resources.h.glsl"
#include "../math.h.glsl"

layout(local_size_x = 16, local_size_y = 16) in;

layout(push_constant) uniform PC {
    uint in_tex_idx;
    uint out_tex_idx;
    uint flags;
    uint tonemap_type;
};

#define TONEMAP_BIT 0x1
#define GAMMA_CORRECT_BIT 0x2
#define DISABLED_BIT 0x4

VK2_DECLARE_STORAGE_IMAGES(image2D);

void main() {
    ivec2 img_size = imageSize(vk2_get_storage_img(image2D, in_tex_idx)).xy;
    // TODO: vectorize
    if (gl_GlobalInvocationID.x >= img_size.x || gl_GlobalInvocationID.y >= img_size.y) {
        return;
    }
    vec4 color = imageLoad(vk2_get_storage_img(image2D, in_tex_idx), ivec2(gl_GlobalInvocationID.xy));
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
    imageStore(vk2_get_storage_img(image2D, out_tex_idx), ivec2(gl_GlobalInvocationID.xy), vec4(color.rgb, color.a));
}
