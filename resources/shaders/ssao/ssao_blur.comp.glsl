#version 460 core

#extension GL_GOOGLE_include_directive : enable

#include "../common.h.glsl"

VK2_DECLARE_STORAGE_IMAGES(image2D);

layout(local_size_x = 16, local_size_y = 16) in;

layout(push_constant) uniform PC {
    uint ssao_in;
    uint blurred_out;
} pc;

void main() {
    ivec2 tex_coord = ivec2(gl_GlobalInvocationID);
    ivec2 img_size = imageSize(vk2_get_storage_img(image2D, pc.ssao_in));
    if (any(greaterThanEqual(tex_coord, img_size))) {
        return;
    }
    float blurred_result = 0.0;
    for (int x = -2; x < 2; x++) {
        for (int y = -2; y < 2; y++) {
            ivec2 offset = ivec2(x, y);
            ivec2 sample_pos = clamp(tex_coord + offset, ivec2(0), img_size - 1);
            blurred_result += imageLoad(vk2_get_storage_img(image2D, pc.ssao_in), sample_pos).r;
        }
    }
    blurred_result /= 16.0;
    imageStore(vk2_get_storage_img(image2D, pc.blurred_out), tex_coord, vec4(blurred_result, vec3(0.0)));
}
