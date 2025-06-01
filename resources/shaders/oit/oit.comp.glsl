#version 460

#extension GL_GOOGLE_include_directive : enable

#include "../resources.h.glsl"

layout(local_size_x = 16, local_size_y = 16) in;

#define MAX_FRAGMENTS 64

VK2_DECLARE_STORAGE_IMAGES_FORMAT(uimage2D, r32ui);
VK2_DECLARE_STORAGE_IMAGES(image2D);

struct OITTransparentFragment {
    f16vec4 color;
    float depth;
    uint next;
};

layout(scalar, buffer_reference) readonly buffer OITListsBuf {
    OITTransparentFragment frags[];
};

layout(push_constant) uniform PC {
    u64 oit_lists_buf;
    uint oit_heads_tex;
    float opacity_boost;
    uint color_tex;
    uint out_tex;
    float time;
    uint show_heatmap;
} pc;

void main() {
    ivec2 tex_coord = ivec2(gl_GlobalInvocationID);
    ivec2 img_size = imageSize(vk2_get_storage_img(uimage2D, pc.oit_heads_tex));
    if (tex_coord.x >= img_size.x || tex_coord.y >= img_size.y) {
        return;
    }

    // assemble local array of fragments for current pixel
    OITTransparentFragment frags[MAX_FRAGMENTS];
    uint num_frags = 0;
    uint idx = imageLoad(vk2_get_storage_img(uimage2D, pc.oit_heads_tex), tex_coord).r;
    while (idx != 0xFFFFFFFF && num_frags < MAX_FRAGMENTS) {
        frags[num_frags] = OITListsBuf(pc.oit_lists_buf).frags[idx];
        num_frags++;
        idx = OITListsBuf(pc.oit_lists_buf).frags[idx].next;
    }

    // insertion sort from largest to smallest depth
    for (int i = 1; i < num_frags; i++) {
        OITTransparentFragment to_insert = frags[i];
        uint j = i;
        while (j > 0 && to_insert.depth > frags[j - 1].depth) {
            frags[j] = frags[j - 1];
            j--;
        }
        frags[j] = to_insert;
    }

    // blend sorted frags
    vec4 color = imageLoad(vk2_get_storage_img(image2D, pc.color_tex), tex_coord);
    for (uint i = 0; i < num_frags; i++) {
        color = mix(color, vec4(frags[i].color), clamp(float(frags[i].color.a + pc.opacity_boost), 0.0, 1.0));
    }

    if (pc.show_heatmap != 0 && num_frags > 0) {
        color = (1. + sin(5. * pc.time)) * vec4(vec3(num_frags, num_frags, 0), 0.) / 16.;
    }

    // imageStore(vk2_get_storage_img(image2D, pc.out_tex), tex_coord, vec4(vec3(float(num_frags) / 16.0), 1.));
    imageStore(vk2_get_storage_img(image2D, pc.out_tex), tex_coord, color);
}
