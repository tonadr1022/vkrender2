#version 460

#extension GL_GOOGLE_include_directive : enable

#include "../resources.h.glsl"
#include "./eq_to_cube_comp_common.h.glsl"

layout(local_size_x = 16, local_size_y = 16) in;

VK2_DECLARE_STORAGE_IMAGES_WO(imageCube);
VK2_DECLARE_SAMPLED_IMAGES(texture2D);

vec3 get_world_dir(uvec3 coord, uint imageSize) {
    // [0,1] -> [-1,1]
    vec2 texcoord = 2.0 * vec2(coord.xy) / imageSize - 1.0;
    switch (coord.z) {
        case 0U:
        return normalize(vec3(1.0, texcoord.y, -texcoord.x));
        case 1U:
        return normalize(vec3(-1.0, texcoord.yx));
        case 2U:
        return normalize(vec3(texcoord.x, -1.0, texcoord.y));
        case 3U:
        return normalize(vec3(texcoord.x, 1.0, -texcoord.y));
        case 4U:
        return normalize(vec3(texcoord, 1.0));
        case 5U:
        return normalize(vec3(-texcoord.x, texcoord.y, -1.0));
    }
    return vec3(0.0);
}

const vec2 invAtan = vec2(0.1591, 0.3183);

vec2 sample_spherical_map(vec3 p) {
    vec2 uv = vec2(atan(p.z, p.x), asin(p.y));
    uv *= invAtan;
    uv += .5;
    return uv;
}

void main() {
    vec2 uv = sample_spherical_map(get_world_dir(gl_GlobalInvocationID, imageSize(vk2_get_storage_img(imageCube, out_tex_idx)).x));
    vec4 color = texture(vk2_sampler2D(tex_idx, sampler_idx), uv);
    imageStore(vk2_get_storage_img(imageCube, out_tex_idx), ivec3(gl_GlobalInvocationID), color);
}
