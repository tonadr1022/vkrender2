#version 460

#extension GL_GOOGLE_include_directive : enable

#include "../resources.h.glsl"

layout(push_constant) uniform PC {
    uint in_tex_idx;
    uint out_tex_idx;
    uint sampler_idx;
    uint in_image_size_x;
};

layout(local_size_x = 16, local_size_y = 16) in;

VK2_DECLARE_SAMPLED_IMAGES(textureCube);
VK2_DECLARE_STORAGE_IMAGES_WO(imageCube);

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

const float PI = 3.14159265359;

void main() {
    vec3 normal = get_world_dir(gl_GlobalInvocationID, in_image_size_x);
    vec3 irradiance = vec3(0.0);

    vec3 up = vec3(0.0, 1.0, 0.0);
    vec3 right = normalize(cross(up, normal));
    up = normalize(cross(normal, right));

    float sample_delta = .025;
    float num_samples = 0.;
    for (float phi = 0.; phi < 2. * PI; phi += sample_delta) {
        for (float theta = 0.; theta < .5 * PI; theta += sample_delta) {
            vec3 tangent_sample = vec3(sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta));
            vec3 sample_vec = tangent_sample.x * right + tangent_sample.y * up + tangent_sample.z * normal;
            irradiance += texture(vk2_samplerCube(in_tex_idx, sampler_idx), sample_vec).rgb * cos(theta) * sin(theta);
            num_samples++;
        }
    }
    irradiance = PI * irradiance * (1. / float(num_samples));
    imageStore(vk2_get_storage_img(imageCube, out_tex_idx), ivec3(gl_GlobalInvocationID), vec4(irradiance, 1.));
}
