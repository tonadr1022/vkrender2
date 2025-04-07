#version 460

#extension GL_GOOGLE_include_directive : enable

#include "../resources.h.glsl"
#include "./cube_convolute_raster_common.h.glsl"

layout(location = 0) in vec3 in_normal;

layout(location = 0) out vec4 out_frag_color;

VK2_DECLARE_SAMPLED_IMAGES(textureCube);
VK2_DECLARE_STORAGE_IMAGES_WO(imageCube);

const float PI = 3.14159265359;

void main() {
    vec3 N = normalize(in_normal);
    vec3 irradiance = vec3(0.0);

    vec3 up = vec3(0.0, 1.0, 0.0);
    vec3 right = normalize(cross(up, N));
    up = normalize(cross(N, right));

    float sample_delta = .025;
    float num_samples = 0.;
    for (float phi = 0.; phi < 2. * PI; phi += sample_delta) {
        for (float theta = 0.; theta < .5 * PI; theta += sample_delta) {
            // spherical to cartesian, tangent space
            vec3 tangent_sample = vec3(sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta));
            // tangent to world
            vec3 sample_vec = tangent_sample.x * right + tangent_sample.y * up + tangent_sample.z * N;
            irradiance += texture(vk2_samplerCube(in_tex_idx, sampler_idx), sample_vec).rgb * cos(theta) * sin(theta);
            num_samples++;
        }
    }
    irradiance = PI * irradiance * (1. / float(num_samples));
    out_frag_color = vec4(irradiance, 1.);
}
