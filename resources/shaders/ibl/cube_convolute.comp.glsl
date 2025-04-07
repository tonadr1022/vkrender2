#version 460

// http://alinloghin.com/articles/compute_ibl.html

#extension GL_GOOGLE_include_directive : enable

#include "../resources.h.glsl"

layout(push_constant) uniform PC {
    uint in_tex_idx;
    uint out_tex_idx;
    uint sampler_idx;
    uint in_image_size_x;
};

layout(local_size_x = 16, local_size_y = 16) in;

VK2_DECLARE_STORAGE_IMAGES(imageCube);

vec3 get_world_dir(uvec3 coord, uint imageSize) {
    vec2 texCoord = vec2(coord.xy) / vec2(imageSize);
    texCoord = texCoord * 2.0 - 1.0; // -1..1
    switch (coord.z)
    {
        case 0:
        return vec3(1.0, -texCoord.yx); // posx
        case 1:
        return vec3(-1.0, -texCoord.y, texCoord.x); //negx
        case 2:
        return vec3(texCoord.x, 1.0, texCoord.y); // posy
        case 3:
        return vec3(texCoord.x, -1.0, -texCoord.y); //negy
        case 4:
        return vec3(texCoord.x, -texCoord.y, 1.0); // posz
        case 5:
        return vec3(-texCoord.xy, -1.0); // negz
    }
    return vec3(0.0);
}

ivec3 tex_coord_to_cube(vec3 texCoord, vec2 cubemapSize) {
    vec3 abst = abs(texCoord);
    texCoord /= max(max(abst.x, abst.y), abst.z);

    float cubeFace;
    vec2 uvCoord;
    if (abst.x > abst.y && abst.x > abst.z) {
        // x major
        float negx = step(texCoord.x, 0.0);
        uvCoord = mix(-texCoord.zy, vec2(texCoord.z, -texCoord.y), negx);
        cubeFace = negx;
    }
    else if (abst.y > abst.z) {
        // y major
        float negy = step(texCoord.y, 0.0);
        uvCoord = mix(texCoord.xz, vec2(texCoord.x, -texCoord.z), negy);
        cubeFace = 2.0 + negy;
    } else {
        // z major
        float negz = step(texCoord.z, 0.0);
        uvCoord = mix(vec2(texCoord.x, -texCoord.y), -texCoord.xy, negz);
        cubeFace = 4.0 + negz;
    }
    uvCoord = (uvCoord + 1.0) * 0.5; // 0..1
    uvCoord = uvCoord * cubemapSize;
    uvCoord = clamp(uvCoord, vec2(0.0), cubemapSize - vec2(1.0));
    return ivec3(ivec2(uvCoord), int(cubeFace));
}

const float PI = 3.14159265359;

void main() {
    ivec3 cube_coord = ivec3(gl_GlobalInvocationID);
    vec3 worldPos = get_world_dir(cube_coord, in_image_size_x);
    vec3 normal = normalize(worldPos);
    vec3 up = vec3(0.0, 1.0, 0.0);
    vec3 right = normalize(cross(up, normal));
    up = normalize(cross(normal, right));

    vec3 irradiance = vec3(0.0);
    float sample_delta = .025;
    float num_samples = 0.;
    for (float phi = 0.; phi < 2. * PI; phi += sample_delta) {
        for (float theta = 0.; theta < .5 * PI; theta += sample_delta) {
            float cosTheta = cos(theta);
            float sinTheta = sin(theta);
            float sinPhi = sin(phi);
            float cosPhi = cos(phi);
            vec3 tangent_sample = vec3(sinTheta * cosPhi, sinTheta * sinPhi, cosTheta);
            vec3 sample_vec = tangent_sample.x * right + tangent_sample.y * up + tangent_sample.z * normal;
            irradiance += imageLoad(vk2_get_storage_img(imageCube, in_tex_idx), tex_coord_to_cube(sample_vec, vec2(in_image_size_x))).rgb * cosTheta * sinTheta;
            num_samples++;
        }
    }
    irradiance *= PI * (1. / float(num_samples));
    // irradiance = texture(vk2_samplerCube(in_tex_idx, sampler_idx), normal).rgb;
    imageStore(vk2_get_storage_img(imageCube, out_tex_idx), cube_coord, vec4(irradiance, 1.));
}
