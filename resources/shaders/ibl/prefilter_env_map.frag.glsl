#version 460
// https://cdn2.unrealengine.com/Resources/files/2013SiggraphPresentationsNotes-26915738.pdf

// https://learnopengl.com/PBR/IBL/Specular-IBL

#extension GL_GOOGLE_include_directive : enable
#include "../resources.h.glsl"
#include "./prefilter_env_map_common.h.glsl"
#include "../math.h.glsl"

VK2_DECLARE_SAMPLED_IMAGES(textureCube);

layout(location = 0) in vec3 in_pos;
layout(location = 0) out vec4 out_frag_color;

float radical_inverse_VdC(uint bits) {
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return float(bits) * 2.3283064365386963e-10; // / 0x100000000
}

vec2 hammersley(uint i, uint N) {
    return vec2(float(i) / float(N), radical_inverse_VdC(i));
}

vec3 importance_sample_ggx(vec2 Xi, vec3 N, float roughness) {
    float a = roughness * roughness;

    float phi = 2.0 * PI * Xi.x;
    float cosTheta = sqrt((1.0 - Xi.y) / (1.0 + (a * a - 1.0) * Xi.y));
    float sinTheta = sqrt(1.0 - cosTheta * cosTheta);

    // from spherical coordinates to cartesian coordinates
    vec3 H;
    H.x = cos(phi) * sinTheta;
    H.y = sin(phi) * sinTheta;
    H.z = cosTheta;

    // from tangent-space vector to world-space sample vector
    vec3 up = abs(N.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
    vec3 tangent = normalize(cross(up, N));
    vec3 bitangent = cross(N, tangent);

    vec3 sampleVec = tangent * H.x + bitangent * H.y + N * H.z;
    return normalize(sampleVec);
}

const uint SampleCount = 1024u;

void main() {
    vec3 N = normalize(in_pos);
    vec3 R = N;
    vec3 V = R;
    float total_weight = 0.;
    vec3 color = vec3(0.);
    for (uint i = 0u; i < SampleCount; i++) {
        // get quasi-random pt
        vec2 Xi = hammersley(i, SampleCount);
        // sample a half vector in specular lobe
        vec3 H = importance_sample_ggx(Xi, N, roughness);
        // reflect V along H to get L, sample and add contribution
        vec3 L = normalize(2. * dot(V, H) * H - V);
        float NdotL = max(dot(N, L), 0.0);
        if (NdotL > 0.) {
            color += texture(vk2_samplerCube(cubemap_tex_idx, sampler_idx), L).rgb * NdotL;
            total_weight += NdotL;
        }
    }
    color /= total_weight;
    out_frag_color = vec4(color, 1.);
}
