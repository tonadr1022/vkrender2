#version 460
// https://cdn2.unrealengine.com/Resources/files/2013SiggraphPresentationsNotes-26915738.pdf

// https://learnopengl.com/PBR/IBL/Specular-IBL

#extension GL_GOOGLE_include_directive : enable
#include "../resources.h.glsl"
#include "./prefilter_env_map_common.h.glsl"
#include "../math.h.glsl"
#include "../pbr/pbr.h.glsl"

VK2_DECLARE_SAMPLED_IMAGES(textureCube);

layout(location = 0) in vec3 in_pos;
layout(location = 0) out vec4 out_frag_color;

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

        float HdotV = max(dot(H, V), 0.0);
        float NdotH = max(dot(N, H), 0.0);

        // get mip level based on roughness based on pdf and roughness
        // solves bright spots
        float D = DistributionGGX(NdotH, roughness);
        float pdf = (D * NdotH / (4. * HdotV)) + 0.0001;
        float res = cubemap_res;
        float sa_texel = 4. * PI / (6. * res * res);
        float sa_sample = 1. / (float(SampleCount) * pdf + 0.0001);
        float mip_level = roughness == 0.0 ? 0.0 : 0.5 * log2(sa_sample / sa_texel);

        // reflect V along H to get L, sample and add contribution
        vec3 L = normalize(2. * HdotV * H - V);
        float NdotL = max(dot(N, L), 0.0);
        if (NdotL > 0.) {
            color += textureLod(vk2_samplerCube(cubemap_tex_idx, sampler_idx), L, mip_level).rgb * NdotL;
            total_weight += NdotL;
        }
    }
    color /= total_weight;
    out_frag_color = vec4(color, 1.);
}
