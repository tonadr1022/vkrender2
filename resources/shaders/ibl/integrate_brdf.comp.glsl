#version 460

#extension GL_GOOGLE_include_directive : enable
#include "../resources.h.glsl"

#define GEOMETRY_SMITH_K_IBL
#include "../pbr/pbr.h.glsl"

layout(local_size_x = 16, local_size_y = 16) in;

VK2_DECLARE_STORAGE_IMAGES(image2D);

layout(push_constant) uniform PC {
    uint in_tex_idx;
    uint sampler_idx;
};

vec2 integrate_brdf(float NdotV, float roughness) {
    vec3 V;
    V.x = sqrt(1.0 - NdotV * NdotV);
    V.y = 0.0;
    V.z = NdotV;

    float A = 0.0;
    float B = 0.0;

    vec3 N = vec3(0.0, 0.0, 1.0);

    const uint SAMPLE_COUNT = 1024u;
    for (uint i = 0u; i < SAMPLE_COUNT; ++i)
    {
        vec2 Xi = hammersley(i, SAMPLE_COUNT);
        vec3 H = importance_sample_ggx(Xi, N, roughness);
        vec3 L = normalize(2.0 * dot(V, H) * H - V);

        float NdotL = max(L.z, 0.0);
        float NdotH = max(H.z, 0.0);
        float VdotH = max(dot(V, H), 0.0);

        if (NdotL > 0.0)
        {
            float G = GeometrySmith(N, V, L, roughness);
            float G_Vis = (G * VdotH) / (NdotH * NdotV);
            float Fc = pow(1.0 - VdotH, 5.0);

            A += (1.0 - Fc) * G_Vis;
            B += Fc * G_Vis;
        }
    }
    A /= float(SAMPLE_COUNT);
    B /= float(SAMPLE_COUNT);
    return vec2(A, B);
};

void main() {
    ivec2 tex_coord = ivec2(gl_GlobalInvocationID.xy);
    ivec2 img_size = imageSize(vk2_get_storage_img(image2D, in_tex_idx));
    if (any(greaterThanEqual(tex_coord, img_size))) {
        return;
    }
    vec2 uv = (vec2(tex_coord) + vec2(.5)) / vec2(img_size);
    uv.y = 1. - uv.y;
    vec2 color = integrate_brdf(uv.x, uv.y);
    imageStore(vk2_get_storage_img(image2D, in_tex_idx), tex_coord, vec4(color, 0., 1.));
}
