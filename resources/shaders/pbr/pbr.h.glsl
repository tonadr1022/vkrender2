#ifndef VK2_PBR_H
#define VK2_PBR_H

#define PI 3.14159265358979323846
// F0: surface reflection at zero incidence - how much surface reflects if looking directly at the surface.
// varies per material. tinted on metals. Common practice is to use 0.04 for dielectrics
vec3 FresnelSchlick(float cosTheta, vec3 F0) {
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}
vec3 FresnelSchlickRoughness(float cosTheta, vec3 F0, float roughness) {
    return F0 + (max(vec3(1.0 - roughness), F0) - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

// D in the Cook-Torrance BRDF: normal Distribution function.
// approximates the amount of the surface's microfacets that are aligned to the halfway vector, influenced by
// surface roughness
// This is the Trowbridge-Reitz GGX.
// when roughness is low, a highly concentrated number of microfacets are aligned to halfway vectors over a small radius
// causing a bright spot and darker over the other vectors. On a rough surface, microfacets are aligned in more
// random directions, so a much large number of halfway vectors are relatively aligned to the microfacets, but less concentrated,
// resulting in a more diffuse appearance
// roughness is used here because term specific modifications can be made.
float DistributionGGX(vec3 normal, vec3 halfVector, float roughness) {
    // Disney and Epic Games observed that squaring the roughness in the geometry and normal distribution function is the best
    // value for a.
    float a = roughness * roughness;
    float a2 = a * a;
    float NdotH = max(dot(normal, halfVector), 0.0);
    float NdotH2 = NdotH * NdotH;
    float numerator = a2;
    float denominator = (NdotH2 * (a2 - 1.0) + 1.0);
    denominator = PI * denominator * denominator;
    return numerator / denominator;
}

// approximates relative surface area where its micro surface details overshadow eachother, causing light rays to be
// excluded. 1.0 measures no shadowing, and 0.0 measures complete shadowing. Intuitively, there is more shadowing
// when cos(theta) is closer to 0 compared to looking at a surface straight on.
float GeometrySchlickGGX(float NdotV, float roughness) {
    // remap roughness for direct lighting. Why? Beyond me at this point
    float r = roughness + 1.0;
    float k = (r * r) / 8.0;

    // rougher surfaces have a higher probability of overshadowing microfacets
    float numerator = NdotV;
    float denominator = NdotV * (1.0 - k) + k;
    return numerator / denominator;
}

// to approximate geometry, need to take view direction (geometry obstruction) and light direction (geometry shadowing)
// into account by multiplying the two calculations.
float GeometrySmith(vec3 N, vec3 V, vec3 L, float roughness) {
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    float ggx1 = GeometrySchlickGGX(NdotL, roughness);
    float ggx2 = GeometrySchlickGGX(NdotV, roughness);
    return ggx1 * ggx2;
}

#endif
