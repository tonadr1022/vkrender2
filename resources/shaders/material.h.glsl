struct Material {
    vec4 emissive_factors;
    vec4 albedo_factors;
    vec4 pbr_factors; // x is metallic, y is roughness, w is alpha cutoff
    uvec4 ids; // albedo, normal, metal_rough, emissive
    uvec4 ids2; // ao, w is flags
};
