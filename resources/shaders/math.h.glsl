#ifndef VK2_MATH_H
#define VK2_MATH_H
// courtesy of: zuex niagara engine

// A Survey of Efficient Representations for Independent Unit Vectors
vec2 encodeOct(vec3 v) {
    vec2 p = v.xy * (1.0 / (abs(v.x) + abs(v.y) + abs(v.z)));
    vec2 s = vec2((v.x >= 0.0) ? +1.0 : -1.0, (v.y >= 0.0) ? +1.0 : -1.0);
    vec2 r = (v.z <= 0.0) ? ((1.0 - abs(p.yx)) * s) : p;
    return r;
}

vec3 decodeOct(vec2 e) {
    // https://x.com/Stubbesaurus/status/937994790553227264
    vec3 v = vec3(e.xy, 1.0 - abs(e.x) - abs(e.y));
    float t = max(-v.z, 0);
    v.xy += vec2(v.x >= 0 ? -t : t, v.y >= 0 ? -t : t);
    return normalize(v);
}

vec3 tosrgb(vec3 c) {
    return pow(c.xyz, vec3(1.0 / 2.2));
}

vec4 tosrgb(vec4 c) {
    return vec4(pow(c.xyz, vec3(1.0 / 2.2)), c.w);
}

vec3 fromsrgb(vec3 c) {
    return pow(c.xyz, vec3(2.2));
}

vec4 fromsrgb(vec4 c) {
    return vec4(pow(c.xyz, vec3(2.2)), c.w);
}

// Optimized filmic operator by Jim Hejl and Richard Burgess-Dawson
// http://filmicworlds.com/blog/filmic-tonemapping-operators/
vec3 tonemap(vec3 c) {
    vec3 x = max(vec3(0), c - 0.004);
    return (x * (6.2 * x + .5)) / (x * (6.2 * x + 1.7) + 0.06);
}
vec3 gamma_correct(vec3 c) {
    return pow(c, vec3(1.0 / 2.2));
}
// https://64.github.io/tonemapping/
vec3 ACESFilm(vec3 x) {
    float a = 2.51;
    float b = 0.03;
    float c = 2.43;
    float d = 0.59;
    float e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}
#endif
