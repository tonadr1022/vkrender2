#version 460

#extension GL_GOOGLE_include_directive : enable

#include "../../common.h.glsl"

layout(location = 0) in vec2 in_uv;
layout(location = 0) out vec4 out_frag_color;

layout(push_constant) uniform PC {
    uint input_img;
    float fixed_threshold;
    float relative_threshold;
    float subpixel_blending;
    ivec2 img_size;
} pc;

VK2_DECLARE_STORAGE_IMAGES(image2D);
VK2_DECLARE_SAMPLED_IMAGES(texture2D);

// TODO: replace with variable for sampler idx. in renderer it's set to 3 right now but tbd
#define SAMP(x) texture(vk2_sampler2D(pc.input_img, 3), x)
#define STORE(x) out_frag_color = x

struct LumaNeighborhood {
    float m, n, e, s, w, ne, se, sw, nw;
    float highest, lowest, range;
};

float luminance(vec3 color) {
    return dot(color, vec3(0.299, 0.587, 0.114));
}

float get_luma(vec2 uv) {
    return sqrt(luminance(SAMP(uv).xyz));
}

float get_luma(vec2 uv, vec2 offset) {
    return get_luma(uv + offset);
}

float get_subpixel_blend_factor(in LumaNeighborhood luma) {
    // [1,2,1]
    // [2,m,2]
    // [1,2,1]
    float filt = 2.0 * (luma.n + luma.e + luma.s + luma.w);
    filt += luma.ne + luma.nw + luma.se + luma.sw;
    filt *= 1.0 / 12.0;

    filt = abs(filt - luma.m);
    filt = clamp(filt / luma.range, 0, 1);
    filt = smoothstep(0, 1, filt);

    return filt * filt * pc.subpixel_blending;
}

struct FXAAEdge {
    float pixel_step;
    float luma_gradient, other_luma;
    bool is_horiz;
};

bool is_horizontal_edge(in LumaNeighborhood luma) {
    float horiz = 2.0 * abs(luma.n + luma.s - 2.0 * luma.m) +
            abs(luma.ne + luma.se - 2.0 * luma.e) +
            abs(luma.nw + luma.sw - 2.0 * luma.w);
    float vert = 2.0 * abs(luma.e + luma.w - 2.0 * luma.m) +
            abs(luma.ne + luma.nw - 2.0 * luma.n) +
            abs(luma.se + luma.sw - 2.0 * luma.s);
    return horiz >= vert;
}

void get_fxaa_edge(out FXAAEdge edge, in LumaNeighborhood luma, vec2 texel_size) {
    edge.is_horiz = is_horizontal_edge(luma);
    float lumaP, lumaN;
    if (edge.is_horiz) {
        edge.pixel_step = texel_size.y;
        lumaP = luma.s;
        lumaN = luma.n;
    } else {
        edge.pixel_step = texel_size.x;
        lumaP = luma.e;
        lumaN = luma.w;
    }

    // pos/neg gradients
    float gradient_p = abs(lumaP - luma.m);
    float gradient_n = abs(lumaN - luma.m);
    if (gradient_p < gradient_n) {
        edge.pixel_step = -edge.pixel_step;
        edge.luma_gradient = gradient_p;
        edge.other_luma = lumaN;
    } else {
        edge.luma_gradient = gradient_n;
        edge.other_luma = lumaP;
    }
}
#define EXTRA_EDGE_STEPS 10
#define EDGE_STEP_SIZES 1.0, 1.0, 1.0, 1.0, 1.5, 2.0, 2.0, 2.0, 2.0, 4.0
#define LAST_EDGE_STEP_GUESS 8.0
const float edge_step_sizes[EXTRA_EDGE_STEPS] = float[](EDGE_STEP_SIZES);

float get_edge_blend_factor(in LumaNeighborhood luma, in FXAAEdge edge, vec2 uv, vec2 texel_size) {
    vec2 edge_uv = uv;
    vec2 uv_step = vec2(0.0);
    if (edge.is_horiz) {
        edge_uv.y += .5 * edge.pixel_step;
        uv_step.x = texel_size.x;
    } else {
        edge_uv.x += .5 * edge.pixel_step;
        uv_step.y = texel_size.y;
    }
    float edge_luma = 0.5 * (luma.m + edge.other_luma);
    float gradient_thresh = 0.25 * edge.luma_gradient;

    vec2 uv_p = edge_uv + uv_step;
    float luma_delta_p = get_luma(uv_p) - edge_luma;
    bool at_end_p = abs(luma_delta_p) >= gradient_thresh;
    int i;
    for (i = 0; i < EXTRA_EDGE_STEPS && !at_end_p; i++) {
        uv_p += uv_step * edge_step_sizes[i];
        luma_delta_p = get_luma(uv_p) - edge_luma;
        at_end_p = abs(luma_delta_p) >= gradient_thresh;
    }
    if (!at_end_p) {
        uv_p += uv_step * LAST_EDGE_STEP_GUESS;
    }

    vec2 uv_n = edge_uv - uv_step;
    float luma_delta_n = get_luma(uv_n) - edge_luma;
    bool at_end_n = abs(luma_delta_n) >= gradient_thresh;
    for (i = 0; i < EXTRA_EDGE_STEPS && !at_end_n; i++) {
        uv_n -= uv_step * edge_step_sizes[i];
        luma_delta_n = get_luma(uv_n) - edge_luma;
        bool at_end_n = abs(luma_delta_n) >= gradient_thresh;
    }
    if (!at_end_n) {
        uv_n -= uv_step * LAST_EDGE_STEP_GUESS;
    }

    float dist_to_end_p, dist_to_end_n;
    if (edge.is_horiz) {
        dist_to_end_p = uv_p.x - uv.x;
        dist_to_end_n = uv.x - uv_n.x;
    } else {
        dist_to_end_p = uv_p.y - uv.y;
        dist_to_end_n = uv.y - uv_n.y;
    }
    bool delta_sign;
    float dist_to_nearest_end;
    if (dist_to_end_p <= dist_to_end_n) {
        dist_to_nearest_end = dist_to_end_p;
        delta_sign = luma_delta_p >= 0;
    } else {
        dist_to_nearest_end = dist_to_end_n;
        delta_sign = luma_delta_n >= 0;
    }
    if (delta_sign == (luma.n - edge_luma >= 0)) {
        return 0.0;
    }
    return 0.5 - dist_to_nearest_end / (dist_to_end_p + dist_to_end_n);

    return edge.luma_gradient;
}

void get_luma_neighborhood(vec2 uv, vec2 texel_size, out LumaNeighborhood luma) {
    luma.m = get_luma(uv);
    luma.n = get_luma(uv, vec2(0.0, 1.0) * texel_size);
    luma.e = get_luma(uv, vec2(1.0, 0.0) * texel_size);
    luma.s = get_luma(uv, vec2(0.0, -1.0) * texel_size);
    luma.w = get_luma(uv, vec2(-1.0, 0.0) * texel_size);
    luma.ne = get_luma(uv, vec2(1.0, 1.0) * texel_size);
    luma.se = get_luma(uv, vec2(1.0, -1.0) * texel_size);
    luma.sw = get_luma(uv, vec2(-1.0, -1.0) * texel_size);
    luma.nw = get_luma(uv, vec2(-1.0, 1.0) * texel_size);

    luma.highest = max(max(max(max(luma.m, luma.n), luma.e), luma.s), luma.w);
    luma.lowest = min(min(min(min(luma.m, luma.n), luma.e), luma.s), luma.w);
    luma.range = luma.highest - luma.lowest;
}

void main() {
    vec2 texel_size = 1.0 / vec2(pc.img_size);
    vec2 uv = in_uv;
    vec4 original = SAMP(uv);
    LumaNeighborhood luma;
    get_luma_neighborhood(uv, texel_size, luma);

    vec4 result = vec4(0.0);

    if (luma.range < max(pc.fixed_threshold, pc.relative_threshold * luma.highest)) {
        result = original;
    } else {
        FXAAEdge edge;
        get_fxaa_edge(edge, luma, texel_size);
        float blend_factor = max(get_subpixel_blend_factor(luma), get_edge_blend_factor(luma, edge, uv, texel_size));
        vec2 blend_uv = uv;
        if (edge.is_horiz) {
            blend_uv.y += blend_factor * edge.pixel_step;
        } else {
            blend_uv.x += blend_factor * edge.pixel_step;
        }
        result = SAMP(blend_uv);
    }
    STORE(result);
}
