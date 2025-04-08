layout(push_constant) uniform PC {
    mat4 vp;
    float roughness;
    uint cubemap_tex_idx;
    uint sampler_idx;
    uint vertex_buffer_idx;
};
