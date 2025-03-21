#version 460

#extension GL_EXT_buffer_reference : require
#extension GL_EXT_nonuniform_qualifier : require

layout(location = 0) in vec3 in_normal;
layout(location = 1) in vec2 in_uv;
layout(location = 2) flat in uint material_id;

layout(location = 0) out vec4 out_frag_color;

struct Material {
    uint albedo_id;
    uint normal_id;
};

#define BINDLESS_STORAGE_BUFFER_BINDING 1
#define BINDLESS_SAMPLER_BINDING 4
#define BINDLESS_SAMPLED_IMAGE_BINDING 2

layout(set = 0, binding = BINDLESS_SAMPLER_BINDING) uniform sampler samplers[];

layout(set = 0, binding = BINDLESS_SAMPLED_IMAGE_BINDING) uniform texture2D textures[];

layout(set = 0, binding = BINDLESS_STORAGE_BUFFER_BINDING, std430) readonly buffer MaterialBuffers {
    Material materials[];
} materials[];

struct Vertex {
    vec3 pos;
    float uv_x;
    vec3 normal;
    float uv_y;
};

layout(std430, buffer_reference, buffer_reference_align = 32) buffer VertexBuffer {
    Vertex vertices[];
};

layout(push_constant) uniform PC {
    mat4 view_proj;
    VertexBuffer vertex_buffer;
    uint instance_buffer;
    uint materials_buffer;
} pc;

void main() {
    Material material = materials[pc.materials_buffer].materials[nonuniformEXT(material_id)];
    vec4 color = texture(sampler2D(textures[nonuniformEXT(material.albedo_id)], samplers[1]), in_uv);
    out_frag_color = vec4(color.rgb, 1.);
    // out_frag_color = vec4(in_normal, 1.);
}
