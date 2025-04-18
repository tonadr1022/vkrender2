#version 460

#extension GL_GOOGLE_include_directive : enable
#include "../common.h.glsl"
#include "../vertex_common.h.glsl"

VK2_DECLARE_ARGUMENTS(Basic3PushConstants){
uint scene_buffer;
uint vertex_buffer_idx;
uint instance_buffer;
uint object_data_buffer;
} ;

layout(location = 0) out vec3 out_normal;
// layout(location = 1) out vec2 out_uv;
// layout(location = 2) out vec3 out_frag_pos;
// layout(location = 3) out vec3 out_bitangent;
// layout(location = 4) out vec3 out_tangent;
// layout(location = 5) flat out uint material_id;

struct InstanceData {
    uint material_id;
    uint instance_id;
};

struct ObjectData {
    mat4 model;
    vec4 sphere_radius;
    vec4 extent;
};

VK2_DECLARE_STORAGE_BUFFERS_RO(ObjectDataBuffer){
InstanceData datas[];
} instance_buffers[];

VK2_DECLARE_STORAGE_BUFFERS_RO(InstanceDataBuffers){
ObjectData datas[];
} object_data_buffers[];

void main() {
    InstanceData instance_data = instance_buffers[instance_buffer].datas[gl_InstanceIndex];
    Vertex v = vertex_buffers[vertex_buffer_idx].vertices[gl_VertexIndex];
    mat4 model = object_data_buffers[object_data_buffer].datas[instance_data.instance_id].model;
    vec4 pos = model * vec4(v.pos, 1.);
    gl_Position = scene_data_buffer[scene_buffer].data.view_proj * pos;
    // out_frag_pos = vec3(pos);

    // TODO: something else lol
    // out_normal = normalize(transpose(inverse(mat3(model))) * v.normal);
    out_normal = normalize(vec3(model * vec4(v.normal, 0.)));
}
