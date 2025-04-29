#version 460
#extension GL_GOOGLE_include_directive : enable

#define BDA 1
#include "./common.h.glsl"
#include "./vertex_common.h.glsl"
#include "./shadow_depth_common.h.glsl"

layout(location = 0) out vec2 out_uv;
layout(location = 1) flat out uint material_id;

struct InstanceData {
    uint material_id;
    uint instance_id;
};

struct ObjectData {
    mat4 model;
    vec4 sphere_radius;
    vec4 extent;
};

layout(std430, buffer_reference) readonly buffer InstanceDatas {
    InstanceData datas[];
};

layout(std430, buffer_reference) readonly buffer ObjectDatas {
    ObjectData datas[];
};

void main() {
    InstanceData instance_data = InstanceDatas(instance_buffer).datas[gl_InstanceIndex];
    Vertex v = Vertices(vtx).vertices[gl_VertexIndex];
    vec4 pos = ObjectDatas(object_data_buffer).datas[instance_data.instance_id].model * vec4(v.pos, 1.);
    gl_Position = vp_matrix * pos;
    out_uv = vec2(v.uv_x, v.uv_y);
    material_id = instance_data.material_id;
}
