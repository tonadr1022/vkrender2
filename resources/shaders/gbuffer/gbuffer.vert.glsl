#version 460

#extension GL_GOOGLE_include_directive : enable
#define BDA 1
#include "../common.h.glsl"
#include "../vertex_common.h.glsl"
#include "./gbuffer_common.h.glsl"

layout(location = 0) out vec3 out_normal;
layout(location = 1) out vec2 out_uv;
layout(location = 2) out vec3 out_frag_pos;
layout(location = 3) out vec3 out_bitangent;
layout(location = 4) out vec3 out_tangent;
layout(location = 5) flat out uint material_id;

#define IS_ANIMATED_BIT 1 << 0

struct InstanceData {
    uint material_id;
    uint instance_id;
    uint flags;
};

struct ObjectData {
    mat4 model;
    vec4 sphere_radius;
    vec4 extent;
};

layout(scalar, std430, buffer_reference) readonly buffer InstanceDatas {
    InstanceData datas[];
};

layout(std430, buffer_reference) readonly buffer ObjectDatas {
    ObjectData datas[];
};

void main() {
    InstanceData instance_data = InstanceDatas(instance_buffer).datas[gl_BaseInstance];
    Vertex v = Vertices(vtx).vertices[gl_VertexIndex];

    bool is_animated = (instance_data.flags & INSTANCE_IS_ANIMATED_BIT) != 0;
    mat4 model = ObjectDatas(object_data_buffer).datas[instance_data.instance_id].model;
    vec4 pos = vec4(v.pos, 1.);
    vec4 initial_normal = vec4(v.normal, 0.);
    vec4 initial_tangent = vec4(v.tangent.xyz, 0.);
    if (!is_animated) {
        pos = model * pos;
        initial_normal = model * initial_normal;
        initial_tangent = model * initial_tangent;
    }
    gl_Position = SceneDatas(scene_buffer).data.view_proj * pos;
    out_frag_pos = vec3(pos);

    out_normal = normalize(vec3(initial_normal));
    vec3 T = normalize(vec3(initial_tangent));
    vec3 N = out_normal;
    out_bitangent = cross(N, T);
    out_tangent = normalize(T - dot(N, T) * N);
    out_uv = vec2(v.uv_x, v.uv_y);
    material_id = instance_data.material_id;
}
