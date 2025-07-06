#ifndef GEOMETRY_COMMON_H
#define GEOMETRY_COMMON_H

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

InstanceData get_instance_data(u64 handle) {
    return InstanceDatas(handle).datas[gl_BaseInstance];
}

#endif
