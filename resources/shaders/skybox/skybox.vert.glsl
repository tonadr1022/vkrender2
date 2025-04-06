#version 460

#extension GL_GOOGLE_include_directive : enable
#include "../common.h.glsl"
#include "./skybox_common.h.glsl"
// #include "../vertex_common.h.glsl"

layout(location = 0) out vec3 out_uv;

const vec3 skybox_vertices[36] = vec3[](
        vec3(-1.0f, 1.0f, -1.0f),
        vec3(-1.0f, -1.0f, -1.0f),
        vec3(1.0f, -1.0f, -1.0f),
        vec3(1.0f, -1.0f, -1.0f),
        vec3(1.0f, 1.0f, -1.0f),
        vec3(-1.0f, 1.0f, -1.0f),
        vec3(-1.0f, -1.0f, 1.0f),
        vec3(-1.0f, -1.0f, -1.0f),
        vec3(-1.0f, 1.0f, -1.0f),
        vec3(-1.0f, 1.0f, -1.0f),
        vec3(-1.0f, 1.0f, 1.0f),
        vec3(-1.0f, -1.0f, 1.0f),
        vec3(1.0f, -1.0f, -1.0f),
        vec3(1.0f, -1.0f, 1.0f),
        vec3(1.0f, 1.0f, 1.0f),
        vec3(1.0f, 1.0f, 1.0f),
        vec3(1.0f, 1.0f, -1.0f),
        vec3(1.0f, -1.0f, -1.0f),
        vec3(-1.0f, -1.0f, 1.0f),
        vec3(-1.0f, 1.0f, 1.0f),
        vec3(1.0f, 1.0f, 1.0f),
        vec3(1.0f, 1.0f, 1.0f),
        vec3(1.0f, -1.0f, 1.0f),
        vec3(-1.0f, -1.0f, 1.0f),
        vec3(-1.0f, 1.0f, -1.0f),
        vec3(1.0f, 1.0f, -1.0f),
        vec3(1.0f, 1.0f, 1.0f),
        vec3(1.0f, 1.0f, 1.0f),
        vec3(-1.0f, 1.0f, 1.0f),
        vec3(-1.0f, 1.0f, -1.0f),
        vec3(-1.0f, -1.0f, -1.0f),
        vec3(-1.0f, -1.0f, 1.0f),
        vec3(1.0f, -1.0f, -1.0f),
        vec3(1.0f, -1.0f, -1.0f),
        vec3(-1.0f, -1.0f, 1.0f),
        vec3(1.0f, -1.0f, 1.0)
    );

void main() {
    SceneData data = scene_data_buffer[scene_buffer].data;
    gl_Position = data.proj * mat4(mat3(data.view)) * vec4(skybox_vertices[gl_VertexIndex], 1.0);
    gl_Position.z = 0.;
    out_uv = skybox_vertices[gl_VertexIndex];
}
