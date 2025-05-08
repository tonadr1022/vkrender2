#ifndef SKYBOX_COMMON_H
#define SKYBOX_COMMON_H

layout(push_constant) uniform SkyboxPC {
    uint scene_buffer;
    uint tex_idx;
};

#endif
