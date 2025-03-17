#include "./test.glsl"

#extension GL_EXT_shader_image_load_formatted : require
#extension GL_EXT_nonuniform_qualifier : require
layout(local_size_x = 16, local_size_y = 16) in;

layout(set = 0, binding = 0) uniform image2D images[];

layout(push_constant) uniform PushConstants {
    uint img_idx;
    float time;
};

void main() {
    ivec2 texel_coord = ivec2(gl_GlobalInvocationID.xy);
    ivec2 size = imageSize(images[img_idx]);
    if (texel_coord.x < size.x && texel_coord.y < size.y) {
        vec3 color = vec3(.1);
        color.x *= sin(time * 20.) * .5 + .2;
        imageStore(images[nonuniformEXT(img_idx)], texel_coord, vec4(color, 1.));
    }
}
