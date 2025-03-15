#include "./test.glsl"

layout(local_size_x = 16, local_size_y = 16) in;

layout(rgba8, set = 0, binding = 0) uniform image2D image;

void main() {
    ivec2 texel_coord = ivec2(gl_GlobalInvocationID.xy);
    ivec2 size = imageSize(image);
    // test
    if (texel_coord.x < size.x && texel_coord.y < size.y) {
        imageStore(image, texel_coord, vec4(1.0) * a);
    }
}
