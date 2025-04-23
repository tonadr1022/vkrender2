#version 460

#extension GL_GOOGLE_include_directive : enable
#include "../resources.h.glsl"
#include "../math.h.glsl"

layout(local_size_x = 16, local_size_y = 16) in;

VK2_DECLARE_STORAGE_IMAGES(image2D);
VK2_DECLARE_SAMPLED_IMAGES(texture2D);

layout(push_constant) uniform PC {
    mat4 inv_view_proj;
    uint gbuffer_a_tex;
    uint gbuffer_b_tex;
    uint gbuffer_c_tex;
    uint depth_img;
    uint output_tex;
    uint sampler_idx;
};

void main() {
    ivec2 tex_coord = ivec2(gl_GlobalInvocationID);
    ivec2 img_size = imageSize(vk2_get_storage_img(image2D, gbuffer_a_tex));
    if (tex_coord.x >= img_size.x || tex_coord.y >= img_size.y) {
        return;
    }

    vec2 uv = vec2(tex_coord) / vec2(img_size);
    float depth = texture(vk2_sampler2D(depth_img, sampler_idx), uv).r;
    vec4 clip_pos = vec4(uv * 2. - 1., depth, 1.);
    vec4 wpos_pre_divide = inv_view_proj * clip_pos;
    vec3 world_pos = wpos_pre_divide.xyz / wpos_pre_divide.w;

    vec4 gbuffer_a = imageLoad(vk2_get_storage_img(image2D, gbuffer_a_tex), tex_coord);
    vec4 gbuffer_b = imageLoad(vk2_get_storage_img(image2D, gbuffer_b_tex), tex_coord);
    vec4 gbuffer_c = imageLoad(vk2_get_storage_img(image2D, gbuffer_c_tex), tex_coord);
    vec3 normal = decode_oct(gbuffer_a.rg * 2. - 1.);
    float metallic = gbuffer_a.b;
    float roughness = gbuffer_a.a;
    vec4 albedo = gbuffer_b;
    if (albedo.a == 0) {
        imageStore(vk2_get_storage_img(image2D, output_tex), tex_coord, vec4(0.));
        return;
    }
    vec3 emissive = gbuffer_c.rgb;
    float ao = gbuffer_c.a;
    // view pos

    imageStore(vk2_get_storage_img(image2D, output_tex), tex_coord, vec4(vec3(depth), 1.));
    // imageStore(vk2_get_storage_img(image2D, output_tex), tex_coord, vec4(world_pos, 1.));
    // imageStore(vk2_get_storage_img(image2D, output_tex), tex_coord, vec4(normal * .5 + .5, 1.));
}
