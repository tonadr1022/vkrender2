#version 460

#extension GL_GOOGLE_include_directive : enable

#include "../math.h.glsl"
#include "../common.h.glsl"

VK2_DECLARE_SAMPLED_IMAGES(texture2D);
VK2_DECLARE_STORAGE_IMAGES(image2D);

VK2_DECLARE_STORAGE_BUFFERS_RO(SsaoNoiseBuffers){
vec4 noise[];
} ssao_noises[];

VK2_DECLARE_STORAGE_BUFFERS_RO(SsaoKernelBuffers){
vec4 kernel[];
} ssao_kernels[];

layout(push_constant) uniform PC {
    uint scene_data_buf;
    uint depth_img;
    uint ssao_noise_buf;
    uint ssao_kernel_buf;
    uint output_img;
    uint gbuffer_a;
    float radius;
} pc;

layout(local_size_x = 16, local_size_y = 16) in;

void main() {
    ivec2 tex_coord = ivec2(gl_GlobalInvocationID);
    ivec2 img_size = imageSize(vk2_get_storage_img(image2D, pc.depth_img));
    if (any(greaterThanEqual(tex_coord, img_size))) {
        return;
    }
    SceneData scene_data = scene_data_buffer[pc.scene_data_buf].data;
    vec2 uv = (vec2(tex_coord) + .5) / vec2(img_size);
    float depth = imageLoad(vk2_get_storage_img(image2D, pc.depth_img), tex_coord).r;
    vec4 clip_pos = vec4(uv * 2. - 1., depth, 1.);
    vec4 wpos_pre_divide = scene_data.inverse_view_proj * clip_pos;
    vec3 world_pos = wpos_pre_divide.xyz / wpos_pre_divide.w;
    vec4 view_pos = scene_data.view * vec4(world_pos, 1.0);
    vec3 frag_view_pos = view_pos.xyz;
    vec4 gbuffer_a = imageLoad(vk2_get_storage_img(image2D, pc.gbuffer_a), tex_coord);
    vec3 world_normal = decode_oct(gbuffer_a.rg * 2. - 1.);
    // view space normal
    vec3 N = normalize(mat3(scene_data.view) * world_normal);

    // tiled noise tex
    ivec2 noise_tex_size = ivec2(4, 4);
    ivec2 noise_coord = tex_coord % noise_tex_size;
    int idx = noise_coord.x + noise_coord.y * noise_tex_size.x;
    // noise is vec3([-1,1],[-1,1],0)
    vec3 noise = ssao_noises[pc.ssao_noise_buf].noise[idx].xyz;

    vec3 tangent = normalize(noise - N * dot(noise, N));
    vec3 bitangent = normalize(cross(N, tangent));
    // tangent to view space mat
    mat3 TBN = mat3(tangent, bitangent, N);
    float occlusion = 0.0f;
    int kernel_size = 16;

    for (int i = 0; i < kernel_size; i++) {
        // ssao kernel in view space
        vec3 sample_pos = TBN * ssao_kernels[pc.ssao_kernel_buf].kernel[i].xyz;
        // sample pos in view space
        sample_pos = frag_view_pos + sample_pos * pc.radius;
        // clip space offset
        vec4 offset = scene_data.proj * vec4(sample_pos, 1.0);
        offset.xyz /= offset.w;
        offset.xy = offset.xy * .5 + .5;
        // if (offset.x < 0 || offset.x > 1.0 || offset.y < 0.0 || offset.y > 1.0) {
        //     continue;
        // }

        ivec2 sample_coord = ivec2(offset.xy * vec2(img_size));
        sample_coord = clamp(sample_coord, ivec2(0), img_size - 1);
        float sample_depth = imageLoad(vk2_get_storage_img(image2D, pc.depth_img), sample_coord).r;
        vec4 sample_clip_pos = vec4(offset.xy * 2.0 - 1.0, sample_depth, 1.0);

        vec4 sample_wpos_pre_divide = scene_data.inverse_view_proj * sample_clip_pos;
        vec3 sample_world_pos = sample_wpos_pre_divide.xyz / sample_wpos_pre_divide.w;
        vec4 sample_view_pos = scene_data.view * vec4(sample_world_pos, 1.0);
        float sample_view_z = sample_view_pos.z;

        // don't let samples out of radius contribute
        float range_check = smoothstep(0.0, 1.0, pc.radius / abs(frag_view_pos.z - sample_view_z));

        occlusion += step(sample_pos.z + 0.025, sample_view_z) * range_check;
    }
    occlusion = 1.0 - (occlusion / kernel_size);
    imageStore(vk2_get_storage_img(image2D, pc.output_img), tex_coord, vec4(occlusion, vec3(0.0)));
}
