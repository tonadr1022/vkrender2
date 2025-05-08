#ifndef VK2_RESOURCES
#define VK2_RESOURCES

#ifdef __cplusplus

#define VK2_DECLARE_ARGUMENTS(name) \
    struct name

#else  // GLSL

#define u32 uint
#define u64 uint64_t

#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_shader_image_load_formatted : require   // readable images without explicit format

#define BINDLESS_STORAGE_BUFFER_BINDING 1
#define BINDLESS_SAMPLER_BINDING 0
#define BINDLESS_SAMPLED_IMAGE_BINDING 2
#define BINDLESS_STORAGE_IMAGE_BINDING 0
#define BINDLESS_SAMPLER_SET 1

#define VK2_DECLARE_SAMPLED_IMAGES(type) \
  layout(set = 0, binding = BINDLESS_SAMPLED_IMAGE_BINDING) uniform type t_sampled_images_##type[]
#define VK2_DECLARE_STORAGE_IMAGES(type) \
  layout(set = 0, binding = BINDLESS_STORAGE_IMAGE_BINDING) uniform type t_storage_images_##type[]
#define VK2_DECLARE_STORAGE_IMAGES_WO(type) \
  layout(set = 0, binding = BINDLESS_STORAGE_IMAGE_BINDING) writeonly uniform type t_storage_images_##type[]
#define VK2_DECLARE_STORAGE_IMAGES_RO(type) \
  layout(set = 0, binding = BINDLESS_STORAGE_IMAGE_BINDING) readonly uniform type t_storage_images_##type[]

#define VK2_DECLARE_STORAGE_BUFFERS(blockname) \
  layout(set = 0, binding = BINDLESS_STORAGE_BUFFER_BINDING) buffer blockname
#define VK2_DECLARE_STORAGE_BUFFERS_WO(blockname) \
  layout(set = 0, binding = BINDLESS_STORAGE_BUFFER_BINDING) writeonly buffer blockname
#define VK2_DECLARE_STORAGE_BUFFERS_RO(blockname) \
  layout(set = 0, binding = BINDLESS_STORAGE_BUFFER_BINDING) restrict readonly buffer blockname
#define VK2_DECLARE_STORAGE_BUFFERS_RO_SCALAR(blockname) \
  layout(set = 0, binding = BINDLESS_STORAGE_BUFFER_BINDING, scalar) restrict readonly buffer blockname

#define vk2_get_sampler(idx) \
    s_samplers[nonuniformEXT(idx)]
#define vk2_get_sampled_img(type, idx) \
    t_sampled_images_##type[nonuniformEXT(idx)]

#define vk2_get_storage_img(type, idx) \
    t_storage_images_##type[nonuniformEXT(idx)]

#define vk2_sampler2D(tex_idx, sampler_idx) \
  nonuniformEXT(sampler2D(vk2_get_sampled_img(texture2D, tex_idx), vk2_get_sampler(sampler_idx)))

#define vk2_samplerCube(tex_idx, sampler_idx) \
  nonuniformEXT(samplerCube(vk2_get_sampled_img(textureCube, tex_idx), vk2_get_sampler(sampler_idx)))

#define vk2_sampler2DArray(tex_idx, sampler_idx) \
  sampler2DArray(vk2_get_sampled_img(texture2DArray, tex_idx), vk2_get_sampler(sampler_idx))

#define VK2_DECLARE_ARGUMENTS(name) \
    layout(push_constant, scalar) uniform name

layout(set = BINDLESS_SAMPLER_SET, binding = BINDLESS_SAMPLER_BINDING) uniform sampler s_samplers[];

#define LINEAR_SAMPLER_BINDLESS_IDX 2
#define NEAREST_SAMPLER_BINDLESS_IDX 2

#endif // end GLSL

#endif
