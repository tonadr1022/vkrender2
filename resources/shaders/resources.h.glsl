#ifndef VK2_RESOURCES
#define VK2_RESOURCES

#ifdef __cplusplus

#define vmat4 glm::mat4
#define vmat3 glm::mat3
#define vvec3 glm::vec3
#define vu32 u32
#define vint i32

#define VK2_DECLARE_ARGUMENTS(name) \
    struct name

#else  // GLSL

#define vmat4 mat4
#define vmat3 mat3
#define vvec3 vec3
#define vu32 uint
#define vint int

#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : require

#define BINDLESS_STORAGE_BUFFER_BINDING 1
#define BINDLESS_SAMPLER_BINDING 0
#define BINDLESS_SAMPLED_IMAGE_BINDING 2

#define VK2_DECLARE_SAMPLED_IMAGES(type) \
  layout(set = 0, binding = BINDLESS_SAMPLED_IMAGE_BINDING) uniform type t_sampledImages_##type[]

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
    t_sampledImages_##type[nonuniformEXT(idx)]

#define vk2_sampler2D(tex_idx, sampler_idx) \
  nonuniformEXT(sampler2D(vk2_get_sampled_img(texture2D, tex_idx), vk2_get_sampler(sampler_idx)))

#define vk2_sampler2DArray(tex_idx, sampler_idx) \
  sampler2DArray(vk2_get_sampled_img(texture2DArray, tex_idx), vk2_get_sampler(sampler_idx))

#define VK2_DECLARE_ARGUMENTS(name) \
    layout(push_constant, scalar) uniform name

layout(set = 1, binding = BINDLESS_SAMPLER_BINDING) uniform sampler s_samplers[];

#endif // end GLSL

#endif
