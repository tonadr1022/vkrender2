#pragma once

#include <vk_mem_alloc.h>
#include <vulkan/vulkan_core.h>

#include "Types.hpp"
#include "vk2/Pool.hpp"
#include "vk2/Resource.hpp"

namespace gfx {

enum class ImageUsage : u8 {
  // general computation, i.e. StorageImage
  General,
  // asset textures read from shader
  ReadOnly,
  // attachment textures like gbuffer
  AttachmentReadOnly
};

struct ImageDesc {
  enum class Type : u8 { OneD, TwoD, ThreeD };
  Type type{Type::TwoD};
  Format format{Format::Undefined};
  uvec3 dims{};
  u32 mip_levels{1};
  u32 array_layers{1};
  u32 sample_count{1};
  BindFlag bind_flags{};
  ResourceMiscFlag misc_flags{};
  Usage usage{Usage::Default};
};

class Image;
class ImageView {
 public:
  ImageView() = default;
  [[nodiscard]] const BindlessResourceInfo& storage_img_resource() const {
    return storage_image_resource_info_;
  }
  [[nodiscard]] const BindlessResourceInfo& sampled_img_resource() const {
    return sampled_image_resource_info_;
  }
  [[nodiscard]] VkImageView view() const { return view_; }

 private:
  friend class Device;
  VkImageView view_{};
  BindlessResourceInfo storage_image_resource_info_;
  BindlessResourceInfo sampled_image_resource_info_;
};

class Image {
 public:
  Image() = default;
  [[nodiscard]] VkImage image() const { return image_; }
  [[nodiscard]] Format format() const { return desc_.format; }
  [[nodiscard]] uvec3 size() const { return desc_.dims; }
  [[nodiscard]] ImageViewHandle view() const { return view_.handle; }
  [[nodiscard]] const ImageDesc& get_desc() const { return desc_; }
  VkImageLayout curr_layout{};

 private:
  friend class Device;
  ImageDesc desc_;
  Holder<ImageViewHandle> view_;
  std::vector<ImageView> subresources_;
  VkImage image_{};
  VmaAllocation allocation_{};
};

void blit_img(VkCommandBuffer cmd, VkImage src, VkImage dst, VkExtent3D extent,
              VkImageAspectFlags aspect);

// TODO: device handle deletions
struct TextureDeleteInfo {
  VkImage img;
  VmaAllocation allocation;
};

uint32_t get_mip_levels(VkExtent2D size);
uint32_t get_mip_levels(uvec2 size);
VkImageType vkviewtype_to_img_type(VkImageViewType view_type);
bool format_is_stencil(Format format);
bool format_is_depth(Format format);
bool format_is_srgb(Format format);
bool format_is_color(Format format);

uint32_t format_storage_size(Format format);
bool format_is_block_compreesed(Format format);
u64 block_compressed_image_size(Format format, uvec3 extent);
u64 img_to_buffer_size(Format format, uvec3 extent);

}  // namespace gfx
