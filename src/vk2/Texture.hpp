#pragma once

#include <vulkan/vulkan_core.h>

#include <array>
#include <optional>
#include <string>

#include "vk2/Resource.hpp"
#include "vk_mem_alloc.h"

namespace vk2 {

enum class TextureUsage : u8 {
  // general computation, i.e. StorageImage
  General,
  // asset textures read from shader
  ReadOnly,
  // attachment textures like gbuffer
  AttachmentReadOnly
};

struct TextureCreateInfo {
  std::string name;
  VkImageViewType view_type{};
  VkFormat format{};
  VkExtent3D extent{};
  u32 mip_levels{1};
  u32 array_layers{1};
  VkSampleCountFlagBits samples{VK_SAMPLE_COUNT_1_BIT};
  TextureUsage usage{TextureUsage::General};
};

struct TextureViewCreateInfo {
  VkFormat format;
  VkImageSubresourceRange range;
  VkComponentMapping components{};
  VkImageViewType view_type{VK_IMAGE_VIEW_TYPE_MAX_ENUM};
};

class Texture;
class TextureView {
 public:
  explicit TextureView(const Texture& texture, const TextureViewCreateInfo& info);
  ~TextureView();
  TextureView(TextureView&& other) noexcept;
  TextureView& operator=(TextureView&& other) noexcept;
  TextureView(const TextureView&) = delete;
  TextureView& operator=(const TextureView&) = delete;

  // TODO: use pointers or optionals?
  [[nodiscard]] const BindlessResourceInfo& storage_img_resource() const {
    return storage_image_resource_info_.value();
  }
  [[nodiscard]] const BindlessResourceInfo& sampled_img_resource() const {
    return sampled_image_resource_info_.value();
  }
  [[nodiscard]] VkImageView view() const { return view_; }

 private:
  VkImageView view_;
  TextureViewCreateInfo create_info_;
  // TODO: make a bindless texture view class for this
  std::optional<BindlessResourceInfo> storage_image_resource_info_;
  std::optional<BindlessResourceInfo> sampled_image_resource_info_;
};

class Texture {
 public:
  explicit Texture(const TextureCreateInfo& create_info);
  ~Texture();
  Texture& operator=(const Texture& other) = delete;
  Texture(const Texture& other) = delete;
  Texture(Texture&& other) noexcept;
  Texture& operator=(Texture&& other) noexcept;

  [[nodiscard]] VkExtent2D extent_2d() const {
    return {create_info_.extent.width, create_info_.extent.height};
  }
  [[nodiscard]] VkExtent3D extent() const { return create_info_.extent; }
  [[nodiscard]] VkImage image() const { return image_; }
  [[nodiscard]] VkFormat format() const { return create_info_.format; }

  [[nodiscard]] TextureView& view() { return view_.value(); }
  [[nodiscard]] const TextureView& view() const { return view_.value(); }
  [[nodiscard]] const TextureCreateInfo& create_info() const { return create_info_; }

  VkImageLayout curr_layout{VK_IMAGE_LAYOUT_UNDEFINED};

 private:
  friend class Device;
  friend class BindlessResourceAllocator;
  friend class TextureView;

  TextureCreateInfo create_info_;
  std::optional<TextureView> view_;
  std::string name_;
  VkImage image_;
  VmaAllocation allocation_;
  VmaAllocator allocator_;
  VkDevice device_;
};

struct TextureCubeAndViews {
  explicit TextureCubeAndViews(const TextureCreateInfo& info);
  std::optional<vk2::Texture> texture;
  std::array<std::optional<vk2::TextureView>, 6> img_views;
};

void blit_img(VkCommandBuffer cmd, VkImage src, VkImage dst, VkExtent3D extent,
              VkImageAspectFlags aspect);

struct TextureDeleteInfo {
  VkImage img;
  VmaAllocation allocation;
};

struct TextureViewDeleteInfo {
  std::optional<BindlessResourceInfo> storage_image_resource_info;
  std::optional<BindlessResourceInfo> sampled_image_resource_info;
  VkImageView view;
};

Texture create_texture_2d(VkFormat format, uvec3 dims, TextureUsage usage, std::string name = {});
Texture create_texture_2d_mip(VkFormat format, uvec3 dims, TextureUsage usage, u32 levels);

uint32_t get_mip_levels(VkExtent2D size);
uint32_t get_mip_levels(uvec2 size);
VkImageType vkviewtype_to_img_type(VkImageViewType view_type);
bool format_is_stencil(VkFormat format);
bool format_is_depth(VkFormat format);
bool format_is_srgb(VkFormat format);
bool format_is_color(VkFormat format);

uint32_t format_storage_size(VkFormat format);
bool format_is_block_compreesed(VkFormat format);
u64 block_compressed_image_size(VkFormat format, uvec3 extent);
u64 img_to_buffer_size(VkFormat format, uvec3 extent);

}  // namespace vk2
