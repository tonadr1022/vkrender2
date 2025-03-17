#pragma once

#include <vulkan/vulkan_core.h>

#include <functional>
#include <optional>

#include "vk2/Resource.hpp"
#include "vk_mem_alloc.h"

namespace vk2 {

// TODO: separate view class
// view class owns the bindless resource info.

// There are generally few enough samplers s.t. they don't need to be deleted.
// thus, they are immutable and can be trivially copied
class Sampler {
 public:
  explicit Sampler(const VkSamplerCreateInfo& info);

 private:
  BindlessResourceInfo resource_info_;
  VkSampler sampler_;
};

enum class TextureUsage : u8 {
  // general computation, i.e. StorageImage
  General,
  // asset textures read from shader
  ReadOnly,
  // attachment textures like gbuffer
  AttachmentReadOnly
};

struct TextureCreateInfo {
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
  const BindlessResourceInfo& storage_img_resource() {
    return storage_image_resource_info_.value();
  }
  const BindlessResourceInfo& sampled_img_resource() {
    return sampled_image_resource_info_.value();
  }

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

  [[nodiscard]] VkExtent3D extent() const { return create_info_.extent; }
  [[nodiscard]] VkImage image() const { return image_; }
  [[nodiscard]] VkFormat format() const { return create_info_.format; }

  [[nodiscard]] TextureView& view() { return view_.value(); }

 private:
  friend class Device;
  friend class BindlessResourceAllocator;
  friend class TextureView;

  TextureCreateInfo create_info_;
  std::optional<TextureView> view_;
  VkImage image_;
  VmaAllocation allocation_;
  VmaAllocator allocator_;
  VkDevice device_;
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

Texture create_texture_2d(VkFormat format, uvec3 dims, TextureUsage usage);

uint32_t get_mip_levels(VkExtent2D size);
VkImageType vkviewtype_to_img_type(VkImageViewType view_type);
bool format_is_stencil(VkFormat format);
bool format_is_depth(VkFormat format);
bool format_is_srgb(VkFormat format);
bool format_is_color(VkFormat format);

}  // namespace vk2
