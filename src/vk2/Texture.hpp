#pragma once

#include <vulkan/vulkan_core.h>

#include <optional>

#include "vk2/Resource.hpp"
#include "vk_mem_alloc.h"

namespace vk2 {

class TextureView {
 public:
  explicit TextureView(VkFormat format) {}

 private:
  VkImageView view_;
  std::optional<BindlessResourceInfo> storage_image_resource_info_;
  std::optional<BindlessResourceInfo> sampled_image_resource_info_;
};

// TODO: separate view class
// view class owns the bindless resource info.
// if the texture usage is layout general, only then should the storage resource info be allocated
class Texture {
 public:
  ~Texture();
  Texture& operator=(const Texture& other) = delete;
  Texture(const Texture& other) = delete;
  Texture(Texture&& other) noexcept;
  Texture& operator=(Texture&& other) noexcept;

  [[nodiscard]] VkExtent3D extent() const { return extent_; }
  [[nodiscard]] VkImageView view() const { return view_; }
  [[nodiscard]] VkImage image() const { return image_; }
  [[nodiscard]] VkFormat format() const { return format_; }

 private:
  friend class Device;
  friend class BindlessResourceAllocator;
  Texture();

  VkImage image_;
  VkImageView view_;
  VmaAllocation allocation_;
  VkExtent3D extent_;
  VkFormat format_;
  std::optional<BindlessResourceInfo> storage_image_resource_info_;
  std::optional<BindlessResourceInfo> sampled_image_resource_info_;
};

uint32_t get_mip_levels(VkExtent2D size);
}  // namespace vk2
