#pragma once

#include <vk_mem_alloc.h>
#include <vulkan/vulkan_core.h>

namespace vk2 {

struct AllocatedImage {
  VkImage image;
  VkImageView view;
  VmaAllocation allocation;
  VkExtent3D extent;
  VkFormat format;
  void destroy();
};

struct UniqueImage {
  ~UniqueImage();
  UniqueImage& operator=(const UniqueImage& other) = delete;
  UniqueImage(const UniqueImage& other) = delete;
  UniqueImage(UniqueImage&& other) noexcept;
  UniqueImage& operator=(UniqueImage&& other) noexcept;

  VkImage image;
  VkImageView view;
  VmaAllocation allocation;
  VkExtent3D extent;
  VkFormat format;

 private:
  friend class Device;
  UniqueImage();
};

uint32_t get_mip_levels(VkExtent2D size);

}  // namespace vk2
