#pragma once

#include <vk_mem_alloc.h>
#include <vulkan/vulkan_core.h>

#include <span>

#include "Common.hpp"
#include "VkBootstrap.h"
#include "vk2/DeletionQueue.hpp"

namespace vk2 {

constexpr VkImageSubresourceRange default_img_subresource_range{
    .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
    .baseMipLevel = 0,
    .levelCount = 1,
    .baseArrayLayer = 0,
    .layerCount = 1,
};
struct ImageViewCreateInfo {
  VkImage image;
  VkImageViewType view_type;
  VkFormat format;
  VkImageSubresourceRange subresource_range{default_img_subresource_range};
};
class Device {
 public:
  struct CreateInfo {
    vkb::Instance instance;
    VkSurfaceKHR surface;
  };
  static void init(const CreateInfo& info);
  static Device& get();
  static void destroy();

  [[nodiscard]] VkDevice device() const { return vkb_device_.device; }
  [[nodiscard]] VkPhysicalDevice phys_device() const { return vkb_phys_device_.physical_device; }
  [[nodiscard]] const vkb::Device& vkb_device() const { return vkb_device_; }

  VkFormat get_swapchain_format();
  [[nodiscard]] VkImageView create_image_view(const ImageViewCreateInfo& info) const;

  [[nodiscard]] VkCommandPool create_command_pool(
      u32 queue_idx,
      VkCommandPoolCreateFlags flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT) const;
  void create_command_buffers(VkCommandPool pool, std::span<VkCommandBuffer> buffers) const;
  [[nodiscard]] VkCommandBuffer create_command_buffer(VkCommandPool pool) const;
  [[nodiscard]] VkFence create_fence(VkFenceCreateFlags flags = VK_FENCE_CREATE_SIGNALED_BIT) const;
  [[nodiscard]] VkSemaphore create_semaphore() const;
  void destroy_fence(VkFence fence) const;
  void destroy_semaphore(VkSemaphore semaphore) const;
  void destroy_command_pool(VkCommandPool pool) const;

 private:
  void init_impl(const CreateInfo& info);
  void destroy_impl();

  // non owning
  VkSurfaceKHR surface_;

  // owning
  DeletionQueue main_del_queue_;
  vkb::PhysicalDevice vkb_phys_device_;
  vkb::Device vkb_device_;
  VmaAllocator allocator_;
};

Device& device();

}  // namespace vk2
