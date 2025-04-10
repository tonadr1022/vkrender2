#pragma once

#include <vk_mem_alloc.h>
#include <vulkan/vulkan_core.h>

#include <memory>
#include <span>

#include "Common.hpp"
#include "VkBootstrap.h"
#include "util/IndexAllocator.hpp"
#include "vk2/DeletionQueue.hpp"
#include "vk2/Texture.hpp"

namespace gfx {

template <typename T>
using RefPtr = std::shared_ptr<T>;

}  // namespace gfx

namespace gfx::vk2 {

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

  [[nodiscard]] VkCommandPool create_command_pool(
      u32 queue_idx,
      VkCommandPoolCreateFlags flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT) const;
  void create_command_buffers(VkCommandPool pool, std::span<VkCommandBuffer> buffers) const;
  [[nodiscard]] VkCommandBuffer create_command_buffer(VkCommandPool pool) const;
  [[nodiscard]] VkFence create_fence(VkFenceCreateFlags flags = VK_FENCE_CREATE_SIGNALED_BIT) const;
  [[nodiscard]] VkSemaphore create_semaphore(bool timeline = false) const;
  void destroy_fence(VkFence fence) const;
  void destroy_semaphore(VkSemaphore semaphore) const;
  void destroy_command_pool(VkCommandPool pool) const;
  void create_buffer(const VkBufferCreateInfo* info, const VmaAllocationCreateInfo* alloc_info,
                     VkBuffer& buffer, VmaAllocation& allocation,
                     VmaAllocationInfo& out_alloc_info);

  // void destroy_img(AllocatedImage& img);
  [[nodiscard]] VmaAllocator allocator() const { return allocator_; }

  [[nodiscard]] RefPtr<Image> create_image(const ImageCreateInfo& info);

 private:
  void init_impl(const CreateInfo& info);
  void destroy_impl();

  util::IndexAllocator image_index_allocator_;
  std::vector<std::unique_ptr<vk2::Image>> images_;

  // non owning
  VkSurfaceKHR surface_;
  VkDevice device_;

  // owning
  DeletionQueue main_del_queue_;
  vkb::PhysicalDevice vkb_phys_device_;
  vkb::Device vkb_device_;
  VmaAllocator allocator_;
};

Device& get_device();

}  // namespace gfx::vk2
