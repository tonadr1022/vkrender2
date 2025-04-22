#pragma once

#include <vk_mem_alloc.h>
#include <vulkan/vulkan_core.h>

#include <span>
#include <utility>

#include "Common.hpp"
#include "VkBootstrap.h"
#include "vk2/Buffer.hpp"
#include "vk2/DeletionQueue.hpp"
#include "vk2/Pool.hpp"
#include "vk2/Texture.hpp"

namespace gfx::vk2 {

class Device;  // forward declaration

template <typename T>
void destroy(T data);

Device& get_device();  // forward declaration

template <typename T>
struct Holder {
  Holder() = default;
  explicit Holder(Device* device, T&& data) : handle(std::move(data)), device_(device) {}

  Holder(const Holder& other) = delete;
  Holder& operator=(const Holder& other) = delete;

  Holder(Holder&& other) noexcept
      : handle(std::exchange(other.handle, T{})), device_(std::exchange(other.device_, nullptr)) {}

  Holder& operator=(Holder&& other) noexcept {
    if (&other == this) {
      return *this;
    }
    gfx::vk2::destroy(handle);
    handle = std::move(std::exchange(other.handle, T{}));
    device_ = std::exchange(other.device_, nullptr);
    return *this;
  }

  ~Holder() { gfx::vk2::destroy(handle); }

  T handle{};

 private:
  Device* device_{};
};

// struct ImageView2 {
//   ImageView2() = default;
//   ImageView2()
//   VkImageView view;
//   ImageViewCreateInfo create_info;
//   std::optional<BindlessResourceInfo> storage_image_resource_info;
//   std::optional<BindlessResourceInfo> sampled_image_resource_info;
// };

using ImageViewHandle = GenerationalHandle<class ::gfx::vk2::ImageView>;
// struct Image2 {
//   explicit Image2(const ImageCreateInfo& info);
//   Image2() = default;
//
//   Image2(Image2&&) noexcept;
//   Image2& operator=(Image2&&) noexcept;
//
//   Image2(const Image2&) = delete;
//   Image2& operator=(const Image2&) = delete;
//   ~Image2();
//   ImageCreateInfo create_info;
//   VkImage image{};
//   VkImageUsageFlags usage{};
//   Holder<ImageViewHandle> default_view;
//   VmaAllocation allocation{};
// };

using ImageHandle = GenerationalHandle<class ::gfx::vk2::Image>;
using BufferHandle = GenerationalHandle<class ::gfx::vk2::Buffer>;

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
  void destroy_resources();

  [[nodiscard]] VmaAllocator allocator() const { return allocator_; }

  Pool<ImageHandle, Image> img_pool_;
  Pool<ImageViewHandle, ImageView> img_view_pool_;
  Pool<BufferHandle, Buffer> buffer_pool_;

  BufferHandle create_buffer(const BufferCreateInfo& info);
  Holder<BufferHandle> create_buffer_holder(const BufferCreateInfo& info);
  ImageViewHandle create_image_view(const Image& image, const ImageViewCreateInfo& info);
  ImageHandle create_image(const ImageCreateInfo& info);
  Holder<ImageHandle> create_image_holder(const ImageCreateInfo& info);
  Holder<ImageViewHandle> create_image_view_holder(const Image& image,
                                                   const ImageViewCreateInfo& info);
  void destroy(ImageHandle handle);
  void destroy(ImageViewHandle handle);
  void destroy(BufferHandle handle);

  Image* get_image(ImageHandle handle) { return img_pool_.get(handle); }
  ImageView* get_image_view(ImageViewHandle handle) { return img_view_pool_.get(handle); }
  ImageView* get_image_view(const Holder<ImageViewHandle>& handle) {
    return img_view_pool_.get(handle.handle);
  }
  Buffer* get_buffer(BufferHandle handle) { return buffer_pool_.get(handle); }
  Buffer* get_buffer(const Holder<BufferHandle>& handle) { return get_buffer(handle.handle); }

 private:
  Image make_img_impl(const ImageCreateInfo& info);
  void init_impl(const CreateInfo& info);
  void destroy_impl();

  // non owning
  VkSurfaceKHR surface_;
  VkDevice device_;

  // owning
  DeletionQueue main_del_queue_;
  vkb::PhysicalDevice vkb_phys_device_;
  vkb::Device vkb_device_;
  VmaAllocator allocator_;
};

template <typename T>
void destroy(T data) {
  get_device().destroy(data);
}

}  // namespace gfx::vk2
