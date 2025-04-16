#pragma once

#include <vk_mem_alloc.h>
#include <vulkan/vulkan_core.h>

#include <memory>
#include <span>

#include "Common.hpp"
#include "VkBootstrap.h"
#include "vk2/DeletionQueue.hpp"
#include "vk2/Texture.hpp"

namespace gfx {

template <typename T>
using RefPtr = std::shared_ptr<T>;

}  // namespace gfx

namespace gfx::vk2 {

template <typename HandleT>
struct Handle {
 private:
  friend class ObjectPool;
  uint32_t idx_{};
  uint32_t gen_{};
};

struct ImageView2 {
  VkImageView view_;
  ImageViewCreateInfo create_info_;
  std::optional<BindlessResourceInfo> storage_image_resource_info_;
  std::optional<BindlessResourceInfo> sampled_image_resource_info_;
};

using ImageViewHandle = Handle<struct ::gfx::vk2::ImageView2>;

struct Image2 {
  explicit Image2(const ImageCreateInfo& info);
  Image2() = default;
  ImageCreateInfo create_info;
  VkImage image{};
  ImageViewHandle default_view{};
  VkImageLayout curr_layout{VK_IMAGE_LAYOUT_UNDEFINED};
  VmaAllocation allocation{};
};
using ImageHandle = Handle<struct ::gfx::vk2::Image2>;

// ObjectT should be default constructible and have sane default constructed state
template <typename HandleT, typename ObjectT>
struct Pool {
  using IndexT = uint32_t;
  Pool() : entries_(10) {}
  explicit Pool(IndexT size) : entries_(size) {}

  struct Entry {
    explicit Entry(ObjectT&& obj) : object(obj) {}
    ObjectT object{};
    uint32_t gen_{1};
  };

  // TODO: support perfect forwarding
  HandleT alloc(const ObjectT& obj) {
    IndexT idx;
    if (free_list_.empty()) {
      idx = free_list_.back();
      free_list_.pop_back();
    } else {
      free_list_.emplace_back();
    }
    entries_[idx] = Entry{std::move(obj)};
    size_++;
  }
  HandleT alloc(ObjectT&& obj) {
    IndexT idx;
    if (free_list_.empty()) {
      idx = free_list_.back();
      free_list_.pop_back();
    } else {
      free_list_.emplace_back();
    }
    entries_[idx] = Entry{std::move(obj)};
    size_++;
  }

  [[nodiscard]] IndexT size() const { return size_; }

  void destroy(HandleT handle) {
    assert(handle.idx_ < entries_.size());
    if (handle.idx_ >= entries_.size()) {
      return;
    }
    assert(handle.gen_ == entries_[handle.idx_].gen_);
    if (entries_[handle.idx_].gen_ != handle.gen_) {
      return;
    }
    entries_[handle.idx_].gen_++;
    entries_[handle.idx_].object = {};
    free_list_.emplace_back(handle.idx_);
    size_--;
  }

  ObjectT* get(HandleT handle) {
    assert(handle.idx_ < entries_.size());
    if (handle.idx_ >= entries_.size()) {
      return nullptr;
    }
    assert(handle.gen_ == entries_[handle.idx_].gen_);
    if (entries_[handle.idx_].gen_ != handle.gen_) {
      return nullptr;
    }
    return &entries_[handle.idx].object;
  }

 private:
  std::vector<IndexT> free_list_;
  std::vector<Entry> entries_;
  IndexT size_{};
};

// template<typename T>
// struct Holder {
// private:
//   T data_;
// };
//
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

  [[nodiscard]] VmaAllocator allocator() const { return allocator_; }

  Pool<ImageHandle, Image2> img_pool_;
  Pool<ImageViewHandle, ImageView2> img_view_pool_;
  ImageViewHandle create_img_view(ImageHandle texture, const ImageViewCreateInfo& info);
  ImageHandle create_img(const ImageCreateInfo& info);

 private:
  Image2 make_img_impl(const ImageCreateInfo& info);
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

Device& get_device();

}  // namespace gfx::vk2
