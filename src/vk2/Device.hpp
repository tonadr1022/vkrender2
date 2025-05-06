#pragma once

#include <vk_mem_alloc.h>
#include <vulkan/vulkan_core.h>

#include <mutex>
#include <span>

#include "Common.hpp"
#include "Types.hpp"
#include "VkBootstrap.h"
#include "vk2/Buffer.hpp"
#include "vk2/Pool.hpp"
#include "vk2/Swapchain.hpp"
#include "vk2/Texture.hpp"

#ifndef NDEBUG
#define VALIDATION_LAYERS_ENABLED 1
#define DEBUG_CALLBACK_ENABLED 1
#endif
template <>
void destroy(gfx::ImageHandle data);
template <>
void destroy(gfx::BufferHandle data);
template <>
void destroy(gfx::ImageViewHandle data);

struct GLFWwindow;
namespace gfx::vk2 {

enum class DeviceFeature : u8 { DrawIndirectCount };

struct CopyAllocator {
  struct CopyCmd {
    VkCommandPool transfer_cmd_pool{};
    VkCommandBuffer transfer_cmd_buf{};
    VkFence fence{};
    BufferHandle staging_buffer;
    [[nodiscard]] bool is_valid() const { return transfer_cmd_buf != VK_NULL_HANDLE; }
  };
  CopyCmd allocate(u64 size);
  void submit(CopyCmd cmd);
  void destroy();

 private:
  std::mutex free_list_mtx_;
  std::vector<CopyCmd> free_copy_cmds_;
};

enum class QueueType : u8 {
  Graphics,
  Compute,
  Transfer,
  Count,
};

class Device {
 public:
  Device(const Device&) = delete;
  Device(Device&&) = delete;
  Device& operator=(const Device&) = delete;
  Device& operator=(Device&&) = delete;

 private:
  Device() = default;
  ~Device();

 public:
  struct CreateInfo {
    const char* app_name;
    GLFWwindow* window;
    bool vsync{true};
  };
  struct Queue {
    VkQueue queue{};
    u32 family_idx{};
  };

  static void init(const CreateInfo& info);
  static Device& get();
  static void destroy();
  void on_imgui() const;
  void queue_submit(QueueType type, std::span<VkSubmitInfo2> submits);
  void queue_submit(QueueType type, std::span<VkSubmitInfo2> submits, VkFence fence);

  [[nodiscard]] VkDevice device() const {
    assert(device_);
    return device_;
  }
  [[nodiscard]] VkPhysicalDevice get_physical_device() const {
    return vkb_phys_device_.physical_device;
  }

  // VkFormat get_swapchain_format();

  VkImage acquire_next_image();
  // TODO: eradicate this
  [[nodiscard]] VkInstance get_instance() const { return instance_.instance; }
  [[nodiscard]] VkSurfaceKHR get_surface() const { return surface_; }

  // TODO: no resetting individual command buffers
  [[nodiscard]] VkCommandPool create_command_pool(
      QueueType type,
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

  // TODO: better args
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
  Image* get_image(const Holder<ImageHandle>& handle) { return img_pool_.get(handle.handle); }
  ImageView* get_image_view(ImageViewHandle handle) { return img_view_pool_.get(handle); }
  ImageView* get_image_view(const Holder<ImageViewHandle>& handle) {
    return img_view_pool_.get(handle.handle);
  }
  Buffer* get_buffer(BufferHandle handle) { return buffer_pool_.get(handle); }
  Buffer* get_buffer(const Holder<BufferHandle>& handle) { return get_buffer(handle.handle); }
  void init_imgui();

  [[nodiscard]] const Queue& get_queue(QueueType type) const { return queues_[(u32)type]; }

  [[nodiscard]] constexpr u32 get_frames_in_flight() const { return frames_in_flight; }
  [[nodiscard]] AttachmentInfo get_swapchain_info() const;
  [[nodiscard]] VkImage get_swapchain_img(u32 idx) const;
  void submit_to_graphics_queue();
  void begin_frame();
  [[nodiscard]] u32 curr_frame_num() const { return curr_frame_num_; }
  [[nodiscard]] u32 curr_frame_in_flight() const {
    return curr_frame_num() % get_frames_in_flight();
  }

  struct PerFrameData {
    VkSemaphore render_semaphore{};
    VkFence render_fence{};
  };
  PerFrameData& curr_frame() { return per_frame_data_[curr_frame_num_ % per_frame_data_.size()]; }

  VkSemaphore curr_swapchain_semaphore() {
    return swapchain_.acquire_semaphores[swapchain_.acquire_semaphore_idx];
  }

  VkFence allocate_fence(bool reset);
  void free_fence(VkFence fence);
  void wait_idle();
  [[nodiscard]] bool is_supported(DeviceFeature feature) const;

 private:
  VkPhysicalDeviceVulkan12Features supported_features12_{
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
  Queue queues_[(u32)QueueType::Count];
  Image make_img_impl(const ImageCreateInfo& info);
  void init_impl(const CreateInfo& info);

  std::vector<PerFrameData> per_frame_data_;
  std::vector<VkFence> free_fences_;
  VkSurfaceKHR surface_;
  VkDevice device_;
  Swapchain swapchain_;
  GLFWwindow* window_;
  vkb::Instance instance_;
  vkb::PhysicalDevice vkb_phys_device_;
  vkb::Device vkb_device_{};
  VmaAllocator allocator_;
  VkDescriptorPool imgui_descriptor_pool_;
  u32 curr_frame_num_{};
  bool resize_swapchain_req_{};
  static constexpr u32 frames_in_flight = 2;
};

Device& get_device();  // forward declaration

}  // namespace gfx::vk2
