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
namespace gfx {
class Device;
namespace constants {
inline constexpr u32 remaining_array_layers = ~0U;
inline constexpr u32 remaining_mip_layers = ~0U;
}  // namespace constants

enum class DeviceFeature : u8 { DrawIndirectCount };

class Sampler {
 public:
  Sampler() = default;
  Sampler(const Sampler&) = default;
  Sampler(Sampler&&) = default;
  Sampler& operator=(const Sampler&) = default;
  Sampler& operator=(Sampler&&) = default;
  Sampler(VkSampler sampler, BindlessResourceInfo bindless_info)
      : sampler_(sampler), bindless_info_(bindless_info) {}

 private:
  VkSampler sampler_{};
  BindlessResourceInfo bindless_info_{};
  friend class Pool;
  friend class Device;
};

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
  Device* device_{};
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
  Pool<SamplerHandle, Sampler> sampler_pool_;
  SamplerHandle null_sampler_;
  SamplerHandle get_or_create_sampler(const SamplerCreateInfo& info);
  u32 get_bindless_idx(SamplerHandle sampler);
  u32 get_bindless_idx(ImageHandle img, SubresourceType type);
  // TODO: remove
  VkSampler get_sampler_vk(SamplerHandle sampler);

  // contains hash -> [handle, ref count]
  std::unordered_map<uint64_t, std::pair<SamplerHandle, u32>> sampler_cache_;

  // TODO: better args
  BufferHandle create_buffer(const BufferCreateInfo& info);
  Holder<BufferHandle> create_buffer_holder(const BufferCreateInfo& info);
  // returns subresource handle
  ImageViewHandle create_image_view(ImageHandle image_handle, u32 base_mip_level, u32 level_count,
                                    u32 base_array_layer, u32 layer_count);
  ImageHandle create_image(const ImageDesc& desc);
  Holder<ImageHandle> create_image_holder(const ImageDesc& desc);
  void destroy(ImageHandle handle);
  void destroy(SamplerHandle handle);
  void destroy(ImageViewHandle handle);
  void destroy(BufferHandle handle);
  void set_name(VkPipeline pipeline, const char* name);
  void set_name(ImageHandle handle, const char* name);
  void set_name(ImageViewHandle handle, const char* name);

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
  void set_name(const char* name, u64 handle, VkObjectType type);

  std::vector<PerFrameData> per_frame_data_;
  std::vector<VkFence> free_fences_;
  VkSurfaceKHR surface_;
  VkDevice device_;
  vk2::Swapchain swapchain_;
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

}  // namespace gfx
