#pragma once

#include <vk_mem_alloc.h>
#include <vulkan/vulkan_core.h>

#include <deque>
#include <memory>
#include <mutex>
#include <span>

#include "Common.hpp"
#include "Types.hpp"
#include "VkBootstrap.h"
#include "core/FixedVector.hpp"
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

struct GLFWwindow;
namespace gfx {
class Device;
struct CmdEncoder;

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

class Device {
 public:
  Device(const Device&) = delete;
  Device(Device&&) = delete;
  Device& operator=(const Device&) = delete;
  Device& operator=(Device&&) = delete;

 private:
  Device();
  ~Device();

 public:
  struct CreateInfo {
    const char* app_name;
    GLFWwindow* window;
    bool vsync{true};
  };

  static void init(const CreateInfo& info);
  static Device& get();
  static void destroy();
  void on_imgui() const;

  [[nodiscard]] VkDevice device() const {
    assert(device_);
    return device_;
  }
  [[nodiscard]] VkPhysicalDevice get_physical_device() const {
    return vkb_phys_device_.physical_device;
  }

  // VkFormat get_swapchain_format();

  VkImage acquire_next_image(CmdEncoder* cmd);
  // TODO: eradicate this
  [[nodiscard]] VkInstance get_instance() const { return instance_.instance; }
  [[nodiscard]] VkSurfaceKHR get_surface() const { return surface_; }

  // TODO: no resetting individual command buffers
  [[nodiscard]] VkCommandPool create_command_pool(
      QueueType type,
      VkCommandPoolCreateFlags flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
      const char* name = "cmd pool") const;
  void create_command_buffers(VkCommandPool pool, std::span<VkCommandBuffer> buffers) const;
  [[nodiscard]] VkCommandBuffer create_command_buffer(VkCommandPool pool) const;
  [[nodiscard]] VkFence create_fence(VkFenceCreateFlags flags = VK_FENCE_CREATE_SIGNALED_BIT) const;
  [[nodiscard]] VkSemaphore create_semaphore(bool timeline = false,
                                             const char* name = "semaphore") const;
  void destroy_fence(VkFence fence) const;
  void destroy_semaphore(VkSemaphore semaphore) const;
  void destroy_command_pool(VkCommandPool pool) const;

  [[nodiscard]] VmaAllocator allocator() const { return allocator_; }

  Pool<ImageHandle, Image> img_pool_;
  Pool<BufferHandle, Buffer> buffer_pool_;
  Pool<SamplerHandle, Sampler> sampler_pool_;
  SamplerHandle null_sampler_;
  SamplerHandle get_or_create_sampler(const SamplerCreateInfo& info);
  u32 get_bindless_idx(SamplerHandle sampler);
  u32 get_bindless_idx(ImageHandle img, SubresourceType type, int subresource = -1);
  // TODO: remove
  VkImageView get_image_view(ImageHandle img, SubresourceType type, int subresource = -1);
  u32 get_bindless_idx(const Holder<ImageHandle>& img, SubresourceType type, int subresource = -1);

  u32 get_bindless_idx(BufferHandle buffer);
  u32 get_bindless_idx(const Holder<BufferHandle>& buffer) {
    return get_bindless_idx(buffer.handle);
  }

  // contains hash -> [handle, ref count]
  std::unordered_map<uint64_t, std::pair<SamplerHandle, u32>> sampler_cache_;

  // TODO: better args
  BufferHandle create_buffer(const BufferCreateInfo& info);
  BufferHandle create_staging_buffer(u64 size);
  Holder<BufferHandle> create_buffer_holder(const BufferCreateInfo& info);
  // returns subresource handle
  i32 create_subresource(ImageHandle image_handle, u32 base_mip_level, u32 level_count,
                         u32 base_array_layer, u32 layer_count);
  ImageView2 create_image_view2(ImageHandle image_handle, SubresourceType type, u32 base_mip_level,
                                u32 level_count, u32 base_array_layer, u32 layer_count);
  ImageHandle create_image(const ImageDesc& desc, void* initial_data = nullptr);
  Holder<ImageHandle> create_image_holder(const ImageDesc& desc, void* initial_data = nullptr);
  void destroy(ImageHandle handle);
  void destroy(SamplerHandle handle);
  void destroy(BufferHandle handle);
  void set_name(VkPipeline pipeline, const char* name);
  void set_name(VkFence fence, const char* name);
  void set_name(ImageHandle handle, const char* name);
  void set_name(VkSemaphore semaphore, const char* name) const;
  void set_name(VkCommandPool pool, const char* name) const;
  Image* get_image(ImageHandle handle) { return img_pool_.get(handle); }
  Image* get_image(const Holder<ImageHandle>& handle) { return img_pool_.get(handle.handle); }
  Buffer* get_buffer(BufferHandle handle) { return buffer_pool_.get(handle); }
  Buffer* get_buffer(const Holder<BufferHandle>& handle) { return get_buffer(handle.handle); }
  void init_imgui();
  void render_imgui(CmdEncoder& cmd);
  void new_imgui_frame();

  [[nodiscard]] constexpr u32 get_frames_in_flight() const { return frames_in_flight; }
  [[nodiscard]] AttachmentInfo get_swapchain_info() const;
  [[nodiscard]] VkImage get_swapchain_img(u32 idx) const;
  void submit_commands();
  void begin_frame();
  [[nodiscard]] u32 curr_frame_num() const { return curr_frame_num_; }
  [[nodiscard]] u32 curr_frame_in_flight() const {
    return curr_frame_num() % get_frames_in_flight();
  }

  VkSemaphore curr_swapchain_semaphore() {
    return swapchain_.acquire_semaphores[swapchain_.acquire_semaphore_idx];
  }

  VkFence allocate_fence(bool reset);
  void free_fence(VkFence fence);
  void wait_idle();
  [[nodiscard]] bool is_supported(DeviceFeature feature) const;
  void cmd_list_wait(CmdEncoder* cmd_list, CmdEncoder* wait_for);

 private:
  VkSemaphore new_semaphore();
  void free_semaphore(VkSemaphore semaphore);
  void free_semaphore_unsafe(VkSemaphore semaphore);
  std::mutex semaphore_pool_mtx_;
  std::vector<VkSemaphore> free_semaphores_;

 public:
  struct Queue {
    VkQueue queue{};
    u32 family_idx{UINT32_MAX};
    VkSemaphore frame_semaphores[frames_in_flight][(u32)QueueType::Count]{};
    std::vector<VkSemaphoreSubmitInfo> signal_semaphore_infos;
    std::vector<VkSemaphore> signal_semaphores;
    std::vector<VkSemaphoreSubmitInfo> wait_semaphores_infos;
    std::vector<VkCommandBufferSubmitInfo> submit_cmds;

    std::vector<vk2::Swapchain*> swapchain_updates;
    std::vector<VkSwapchainKHR> submit_swapchains;
    std::vector<u32> submit_swapchain_img_indices;
    std::mutex mtx_;

    void clear();
    void submit(Device* device, VkFence fence);
    void wait(VkSemaphore semaphore);
    void signal(VkSemaphore semaphore);
    void submit(u32 submit_count, const VkSubmitInfo2* submits, VkFence fence);
  };

  [[nodiscard]] const Queue& get_queue(QueueType type) const { return queues_[(u32)type]; }
  [[nodiscard]] Queue& get_queue(QueueType type) { return queues_[(u32)type]; }
  u32 cmd_buf_count_{};
  CmdEncoder* begin_command_list(QueueType queue_type);
  void begin_swapchain_blit(CmdEncoder* cmd);
  void blit_to_swapchain(CmdEncoder* cmd, const Image& img, uvec2 dims);

 private:
  struct TransitionHandler {
    VkCommandPool cmd_pool{};
    VkCommandBuffer cmd_buf{};
    VkSemaphore semaphores[(u32)QueueType::Count]{};
  };

  // smart ptr to handle resizing
  std::vector<std::unique_ptr<CmdEncoder>> cmd_lists_;
  TransitionHandler transition_handlers_[frames_in_flight];

  // TODO: fix
 public:
  std::vector<VkImageMemoryBarrier2> init_transitions_;
  struct CopyAllocator {
    explicit CopyAllocator(Device* device, QueueType type) : device_(device), type_(type) {}
    struct CopyCmd {
      VkCommandPool transfer_cmd_pool{};
      VkCommandBuffer transfer_cmd_buf{};
      VkFence fence{};
      BufferHandle staging_buffer;
      [[nodiscard]] bool is_valid() const { return transfer_cmd_buf != VK_NULL_HANDLE; }
      void copy_buffer(Device* device, const Buffer& dst, u64 src_offset, u64 dst_offset,
                       u64 size) const;
    };
    CopyCmd allocate(u64 size);
    CmdEncoder* allocate2(u64 size);
    void submit(CopyCmd cmd);
    void destroy();

   private:
    Device* device_{};
    QueueType type_{QueueType::Count};
    std::mutex free_list_mtx_;
    std::vector<CopyCmd> free_copy_cmds_;
    std::vector<BufferHandle> free_staging_buffers2_;
  };

 private:
  VkFence frame_fences_[frames_in_flight][(u32)QueueType::Count]{};

  VkPhysicalDeviceVulkan12Features supported_features12_{
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
  Queue queues_[(u32)QueueType::Count];
  util::fixed_vector<u32, (u32)QueueType::Count> queue_family_indices_;
  void init_impl(const CreateInfo& info);
  void set_name(const char* name, u64 handle, VkObjectType type) const;

 public:
  CopyAllocator graphics_copy_allocator_;
  CopyAllocator transfer_copy_allocator_;

 private:
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

  void destroy(Image& img);

 public:
  static constexpr u32 max_resource_descriptors{100'000};
  static constexpr u32 max_sampler_descriptors{128};
  static constexpr u32 bindless_storage_image_binding{0};
  static constexpr u32 bindless_storage_buffer_binding{1};
  static constexpr u32 bindless_sampled_image_binding{2};
  static constexpr u32 bindless_combined_image_sampler_binding{3};
  static constexpr u32 bindless_sampler_binding{0};

  static void init(VkDevice device, VmaAllocator allocator);
  static void shutdown();
  [[nodiscard]] VkDescriptorSetLayout main_set_layout() const { return main_set_layout_; }
  [[nodiscard]] VkDescriptorSet main_set() const { return main_set_; }
  BindlessResourceInfo allocate_storage_buffer_descriptor(VkBuffer buffer);
  BindlessResourceInfo allocate_storage_img_descriptor(VkImageView view, VkImageLayout layout);
  BindlessResourceInfo allocate_sampled_img_descriptor(VkImageView view, VkImageLayout layout);
  BindlessResourceInfo allocate_sampler_descriptor(VkSampler sampler);
  void allocate_bindless_resource(VkDescriptorType descriptor_type, VkDescriptorImageInfo* img,
                                  VkDescriptorBufferInfo* buffer, u32 idx, u32 binding);

  // TODO: remove these
  void enqueue_delete_texture_view(VkImageView view);
  void enqueue_delete_swapchain(VkSwapchainKHR swapchain);
  void enqueue_delete_pipeline(VkPipeline pipeline);
  void enqueue_delete_sempahore(VkSemaphore semaphore);

  void flush_deletions();

 private:
  static constexpr u64 timeout_value = 2000000000ull;  // 2 sec

  u32 resource_to_binding(ResourceType type);
  void init_bindless();
  template <typename T>
  struct DeleteQEntry {
    T data;
    u32 frame;
  };

  struct TextureDeleteInfo {
    VkImage img;
    VmaAllocation allocation;
  };
  void delete_texture(const TextureDeleteInfo& img);
  std::deque<DeleteQEntry<TextureDeleteInfo>> texture_delete_q_;
  std::deque<DeleteQEntry<ImageView2>> texture_view_delete_q3_;
  std::deque<DeleteQEntry<VkImageView>> texture_view_delete_q2_;
  std::deque<DeleteQEntry<BufferHandle>> storage_buffer_delete_q_;
  std::deque<DeleteQEntry<VkSwapchainKHR>> swapchain_delete_q_;
  std::deque<DeleteQEntry<VkSemaphore>> semaphore_delete_q_;
  std::deque<DeleteQEntry<VkPipeline>> pipeline_delete_q_;

  struct IndexAllocator {
    explicit IndexAllocator(u32 size = 64);
    void free(u32 idx);
    [[nodiscard]] u32 alloc();

   private:
    std::vector<u32> free_list_;
    u32 next_index_{};
  };

  IndexAllocator storage_image_allocator_{max_resource_descriptors};
  IndexAllocator storage_buffer_allocator_{max_resource_descriptors};
  IndexAllocator sampled_image_allocator_{max_resource_descriptors};
  IndexAllocator sampler_allocator_{max_sampler_descriptors};

  VkDescriptorPool main_pool_{};
  VkDescriptorSet main_set_{};
  VkDescriptorSetLayout main_set_layout_{};
  VkDescriptorSetLayout main_set2_layout_{};
  VkDescriptorSet main_set2_{};
  // TODO: fix
 public:
  VkPipelineLayout default_pipeline_layout_{};
  VkSampler get_sampler_vk(SamplerHandle sampler);
  void bind_bindless_descriptors(CmdEncoder& cmd);
};

Device& get_device();  // forward declaration

}  // namespace gfx
