#pragma once

#include <vulkan/vulkan_core.h>

#include <deque>

#include "Common.hpp"
#include "util/IndexAllocator.hpp"
#include "vk2/Buffer.hpp"
#include "vk2/Resource.hpp"
#include "vk2/Texture.hpp"

namespace gfx::vk2 {

class ResourceAllocator {
 public:
  void bind_desc_sets(VkCommandBuffer cmd);
  static constexpr u32 max_resource_descriptors{100'000};
  static constexpr u32 max_sampler_descriptors{128};

  static constexpr u32 bindless_storage_image_binding{0};
  static constexpr u32 bindless_storage_buffer_binding{1};
  static constexpr u32 bindless_sampled_image_binding{2};
  static constexpr u32 bindless_combined_image_sampler_binding{3};
  static constexpr u32 bindless_sampler_binding{0};

  u32 resource_to_binding(ResourceType type);

  static ResourceAllocator& get();
  static void init(VkDevice device, VmaAllocator allocator);
  static void shutdown();
  [[nodiscard]] VkDescriptorSetLayout main_set_layout() const { return main_set_layout_; }
  [[nodiscard]] VkDescriptorSet main_set() const { return main_set_; }
  void set_frame_num(u32 frame_num, u32 buffer_count);

  BindlessResourceInfo allocate_storage_buffer_descriptor(VkBuffer buffer);
  BindlessResourceInfo allocate_storage_img_descriptor(VkImageView view, VkImageLayout layout);
  BindlessResourceInfo allocate_sampled_img_descriptor(VkImageView view, VkImageLayout layout);
  BindlessResourceInfo allocate_sampler_descriptor(VkSampler sampler);
  void allocate_bindless_resource(VkDescriptorType descriptor_type, VkDescriptorImageInfo* img,
                                  VkDescriptorBufferInfo* buffer, u32 idx, u32 binding);

  void delete_texture(const TextureDeleteInfo& img);
  void delete_texture_view(const TextureViewDeleteInfo& info);
  void delete_buffer(const BufferDeleteInfo& info);
  void enqueue_delete_swapchain(VkSwapchainKHR swapchain);
  void enqueue_delete_pipeline(VkPipeline pipeline);

  void flush_deletions();

  VkDescriptorSetLayout main_set2_layout_{};
  VkDescriptorSet main_set2_{};

 private:
  template <typename T>
  struct DeleteQEntry {
    T data;
    u32 frame;
  };

  std::deque<DeleteQEntry<TextureDeleteInfo>> texture_delete_q_;
  std::deque<DeleteQEntry<TextureViewDeleteInfo>> texture_view_delete_q_;
  std::deque<DeleteQEntry<BufferDeleteInfo>> storage_buffer_delete_q_;
  std::deque<DeleteQEntry<VkSwapchainKHR>> swapchain_delete_q_;
  std::deque<DeleteQEntry<VkPipeline>> pipeline_delete_q_;

  ~ResourceAllocator();
  ResourceAllocator(VkDevice device, VmaAllocator allocator);

  VkDevice device_;
  VmaAllocator allocator_;
  util::IndexAllocator storage_image_allocator_{max_resource_descriptors};
  util::IndexAllocator storage_buffer_allocator_{max_resource_descriptors};
  util::IndexAllocator sampled_image_allocator_{max_resource_descriptors};
  util::IndexAllocator sampler_allocator_{max_sampler_descriptors};
  VkDescriptorPool main_pool_{};
  VkDescriptorSet main_set_{};
  VkDescriptorSetLayout main_set_layout_{};
  u32 buffer_count_{};
  u64 frame_num_{};
};
}  // namespace gfx::vk2
