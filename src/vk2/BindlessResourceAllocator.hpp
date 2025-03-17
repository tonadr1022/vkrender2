#pragma once

#include <vulkan/vulkan_core.h>

#include <deque>
#include <vector>

#include "Common.hpp"
#include "vk2/Buffer.hpp"
#include "vk2/Resource.hpp"
#include "vk2/Texture.hpp"

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

struct IndexAllocator {
  explicit IndexAllocator(u32 size);
  [[nodiscard]] u32 alloc();
  void free(u32 idx);

 private:
  std::vector<u32> free_list_;
};
// template <typename T>
// struct ResourceDeleteQueue {
//   ResourceDeleteQueue(const ResourceDeleteQueue&) = delete;
//   ResourceDeleteQueue(ResourceDeleteQueue&&) = delete;
//   ResourceDeleteQueue& operator=(const ResourceDeleteQueue&) = delete;
//   ResourceDeleteQueue& operator=(ResourceDeleteQueue&&) = delete;
//
//   void push(T&& data, u32 frame) { queue.push_back(Entry{data, frame}); }
//
//   void flush(u32 frame) {
//     std::erase_if(queue, [frame](const Entry& e) { return static_cast<bool>(e.frame < frame); });
//   }
//   std::deque<Entry> queue;
// };

class BindlessResourceAllocator {
 public:
  static constexpr u32 max_resource_descriptors{100'000};
  static constexpr u32 max_sampler_descriptors{1000};

  static constexpr u32 bindless_storage_image_binding{0};
  static constexpr u32 bindless_storage_buffer_binding{1};
  static constexpr u32 bindless_sampled_image_binding{2};
  static constexpr u32 bindless_combined_image_sampler_binding{3};
  static constexpr u32 bindless_sampler_binding{4};

  u32 resource_to_binding(ResourceType type);

  static BindlessResourceAllocator& get();
  static void init(VkDevice device, VmaAllocator allocator);
  static void shutdown();
  [[nodiscard]] VkDescriptorSetLayout main_set_layout() const { return main_set_layout_; }
  [[nodiscard]] VkDescriptorSet main_set() const { return main_set_; }
  [[nodiscard]] VkImageView create_image_view(const ImageViewCreateInfo& info) const;
  void set_frame_num(u32 frame_num);

  BindlessResourceInfo allocate_storage_buffer_descriptor(VkBuffer buffer);
  BindlessResourceInfo allocate_storage_img_descriptor(VkImageView view, VkImageLayout layout);
  BindlessResourceInfo allocate_sampled_img_descriptor(VkImageView view, VkImageLayout layout);
  BindlessResourceInfo allocate_sampler_descriptor(VkSampler sampler);
  void allocate_bindless_resource(VkDescriptorType descriptor_type, VkDescriptorImageInfo* img,
                                  VkDescriptorBufferInfo* buffer, u32 idx, u32 binding);

  void delete_texture(const TextureDeleteInfo& img);
  void delete_texture_view(const TextureViewDeleteInfo& info);
  void delete_buffer(const BufferDeleteInfo& info);

  void flush_deletions();

 private:
  template <typename T>
  struct DeleteQEntry {
    T data;
    u32 frame;
  };

  std::deque<DeleteQEntry<TextureDeleteInfo>> texture_delete_q_;
  std::deque<DeleteQEntry<TextureViewDeleteInfo>> texture_view_delete_q_;
  std::deque<DeleteQEntry<BufferDeleteInfo>> storage_buffer_delete_q_;

  ~BindlessResourceAllocator();
  BindlessResourceAllocator(VkDevice device, VmaAllocator allocator);

  VkDevice device_;
  VmaAllocator allocator_;
  IndexAllocator storage_image_allocator_{max_resource_descriptors};
  IndexAllocator storage_buffer_allocator_{max_resource_descriptors};
  IndexAllocator sampled_image_allocator_{max_resource_descriptors};
  IndexAllocator sampler_allocator_{max_sampler_descriptors};
  VkDescriptorPool main_pool_{};
  VkDescriptorSet main_set_{};
  VkDescriptorSetLayout main_set_layout_{};
  u64 frame_num_;
};
}  // namespace vk2
