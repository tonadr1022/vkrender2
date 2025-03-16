#pragma once

#include <vulkan/vulkan_core.h>

#include <vector>

#include "Common.hpp"
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

class BindlessResourceAllocator {
 public:
  static constexpr u32 max_resource_descriptors{100'000};
  static constexpr u32 max_sampler_descriptors{1000};
  static BindlessResourceAllocator& get();
  static void init(VkDevice device, VmaAllocator allocator);
  static void shutdown();
  [[nodiscard]] VkDescriptorSetLayout main_set_layout() const { return main_set_layout_; }
  [[nodiscard]] VkDescriptorSet main_set() const { return main_set_; }
  [[nodiscard]] VkImageView create_image_view(const ImageViewCreateInfo& info) const;

  // TODO: img alloc info
  Texture alloc_img(const VkImageCreateInfo& create_info,
                    VkMemoryPropertyFlags req_flags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                    bool mapped = false);
  Texture alloc_img_with_view(const VkImageCreateInfo& create_info,
                              const VkImageSubresourceRange& range, VkImageViewType type,
                              VkMemoryPropertyFlags req_flags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                              bool mapped = false);
  void destroy_image(Texture& image);

 private:
  ~BindlessResourceAllocator();
  BindlessResourceAllocator(VkDevice device, VmaAllocator allocator);
  VkDevice device_;
  VmaAllocator allocator_;
  IndexAllocator storage_image_allocator_{max_resource_descriptors};
  IndexAllocator sampled_image_allocator_{max_resource_descriptors};
  VkDescriptorPool main_pool_{};
  VkDescriptorSet main_set_{};
  VkDescriptorSetLayout main_set_layout_{};
};
}  // namespace vk2
