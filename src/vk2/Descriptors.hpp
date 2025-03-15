#pragma once

#include <vulkan/vulkan_core.h>

#include <unordered_map>

namespace vk2 {

struct DescriptorSetLayoutAndHash {
  VkDescriptorSetLayout layout;
  uint64_t hash;
};
struct DescriptorSetLayoutCache {
  void init(VkDevice device);
  void shutdown();
  void clear();
  DescriptorSetLayoutAndHash create_layout(VkDevice device,
                                           const VkDescriptorSetLayoutCreateInfo& create_info);
  [[nodiscard]] VkDescriptorSetLayout dummy_layout() const { return dummy_layout_; }

 private:
  VkDescriptorSetLayout dummy_layout_;
  std::unordered_map<uint64_t, VkDescriptorSetLayout> cache_;
  VkDevice device_;
};
}  // namespace vk2
