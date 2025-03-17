#pragma once

#include <vulkan/vulkan_core.h>

#include <unordered_map>
namespace vk2 {

// TODO: manage lifetimes instead of clearing all
struct SamplerCache {
  static SamplerCache& get();
  [[nodiscard]] VkSampler get_or_create_sampler(const VkSamplerCreateInfo& info);
  static void init(VkDevice device);
  void clear();
  static void destroy();

 private:
  VkDevice device_;
  std::unordered_map<uint64_t, VkSampler> sampler_cache_;
};

}  // namespace vk2
