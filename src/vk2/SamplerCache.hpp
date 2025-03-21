#pragma once

#include <vulkan/vulkan_core.h>

#include <unordered_map>

#include "vk2/Resource.hpp"
namespace vk2 {

// TODO: manage lifetimes instead of clearing all
struct Sampler {
  VkSampler sampler;
  BindlessResourceInfo resource_info;
};

struct SamplerCache {
  static SamplerCache& get();
  [[nodiscard]] Sampler get_or_create_sampler(const VkSamplerCreateInfo& info);
  static void init(VkDevice device);
  void clear();
  static void destroy();

 private:
  VkDevice device_;
  std::unordered_map<uint64_t, Sampler> sampler_cache_;
};

}  // namespace vk2
