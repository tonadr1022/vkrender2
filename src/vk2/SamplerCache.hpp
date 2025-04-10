#pragma once

#include <vulkan/vulkan_core.h>

#include <unordered_map>

#include "vk2/Resource.hpp"
namespace gfx::vk2 {

// TODO: manage lifetimes instead of clearing all
struct Sampler {
  VkSampler sampler;
  BindlessResourceInfo resource_info;
};

struct SamplerCreateInfo {
  VkFilter min_filter{VK_FILTER_NEAREST};
  VkFilter mag_filter{VK_FILTER_NEAREST};
  VkSamplerMipmapMode mipmap_mode{VK_SAMPLER_MIPMAP_MODE_NEAREST};
  VkSamplerAddressMode address_mode{VK_SAMPLER_ADDRESS_MODE_REPEAT};
  float min_lod{-1000.f};
  float max_lod{1000.f};
  VkBorderColor border_color{VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK};
  bool anisotropy_enable{};
  float max_anisotropy{};
  bool compare_enable{false};
  VkCompareOp compare_op{};
};

struct SamplerCache {
  static SamplerCache& get();
  [[nodiscard]] Sampler get_or_create_sampler(const SamplerCreateInfo& info);
  static void init(VkDevice device);
  void clear();
  Sampler get_linear_sampler();
  static void destroy();

 private:
  VkDevice device_;
  std::unordered_map<uint64_t, Sampler> sampler_cache_;
};

}  // namespace gfx::vk2
