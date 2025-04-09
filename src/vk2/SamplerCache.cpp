#include "SamplerCache.hpp"

#include <volk.h>
#include <vulkan/vulkan_core.h>

#include <tracy/Tracy.hpp>

#include "vk2/BindlessResourceAllocator.hpp"
#include "vk2/Hash.hpp"

namespace vk2 {

Sampler SamplerCache::get_or_create_sampler(const SamplerCreateInfo& info) {
  ZoneScoped;
  auto h =
      std::make_tuple(info.address_mode, info.min_filter, info.mag_filter, info.anisotropy_enable,
                      info.max_anisotropy, info.compare_enable, info.compare_op);
  auto hash = detail::hashing::hash<decltype(h)>{}(h);

  auto it = sampler_cache_.find(hash);
  if (it != sampler_cache_.end()) {
    return it->second;
  }

  Sampler sampler;
  VkSamplerCreateInfo cinfo{.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
                            .magFilter = info.mag_filter,
                            .minFilter = info.min_filter,
                            .mipmapMode = info.mipmap_mode,
                            .addressModeU = info.address_mode,
                            .addressModeV = info.address_mode,
                            .addressModeW = info.address_mode,
                            .anisotropyEnable = info.anisotropy_enable,
                            .maxAnisotropy = info.max_anisotropy,
                            .compareEnable = info.compare_enable,
                            .compareOp = info.compare_op,
                            .minLod = info.min_lod,
                            .maxLod = info.max_lod,
                            .borderColor = info.border_color};
  vkCreateSampler(device_, &cinfo, nullptr, &sampler.sampler);
  assert(sampler.sampler);
  sampler.resource_info =
      BindlessResourceAllocator::get().allocate_sampler_descriptor(sampler.sampler);
  sampler_cache_.emplace(hash, sampler);

  return sampler;
}

void SamplerCache::clear() {
  for (auto& [_, sampler] : sampler_cache_) {
    vkDestroySampler(device_, sampler.sampler, nullptr);
  }
  sampler_cache_.clear();
}

namespace {
SamplerCache* instance{nullptr};
}  // namespace

void SamplerCache::init(VkDevice device) {
  assert(!instance);
  instance = new SamplerCache;
  instance->device_ = device;
}

SamplerCache& SamplerCache::get() { return *instance; }

void SamplerCache::destroy() {
  assert(instance);
  instance->clear();
  delete instance;
}

Sampler SamplerCache::get_linear_sampler() {
  return get_or_create_sampler({
      .min_filter = VK_FILTER_LINEAR,
      .mag_filter = VK_FILTER_LINEAR,
      .mipmap_mode = VK_SAMPLER_MIPMAP_MODE_LINEAR,
      .address_mode = VK_SAMPLER_ADDRESS_MODE_REPEAT,
  });
}
}  // namespace vk2
