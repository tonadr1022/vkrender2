#include "SamplerCache.hpp"

#include <volk.h>

#include <tracy/Tracy.hpp>

#include "vk2/Hash.hpp"

namespace vk2 {

VkSampler SamplerCache::get_or_create_sampler(const VkSamplerCreateInfo& info) {
  ZoneScoped;
  auto h = std::make_tuple(info.addressModeU, info.addressModeV, info.addressModeW, info.minFilter,
                           info.magFilter, info.anisotropyEnable, info.maxAnisotropy, info.flags,
                           info.compareEnable, info.compareOp);
  auto hash = detail::hashing::hash<decltype(h)>{}(h);

  auto it = sampler_cache_.find(hash);
  if (it != sampler_cache_.end()) {
    return it->second;
  }

  VkSampler sampler;
  vkCreateSampler(device_, &info, nullptr, &sampler);

  sampler_cache_.emplace(hash, sampler);

  return sampler;
}

void SamplerCache::clear() {
  for (auto& [_, sampler] : sampler_cache_) {
    vkDestroySampler(device_, sampler, nullptr);
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

}  // namespace vk2
