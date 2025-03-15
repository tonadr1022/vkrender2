#include "Descriptors.hpp"

#include <volk.h>

#include <tracy/Tracy.hpp>

#include "vk2/Hash.hpp"
#include "vk2/VkCommon.hpp"

namespace {

uint64_t hash_descriptor_set_layout_create_info(const VkDescriptorSetLayoutCreateInfo& info) {
  uint64_t hash = 0;
  vk2::detail::hashing::hash_combine(hash, info.flags);
  for (uint32_t i = 0; i < info.bindingCount; i++) {
    const auto& binding = info.pBindings[i];
    vk2::detail::hashing::hash_combine(hash, binding.binding);
    vk2::detail::hashing::hash_combine(hash, binding.descriptorCount);
    vk2::detail::hashing::hash_combine(hash, binding.stageFlags);
    vk2::detail::hashing::hash_combine(hash, binding.descriptorType);
  }
  return hash;
}

}  // namespace

namespace vk2 {

DescriptorSetLayoutAndHash DescriptorSetLayoutCache::create_layout(
    VkDevice device, const VkDescriptorSetLayoutCreateInfo& create_info) {
  ZoneScoped;
  auto hash = hash_descriptor_set_layout_create_info(create_info);
  auto it = cache_.find(hash);
  if (it != cache_.end()) {
    return {it->second, it->first};
  }
  VkDescriptorSetLayout layout;
  VK_CHECK(vkCreateDescriptorSetLayout(device, &create_info, nullptr, &layout));
  cache_.emplace(hash, layout);
  return {layout, hash};
}

void DescriptorSetLayoutCache::clear() {
  for (auto& [hash, layout] : cache_) {
    vkDestroyDescriptorSetLayout(device_, layout, nullptr);
  }
  cache_.clear();
}

void DescriptorSetLayoutCache::shutdown() {
  vkDestroyDescriptorSetLayout(device_, dummy_layout_, nullptr);
  clear();
}

void DescriptorSetLayoutCache::init(VkDevice device) {
  device_ = device;
  VkDescriptorSetLayoutCreateInfo create_info{};
  create_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  create_info.bindingCount = 0;
  create_info.pBindings = nullptr;
  VK_CHECK(vkCreateDescriptorSetLayout(device_, &create_info, nullptr, &dummy_layout_));
}
}  // namespace vk2
