#include "Fence.hpp"

#include <volk.h>

#include <cassert>

#include "vk2/VkCommon.hpp"

namespace gfx::vk2 {

VkFence FencePool::allocate(bool reset) {
  if (!free_fences_.empty()) {
    VkFence f = free_fences_.back();
    free_fences_.pop_back();
    return f;
  }

  VkFenceCreateInfo info{.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
                         .flags = VK_FENCE_CREATE_SIGNALED_BIT};
  VkFence fence;
  VK_CHECK(vkCreateFence(device_, &info, nullptr, &fence));
  if (reset) {
    VK_CHECK(vkResetFences(device_, 1, &fence));
  }
  return fence;
}

void FencePool::free(VkFence fence) {
  VK_CHECK(vkResetFences(device_, 1, &fence));
  free_fences_.push_back(fence);
}
namespace {
FencePool* instance{};
}
void FencePool::destroy() {
  assert(instance);
  delete instance;
}

void FencePool::init(VkDevice device) {
  assert(!instance);
  instance = new FencePool{device};
}
FencePool& FencePool::get() {
  assert(instance);
  return *instance;
}
FencePool::FencePool(VkDevice device) : device_(device) {}
FencePool::~FencePool() {
  for (auto& f : free_fences_) {
    vkDestroyFence(device_, f, nullptr);
  }
}
}  // namespace gfx::vk2
