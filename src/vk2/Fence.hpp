#pragma once
#include <vulkan/vulkan_core.h>

#include <vector>
namespace vk2 {

// TODO: thread safety
struct FencePool {
  static FencePool& get();
  static void destroy();
  static void init(VkDevice);

  VkFence allocate(bool reset);
  void free(VkFence fence);
  // contains fences in the reset state
  std::vector<VkFence> free_fences_;
  VkDevice device_;

 private:
  explicit FencePool(VkDevice device);
  ~FencePool();
};
}  // namespace vk2
