#pragma once

#include <vulkan/vulkan_core.h>

#include <vector>

#include "Common.hpp"

namespace gfx::vk2 {

struct UpdateSwapchainInfo {
  VkPhysicalDevice phys_device;
  VkDevice device;
  VkSurfaceKHR surface;
  VkPresentModeKHR present_mode;
  uvec2 dims;
  u32 queue_idx;
  bool requested_resize;
};

struct SwapchainDesc {
  u32 width{};
  u32 height{};
  u32 buffer_count;
  bool fullscreen{};
  bool vsync{true};
};
struct Swapchain {
  enum class Status : u8 { Ready, Resized, NotReady };
  SwapchainDesc desc;
  std::vector<VkImage> imgs;
  std::vector<VkImageView> img_views;
  std::vector<VkSemaphore> acquire_semaphores;
  VkSemaphore release_semaphore{};
  u32 acquire_semaphore_idx{};
  u32 curr_swapchain_idx{};
  VkSwapchainKHR swapchain{};
  VkSurfaceKHR surface{};
  VkPresentModeKHR present_mode;
  VkFormat format;
  uvec2 dims;
  Status update(const UpdateSwapchainInfo& info);
  void destroy(VkDevice device);
  // void init(const UpdateSwapchainInfo& info);

 private:
  // void init(const UpdateSwapchainInfo& info, VkSwapchainKHR old);
};
// cannot fail
void create_swapchain(Swapchain& swapchain, const SwapchainDesc& desc);

}  // namespace gfx::vk2
