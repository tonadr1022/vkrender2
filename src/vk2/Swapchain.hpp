#pragma once

#include <vulkan/vulkan_core.h>

#include <vector>

#include "Common.hpp"

namespace vk2 {

struct UpdateSwapchainInfo {
  VkPhysicalDevice phys_device;
  VkDevice device;
  VkSurfaceKHR surface;
  VkSurfaceCapabilitiesKHR* surface_caps;
  VkPresentModeKHR present_mode;
  uvec2 dims;
  u32 queue_idx;
  bool requested_resize;
};

struct Swapchain {
  enum class Status : u8 { Ready, Resized, NotReady };
  // at most 4 images, quintuple buffering maybe in 2050
  std::vector<VkImage> imgs;
  std::vector<VkImageView> img_views;
  u32 img_cnt;
  VkSwapchainKHR swapchain{};
  VkPresentModeKHR present_mode;
  VkFormat format;
  uvec2 dims;
  void init(const UpdateSwapchainInfo& info, VkFormat format);
  Status update(const UpdateSwapchainInfo& info);
  void destroy(VkDevice device);
  void recreate_img_views(VkDevice device);
};

}  // namespace vk2
