#include "Swapchain.hpp"

#include <volk.h>
#include <vulkan/vk_enum_string_helper.h>
#include <vulkan/vulkan_core.h>

#include <tracy/Tracy.hpp>
#include <vector>

#include "vk2/Device.hpp"
#include "vk2/VkCommon.hpp"

namespace vk2 {
namespace {

VkSwapchainKHR create_swapchain(const UpdateSwapchainInfo& info, VkFormat format,
                                VkSwapchainKHR old, VkSurfaceCapabilitiesKHR surface_caps) {
  ZoneScoped;
  VkCompositeAlphaFlagBitsKHR surface_composite =
      (surface_caps.supportedCompositeAlpha & VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR)
          ? VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR
      : (surface_caps.supportedCompositeAlpha & VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR)
          ? VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR
      : (surface_caps.supportedCompositeAlpha & VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR)
          ? VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR
          : VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR;
  VkSwapchainKHR res;
  VkExtent2D extent = {info.dims.x, info.dims.y};
  VK_CHECK(vkCreateSwapchainKHR(
      info.device,
      addr(VkSwapchainCreateInfoKHR{
          .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
          .surface = info.surface,
          .minImageCount = std::max(2u, surface_caps.minImageCount),
          .imageFormat = format,
          .imageColorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR,
          .imageExtent = extent,
          .imageArrayLayers = 1,
          .imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
          .queueFamilyIndexCount = 1,
          .pQueueFamilyIndices = &info.queue_idx,
          .preTransform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR,
          .compositeAlpha = surface_composite,
          .presentMode = info.present_mode,
          .oldSwapchain = old}),
      nullptr, &res));
  return res;
}

}  // namespace

void Swapchain::init(const UpdateSwapchainInfo& info, VkFormat format, VkSwapchainKHR old) {
  VkSurfaceCapabilitiesKHR surface_caps;
  {
    ZoneScopedN("get surface caps");
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(vk2::get_device().phys_device(), info.surface,
                                              &surface_caps);
  }
  VkSwapchainKHR new_swapchain = create_swapchain(info, format, old, surface_caps);
  assert(new_swapchain);
  uint32_t new_img_cnt = 0;
  VK_CHECK(vkGetSwapchainImagesKHR(info.device, new_swapchain, &new_img_cnt, nullptr));
  imgs.resize(new_img_cnt);
  VK_CHECK(vkGetSwapchainImagesKHR(info.device, new_swapchain, &new_img_cnt, imgs.data()));
  swapchain = new_swapchain;
  this->present_mode = info.present_mode;
  img_cnt = new_img_cnt;
  this->format = format;
  this->dims = info.dims;
}

Swapchain::Status Swapchain::update(const UpdateSwapchainInfo& info) {
  ZoneScoped;
  if (info.dims.x == 0 || info.dims.y == 0) {
    return Swapchain::Status::NotReady;
  }

  if (dims.x == info.dims.x && dims.y == info.dims.y && !info.requested_resize) {
    return Swapchain::Status::Ready;
  }

  VkSwapchainKHR old = swapchain;
  init(info, format, old);
  VK_CHECK(vkDeviceWaitIdle(info.device));

  vkDestroySwapchainKHR(info.device, old, nullptr);

  return Swapchain::Status::Resized;
}

void Swapchain::destroy(VkDevice device) {
  for (u32 i = 0; i < img_cnt; i++) {
    if (img_views[i]) {
      vkDestroyImageView(device, img_views[i], nullptr);
      img_views[i] = nullptr;
    }
  }
  vkDestroySwapchainKHR(device, swapchain, nullptr);
  swapchain = nullptr;
  img_cnt = 0;
}

void Swapchain::recreate_img_views(VkDevice device) {
  img_views.resize(img_cnt, nullptr);
  for (u32 i = 0; i < img_cnt; i++) {
    auto& img_view = img_views[i];
    if (img_view) {
      vkDestroyImageView(device, img_view, nullptr);
      img_view = nullptr;
    }

    auto view_info = VkImageViewCreateInfo{
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .image = imgs[i],
        .viewType = VK_IMAGE_VIEW_TYPE_2D,
        .format = format,
        .subresourceRange = VkImageSubresourceRange{.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                                                    .baseMipLevel = 0,
                                                    .levelCount = VK_REMAINING_MIP_LEVELS,
                                                    .baseArrayLayer = 0,
                                                    .layerCount = VK_REMAINING_ARRAY_LAYERS}};
    VK_CHECK(vkCreateImageView(device, &view_info, nullptr, &img_views[i]));
  }
}

void Swapchain::init(const UpdateSwapchainInfo& info, VkFormat format) {
  init(info, format, nullptr);
}
}  // namespace vk2
