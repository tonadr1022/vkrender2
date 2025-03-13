#include "Swapchain.hpp"

#include <volk.h>
#include <vulkan/vulkan_core.h>

#include <tracy/Tracy.hpp>
#include <vector>

#include "vk2/Device.hpp"
#include "vk2/VkCommon.hpp"

namespace vk2 {
namespace {

VkSwapchainKHR create_swapchain(const UpdateSwapchainInfo& info, VkFormat format,
                                VkSwapchainKHR old) {
  ZoneScoped;
  VkCompositeAlphaFlagBitsKHR surface_composite =
      (info.surface_caps->supportedCompositeAlpha & VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR)
          ? VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR
      : (info.surface_caps->supportedCompositeAlpha & VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR)
          ? VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR
      : (info.surface_caps->supportedCompositeAlpha & VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR)
          ? VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR
          : VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR;
  VkSwapchainKHR res;
  VK_CHECK(vkCreateSwapchainKHR(
      info.device,
      addr(VkSwapchainCreateInfoKHR{
          .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
          .surface = info.surface,
          .minImageCount = std::max(2u, info.surface_caps->minImageCount),
          .imageFormat = format,
          .imageColorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR,
          .imageExtent = {.width = info.dims.x, .height = info.dims.y},
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

void Swapchain::init(const UpdateSwapchainInfo& info, VkFormat format) {
  ZoneScoped;
  VkSwapchainKHR new_swapchain = create_swapchain(info, format, nullptr);
  assert(new_swapchain);
  uint32_t new_img_cnt = 0;
  VK_CHECK(vkGetSwapchainImagesKHR(info.device, new_swapchain, &new_img_cnt, nullptr));
  imgs.resize(new_img_cnt);
  VK_CHECK(vkGetSwapchainImagesKHR(info.device, new_swapchain, &new_img_cnt, imgs.data()));
  swapchain = new_swapchain;
  dims = info.dims;
  this->present_mode = info.present_mode;
  img_cnt = new_img_cnt;
  this->format = format;
}

Swapchain::Status Swapchain::update(const UpdateSwapchainInfo& info) {
  if (info.dims.x == 0 || info.dims.y == 0) {
    return Swapchain::Status::NotReady;
  }

  if (dims.x == info.dims.x && dims.y == info.dims.y && !info.requested_resize) {
    return Swapchain::Status::Ready;
  }

  VkSwapchainKHR old = swapchain;
  init(info, format);
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
    img_views[i] = vk2::device().create_image_view(
        {.image = imgs[i], .view_type = VK_IMAGE_VIEW_TYPE_2D, .format = format});
  }
}

}  // namespace vk2
