#include "Swapchain.hpp"

#include <volk.h>
#include <vulkan/vk_enum_string_helper.h>
#include <vulkan/vulkan_core.h>

#include <tracy/Tracy.hpp>
#include <vector>

#include "core/Logger.hpp"
#include "vk2/Device.hpp"
#include "vk2/VkCommon.hpp"

namespace gfx::vk2 {
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
  VkSwapchainCreateInfoKHR swap_info{
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
      .oldSwapchain = old};
  VK_CHECK(vkCreateSwapchainKHR(info.device, &swap_info, nullptr, &res));
  return res;
}

}  // namespace

void Swapchain::init(const UpdateSwapchainInfo& info, VkSwapchainKHR old) {
  VkSurfaceCapabilitiesKHR surface_caps;
  vkGetPhysicalDeviceSurfaceCapabilitiesKHR(get_device().get_physical_device(), info.surface,
                                            &surface_caps);
  VkSwapchainKHR new_swapchain = create_swapchain(info, format, old, surface_caps);
  assert(new_swapchain);
  uint32_t new_img_cnt = 0;
  VK_CHECK(vkGetSwapchainImagesKHR(info.device, new_swapchain, &new_img_cnt, nullptr));
  imgs.resize(new_img_cnt);
  VK_CHECK(vkGetSwapchainImagesKHR(info.device, new_swapchain, &new_img_cnt, imgs.data()));
  swapchain = new_swapchain;
  this->present_mode = info.present_mode;
  this->dims = info.dims;

  if (acquire_semaphores.size() == 0) {
    for (size_t i = 0; i < imgs.size(); i++) {
      acquire_semaphores.emplace_back(get_device().create_semaphore(false));
    }
  }
}

Swapchain::Status Swapchain::update(const UpdateSwapchainInfo& info) {
  ZoneScoped;
  if (info.dims.x == 0 || info.dims.y == 0) {
    LINFO("not ready");
    return Swapchain::Status::NotReady;
  }

  if (dims.x == info.dims.x && dims.y == info.dims.y && !info.requested_resize) {
    return Swapchain::Status::Ready;
  }

  VkSwapchainKHR old = swapchain;
  init(info, old);
  VK_CHECK(vkDeviceWaitIdle(info.device));

  vkDestroySwapchainKHR(info.device, old, nullptr);

  return Swapchain::Status::Resized;
}

void Swapchain::destroy(VkDevice device) {
  for (auto& img_view : img_views) {
    if (img_view) {
      vkDestroyImageView(device, img_view, nullptr);
      img_view = nullptr;
    }
  }
  for (auto& semaphore : acquire_semaphores) {
    if (semaphore) {
      vkDestroySemaphore(device, semaphore, nullptr);
      semaphore = nullptr;
    }
  }
  vkDestroySwapchainKHR(device, swapchain, nullptr);
  swapchain = nullptr;
}

void Swapchain::init(const UpdateSwapchainInfo& info) { init(info, nullptr); }

void create_swapchain(Swapchain& swapchain, const SwapchainDesc& desc) {
  VkSurfaceCapabilitiesKHR surface_caps;
  VK_CHECK(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(get_device().get_physical_device(),
                                                     swapchain.surface, &surface_caps));
  u32 format_count{};
  VK_CHECK(vkGetPhysicalDeviceSurfaceFormatsKHR(get_device().get_physical_device(),
                                                swapchain.surface, &format_count, nullptr));
  std::vector<VkSurfaceFormatKHR> available_surface_formats(format_count);
  VK_CHECK(vkGetPhysicalDeviceSurfaceFormatsKHR(get_device().get_physical_device(),
                                                swapchain.surface, &format_count,
                                                available_surface_formats.data()));
  u32 present_mode_count;
  VK_CHECK(vkGetPhysicalDeviceSurfacePresentModesKHR(
      get_device().get_physical_device(), swapchain.surface, &present_mode_count, nullptr));
  std::vector<VkPresentModeKHR> available_present_modes(present_mode_count);
  VK_CHECK(vkGetPhysicalDeviceSurfacePresentModesKHR(get_device().get_physical_device(),
                                                     swapchain.surface, &present_mode_count,
                                                     available_present_modes.data()));

  VkSurfaceFormatKHR chosen_surface_format;
  bool found_format = false;
  if (available_surface_formats.size() == 1 &&
      available_surface_formats[0].format == VK_FORMAT_UNDEFINED) {
    found_format = true;
    chosen_surface_format.format = VK_FORMAT_B8G8R8A8_UNORM;
    chosen_surface_format.colorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
  } else {
    for (const auto& available_format : available_surface_formats) {
      if (available_format.format == VK_FORMAT_B8G8R8A8_UNORM &&
          available_format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
        chosen_surface_format = available_format;
        found_format = true;
        break;
      }
    }

    if (!found_format) {
      chosen_surface_format = available_surface_formats[0];
    }
  }

  if (surface_caps.currentExtent.width != 0xFFFFFFFF &&
      surface_caps.currentExtent.height != 0xFFFFFFFF) {
    swapchain.dims = {surface_caps.currentExtent.width, surface_caps.currentExtent.height};
  } else {
    swapchain.dims = {desc.width, desc.height};
    swapchain.dims = {std::max(surface_caps.minImageExtent.width,
                               std::min(surface_caps.maxImageExtent.width, swapchain.dims.x)),
                      std::max(surface_caps.minImageExtent.height,
                               std::min(surface_caps.maxImageExtent.height, swapchain.dims.x))};
  }
  u32 image_count = std::max(desc.buffer_count, surface_caps.minImageCount);
  if ((surface_caps.maxImageCount > 0) && image_count > surface_caps.maxImageCount) {
    image_count = surface_caps.maxImageCount;
  }
  VkPresentModeKHR chosen_present_mode{VK_PRESENT_MODE_FIFO_KHR};
  if (!desc.vsync) {
    for (const auto& avail_present_mode : available_present_modes) {
      if (avail_present_mode == VK_PRESENT_MODE_MAILBOX_KHR) {
        chosen_present_mode = VK_PRESENT_MODE_MAILBOX_KHR;
        break;
      }
      if (avail_present_mode == VK_PRESENT_MODE_IMMEDIATE_KHR) {
        chosen_present_mode = VK_PRESENT_MODE_IMMEDIATE_KHR;
      }
    }
  }
  VkSwapchainCreateInfoKHR swap_info{
      .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
      .surface = swapchain.surface,
      .minImageCount = std::max(2u, surface_caps.minImageCount),
      .imageFormat = chosen_surface_format.format,
      .imageColorSpace = chosen_surface_format.colorSpace,
      .imageExtent = {swapchain.dims.x, swapchain.dims.y},
      .imageArrayLayers = 1,
      .imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                    VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
      .queueFamilyIndexCount = 1,
      .pQueueFamilyIndices = &get_device().get_queue(QueueType::Graphics).family_idx,
      .preTransform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR,
      .compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
      .presentMode = chosen_present_mode,
      .oldSwapchain = swapchain.swapchain};
  VK_CHECK(vkCreateSwapchainKHR(get_device().device(), &swap_info, nullptr, &swapchain.swapchain));

  if (swap_info.oldSwapchain) {
    get_device().enqueue_delete_swapchain(swap_info.oldSwapchain);
  }

  uint32_t new_img_cnt = 0;
  VK_CHECK(
      vkGetSwapchainImagesKHR(get_device().device(), swapchain.swapchain, &new_img_cnt, nullptr));
  swapchain.imgs.resize(new_img_cnt);
  VK_CHECK(vkGetSwapchainImagesKHR(get_device().device(), swapchain.swapchain, &new_img_cnt,
                                   swapchain.imgs.data()));
  swapchain.format = chosen_surface_format.format;
  swapchain.present_mode = chosen_present_mode;
  swapchain.img_views.resize(swapchain.imgs.size(), nullptr);
  for (u32 i = 0; i < swapchain.img_views.size(); i++) {
    auto& img_view = swapchain.img_views[i];
    if (img_view) {
      get_device().delete_texture_view(img_view);
      img_view = nullptr;
    }

    auto view_info = VkImageViewCreateInfo{
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .image = swapchain.imgs[i],
        .viewType = VK_IMAGE_VIEW_TYPE_2D,
        .format = swapchain.format,
        .subresourceRange = VkImageSubresourceRange{.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                                                    .baseMipLevel = 0,
                                                    .levelCount = VK_REMAINING_MIP_LEVELS,
                                                    .baseArrayLayer = 0,
                                                    .layerCount = VK_REMAINING_ARRAY_LAYERS}};
    VK_CHECK(
        vkCreateImageView(get_device().device(), &view_info, nullptr, &swapchain.img_views[i]));
    if (swapchain.acquire_semaphores.empty()) {
      for (size_t i = 0; i < swapchain.imgs.size(); i++) {
        swapchain.acquire_semaphores.emplace_back(get_device().create_semaphore(false));
      }
    }
  }
}

}  // namespace gfx::vk2
