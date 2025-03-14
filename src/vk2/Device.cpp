#include "Device.hpp"

#include <volk.h>
#include <vulkan/vulkan_core.h>

#include <cassert>
#include <tracy/Tracy.hpp>

#include "Common.hpp"
#include "Logger.hpp"
#include "VkBootstrap.h"
#include "VkCommon.hpp"

namespace vk2 {
namespace {
Device* g_device{};
}

void Device::init(const CreateInfo& info) {
  g_device = new Device;
  g_device->init_impl(info);
}

void Device::destroy() { delete g_device; }

Device& Device::get() {
  assert(g_device);
  return *g_device;
}

void Device::init_impl(const CreateInfo& info) {
  ZoneScoped;
  surface_ = info.surface;
  vkb::PhysicalDeviceSelector phys_selector(info.instance, info.surface);
  VkPhysicalDeviceVulkan13Features features13{
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
  features13.dynamicRendering = true;
  features13.synchronization2 = true;
  VkPhysicalDeviceVulkan12Features features12{
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
  features12.bufferDeviceAddress = true;
  features12.descriptorIndexing = true;
  // features12.drawIndirectCount = true;
  phys_selector.set_minimum_version(min_api_version_major, min_api_version_minor)
      .set_required_features_13(features13);
  auto phys_ret = phys_selector.select();
  if (!phys_ret) {
    LCRITICAL("Failed to select physical device: {}", phys_ret.error().message());
    exit(1);
  }
  vkb_phys_device_ = std::move(phys_ret.value());

  vkb::DeviceBuilder dev_builder(vkb_phys_device_);
  auto dev_ret = dev_builder.build();
  if (!dev_ret) {
    LCRITICAL("Failed to acquire logical device: {}", dev_ret.error().message());
    exit(1);
  }
  vkb_device_ = std::move(dev_ret.value());
  main_del_queue_.push([this]() { vkb::destroy_device(vkb_device_); });

  {
    ZoneScopedN("init volk device");
    volkLoadDevice(device());
  }

  VmaVulkanFunctions vma_vulkan_func{};
  vma_vulkan_func.vkGetInstanceProcAddr = vkGetInstanceProcAddr;
  vma_vulkan_func.vkGetDeviceProcAddr = vkGetDeviceProcAddr;
  vma_vulkan_func.vkAllocateMemory = vkAllocateMemory;
  vma_vulkan_func.vkBindBufferMemory = vkBindBufferMemory;
  vma_vulkan_func.vkBindImageMemory = vkBindImageMemory;
  vma_vulkan_func.vkCreateBuffer = vkCreateBuffer;
  vma_vulkan_func.vkCreateImage = vkCreateImage;
  vma_vulkan_func.vkDestroyBuffer = vkDestroyBuffer;
  vma_vulkan_func.vkDestroyImage = vkDestroyImage;
  vma_vulkan_func.vkFlushMappedMemoryRanges = vkFlushMappedMemoryRanges;
  vma_vulkan_func.vkFreeMemory = vkFreeMemory;
  vma_vulkan_func.vkGetBufferMemoryRequirements = vkGetBufferMemoryRequirements;
  vma_vulkan_func.vkGetImageMemoryRequirements = vkGetImageMemoryRequirements;
  vma_vulkan_func.vkGetPhysicalDeviceMemoryProperties = vkGetPhysicalDeviceMemoryProperties;
  vma_vulkan_func.vkGetPhysicalDeviceMemoryProperties2KHR = vkGetPhysicalDeviceMemoryProperties2KHR;
  vma_vulkan_func.vkGetPhysicalDeviceProperties = vkGetPhysicalDeviceProperties;
  vma_vulkan_func.vkInvalidateMappedMemoryRanges = vkInvalidateMappedMemoryRanges;
  vma_vulkan_func.vkMapMemory = vkMapMemory;
  vma_vulkan_func.vkUnmapMemory = vkUnmapMemory;
  vma_vulkan_func.vkCmdCopyBuffer = vkCmdCopyBuffer;
  VmaAllocatorCreateInfo allocator_info{
      .flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT,
      .physicalDevice = phys_device(),
      .device = device(),
      .pVulkanFunctions = &vma_vulkan_func,
      .instance = info.instance,
  };
  {
    ZoneScopedN("init vma");
    VK_CHECK(vmaCreateAllocator(&allocator_info, &allocator_));
    main_del_queue_.push([allocator = allocator_]() { vmaDestroyAllocator(allocator); });
  }
}

void Device::destroy_impl() { vkb::destroy_device(vkb_device_); }

Device& device() { return Device::get(); }

VkFormat Device::get_swapchain_format() {
  u32 cnt;
  VK_CHECK(vkGetPhysicalDeviceSurfaceFormatsKHR(phys_device(), surface_, &cnt, nullptr));
  assert(cnt > 0);
  std::vector<VkSurfaceFormatKHR> formats(cnt);
  VK_CHECK(vkGetPhysicalDeviceSurfaceFormatsKHR(phys_device(), surface_, &cnt, formats.data()));
  for (auto format : formats) {
    if (format.format == VK_FORMAT_R8G8B8A8_UNORM || format.format == VK_FORMAT_B8G8R8A8_UNORM) {
      return format.format;
    }
  }
  return formats[0].format;
}

VkImageView Device::create_image_view(const ImageViewCreateInfo& info) const {
  VkImageView view;
  VkImageViewCreateInfo i{.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                          .image = info.image,
                          .viewType = VK_IMAGE_VIEW_TYPE_2D,
                          .format = info.format,
                          .subresourceRange = info.subresource_range};
  VK_CHECK(vkCreateImageView(device(), &i, nullptr, &view));
  return view;
}
VkCommandPool Device::create_command_pool(u32 queue_idx, VkCommandPoolCreateFlags flags) const {
  VkCommandPoolCreateInfo info{.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                               .flags = flags,
                               .queueFamilyIndex = queue_idx};
  VkCommandPool pool;
  VK_CHECK(vkCreateCommandPool(device(), &info, nullptr, &pool));
  return pool;
}

VkCommandBuffer Device::create_command_buffer(VkCommandPool pool) const {
  VkCommandBufferAllocateInfo all_info{.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                                       .commandPool = pool,
                                       .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                                       .commandBufferCount = 1};
  VkCommandBuffer buffer;
  VK_CHECK(vkAllocateCommandBuffers(device(), &all_info, &buffer));
  return buffer;
}

void Device::create_command_buffers(VkCommandPool pool, std::span<VkCommandBuffer> buffers) const {
  VkCommandBufferAllocateInfo all_info{.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                                       .commandPool = pool,
                                       .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                                       .commandBufferCount = static_cast<uint32_t>(buffers.size())};
  VK_CHECK(vkAllocateCommandBuffers(device(), &all_info, buffers.data()));
}

VkFence Device::create_fence(VkFenceCreateFlags flags) const {
  VkFenceCreateInfo info{.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, .flags = flags};
  VkFence fence;
  VK_CHECK(vkCreateFence(device(), &info, nullptr, &fence));
  return fence;
}

VkSemaphore Device::create_semaphore() const {
  VkSemaphoreCreateInfo info{.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
  VkSemaphore semaphore;
  VK_CHECK(vkCreateSemaphore(device(), &info, nullptr, &semaphore));
  return semaphore;
}

void Device::destroy_fence(VkFence fence) const { vkDestroyFence(device(), fence, nullptr); }
void Device::destroy_semaphore(VkSemaphore semaphore) const {
  vkDestroySemaphore(device(), semaphore, nullptr);
}
void Device::destroy_command_pool(VkCommandPool pool) const {
  vkDestroyCommandPool(device(), pool, nullptr);
}
}  // namespace vk2
