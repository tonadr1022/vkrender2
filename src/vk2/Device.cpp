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
  VkPhysicalDeviceVulkan12Features features12{
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
  features12.bufferDeviceAddress = true;
  features12.descriptorIndexing = true;
  features12.runtimeDescriptorArray = true;
  features12.shaderStorageImageArrayNonUniformIndexing = true;
  features12.shaderUniformBufferArrayNonUniformIndexing = true;
  features12.shaderSampledImageArrayNonUniformIndexing = true;
  features12.shaderStorageBufferArrayNonUniformIndexing = true;
  features12.shaderInputAttachmentArrayNonUniformIndexing = true;
  features12.shaderUniformTexelBufferArrayNonUniformIndexing = true;
  features12.descriptorBindingUniformBufferUpdateAfterBind = true;
  features12.descriptorBindingStorageImageUpdateAfterBind = true;
  features12.descriptorBindingSampledImageUpdateAfterBind = true;
  features12.descriptorBindingStorageBufferUpdateAfterBind = true;
  features12.descriptorBindingPartiallyBound = true;
  features12.descriptorBindingVariableDescriptorCount = true;
  features12.runtimeDescriptorArray = true;
  VkPhysicalDeviceFeatures features{};
  features.shaderStorageImageWriteWithoutFormat = true;
  features.depthClamp = true;

  // VkPhysicalDeviceVulkan13Features features13{
  //     .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
  // features13.dynamicRendering = true;
  // features13.synchronization2 = true;
  VkPhysicalDeviceDynamicRenderingFeatures dynamic_rendering_features{};
  dynamic_rendering_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES;
  dynamic_rendering_features.dynamicRendering = VK_TRUE;
  VkPhysicalDeviceSynchronization2Features sync2_features{};
  sync2_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES;
  sync2_features.synchronization2 = VK_TRUE;
  dynamic_rendering_features.pNext = &sync2_features;
  std::vector<const char*> extensions{{VK_KHR_COPY_COMMANDS_2_EXTENSION_NAME,
                                       VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME,
                                       VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME}};
  // features12.drawIndirectCount = true;
  features12.pNext = &dynamic_rendering_features;
  phys_selector.set_minimum_version(min_api_version_major, min_api_version_minor)
      .set_required_features_12(features12)
      .add_required_extensions(extensions)
      .set_required_features(features);
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
  device_ = vkb_device_.device;
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

Device& get_device() { return Device::get(); }

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

VkCommandPool Device::create_command_pool(u32 queue_idx, VkCommandPoolCreateFlags flags) const {
  VkCommandPoolCreateInfo info{.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                               .flags = flags,
                               .queueFamilyIndex = queue_idx};
  VkCommandPool pool;
  VK_CHECK(vkCreateCommandPool(device_, &info, nullptr, &pool));
  return pool;
}

VkCommandBuffer Device::create_command_buffer(VkCommandPool pool) const {
  VkCommandBufferAllocateInfo all_info{.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                                       .commandPool = pool,
                                       .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                                       .commandBufferCount = 1};
  VkCommandBuffer buffer;
  VK_CHECK(vkAllocateCommandBuffers(device_, &all_info, &buffer));
  return buffer;
}

void Device::create_command_buffers(VkCommandPool pool, std::span<VkCommandBuffer> buffers) const {
  VkCommandBufferAllocateInfo all_info{.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                                       .commandPool = pool,
                                       .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                                       .commandBufferCount = static_cast<uint32_t>(buffers.size())};
  VK_CHECK(vkAllocateCommandBuffers(device_, &all_info, buffers.data()));
}

VkFence Device::create_fence(VkFenceCreateFlags flags) const {
  VkFenceCreateInfo info{.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, .flags = flags};
  VkFence fence;
  VK_CHECK(vkCreateFence(device_, &info, nullptr, &fence));
  return fence;
}

VkSemaphore Device::create_semaphore() const {
  VkSemaphoreCreateInfo info{.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
  VkSemaphore semaphore;
  VK_CHECK(vkCreateSemaphore(device_, &info, nullptr, &semaphore));
  return semaphore;
}

void Device::destroy_fence(VkFence fence) const { vkDestroyFence(device_, fence, nullptr); }
void Device::destroy_semaphore(VkSemaphore semaphore) const {
  vkDestroySemaphore(device_, semaphore, nullptr);
}
void Device::destroy_command_pool(VkCommandPool pool) const {
  vkDestroyCommandPool(device_, pool, nullptr);
}

// void Device::destroy_img(AllocatedImage& img) {
//   vmaDestroyImage(allocator_, img.image, img.allocation);
// }
void Device::create_buffer(const VkBufferCreateInfo* info,
                           const VmaAllocationCreateInfo* alloc_info, VkBuffer& buffer,
                           VmaAllocation& allocation, VmaAllocationInfo& out_alloc_info) {
  VK_CHECK(vmaCreateBuffer(allocator_, info, alloc_info, &buffer, &allocation, &out_alloc_info));
}
}  // namespace vk2
