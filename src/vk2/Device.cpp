#include "Device.hpp"

#include <volk.h>
#include <vulkan/vulkan_core.h>

#include <cassert>
#include <tracy/Tracy.hpp>

#include "Common.hpp"
#include "Logger.hpp"
#include "VkBootstrap.h"
#include "VkCommon.hpp"
#include "vk2/BindlessResourceAllocator.hpp"

namespace gfx::vk2 {
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
  features12.descriptorBindingUpdateUnusedWhilePending = true;
  features12.descriptorBindingPartiallyBound = true;
  features12.descriptorBindingVariableDescriptorCount = true;
  features12.runtimeDescriptorArray = true;
  features12.timelineSemaphore = true;
  VkPhysicalDeviceFeatures features{};
  features.shaderStorageImageWriteWithoutFormat = true;
  features.depthClamp = true;
  features.multiDrawIndirect = true;
  VkPhysicalDeviceVulkan11Features features11{
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
  features11.shaderDrawParameters = true;

  VkPhysicalDeviceVulkan13Features features13{
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
  features13.dynamicRendering = true;
  features13.synchronization2 = true;
  VkPhysicalDeviceDynamicRenderingFeatures dynamic_rendering_features{};
  dynamic_rendering_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES;
  dynamic_rendering_features.dynamicRendering = VK_TRUE;
  VkPhysicalDeviceSynchronization2Features sync2_features{};
  sync2_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES;
  sync2_features.synchronization2 = VK_TRUE;
  std::vector<const char*> extensions{{VK_KHR_COPY_COMMANDS_2_EXTENSION_NAME,
                                       VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME,
                                       // #ifdef TRACY_ENABLE
                                       // VK_KHR_CALIBRATED_TIMESTAMPS_EXTENSION_NAME,
                                       // #endif
                                       VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME}};

  // NOT ON MACOS :(
#ifndef __APPLE__
  features12.drawIndirectCount = true;
#endif

  phys_selector.set_minimum_version(min_api_version_major, min_api_version_minor)
      .set_required_features_12(features12)
      .set_required_features_11(features11)
      .add_required_extensions(extensions)
      .prefer_gpu_device_type(vkb::PreferredDeviceType::discrete)
      .add_required_extension_features(dynamic_rendering_features)
      .add_required_extension_features(sync2_features)
      .set_required_features(features);
  auto phys_ret = phys_selector.select_devices();
  if (!phys_ret || phys_ret.value().empty()) {
    LCRITICAL("Failed to select physical device: {}", phys_ret.error().message());
    exit(1);
  }
  bool found_discrete_device = false;
  for (auto& v : phys_ret.value()) {
    if (v.properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
      vkb_phys_device_ = v;
      found_discrete_device = true;
      break;
    }
  }
  if (!found_discrete_device) {
    vkb_phys_device_ = std::move(phys_ret.value()[0]);
  }
  LINFO("Selected Device: {}", vkb_phys_device_.properties.deviceName);

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

VkSemaphore Device::create_semaphore(bool timeline) const {
  VkSemaphoreTypeCreateInfo timeline_create_info;
  timeline_create_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
  timeline_create_info.pNext = nullptr;
  timeline_create_info.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
  timeline_create_info.initialValue = 0;

  VkSemaphoreCreateInfo info{
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
      .pNext = timeline ? &timeline_create_info : nullptr,
  };
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

void Device::create_buffer(const VkBufferCreateInfo* info,
                           const VmaAllocationCreateInfo* alloc_info, VkBuffer& buffer,
                           VmaAllocation& allocation, VmaAllocationInfo& out_alloc_info) {
  VK_CHECK(vmaCreateBuffer(allocator_, info, alloc_info, &buffer, &allocation, &out_alloc_info));
}

ImageHandle Device::create_image(const ImageCreateInfo& create_info) {
  LINFO("creating image");
  return img_pool_.alloc(create_info);
}

Image2::Image2(const ImageCreateInfo& create_info) {
  this->create_info = create_info;
  VmaAllocationCreateFlags alloc_flags{};
  auto attachment_usage = format_is_color(create_info.format)
                              ? VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT
                              : VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;

  if (create_info.override_usage_flags) {
    usage = create_info.override_usage_flags;
    if (create_info.override_usage_flags & VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT ||
        create_info.override_usage_flags & VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT ||
        create_info.override_usage_flags & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT ||
        create_info.override_usage_flags & VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT) {
      alloc_flags |= VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;
    }
  } else {
    if (create_info.usage == ImageUsage::General) {
      // can't use srgb images for storage. wouldn't want to anyway
      auto storage_usage =
          (format_is_color(create_info.format) && !format_is_srgb(create_info.format))
              ? VK_IMAGE_USAGE_STORAGE_BIT
              : 0;
      usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
              VK_IMAGE_USAGE_SAMPLED_BIT | storage_usage | attachment_usage;

    } else if (create_info.usage == ImageUsage::AttachmentReadOnly) {
      // dedicated memory for attachment textures
      alloc_flags |= VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;
      // can't copy to attachment images, 99% want to sample them as well
      usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | attachment_usage;

    } else {  // ReadOnly
      // copy to/from, sample
      usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
              VK_IMAGE_USAGE_SAMPLED_BIT;
    }
  }

  VmaAllocationCreateInfo alloc_create_info{
      .flags = alloc_flags,
      .usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
  };
  VkImageCreateFlags img_create_flags{};
  if (create_info.view_type == VK_IMAGE_VIEW_TYPE_CUBE ||
      create_info.view_type == VK_IMAGE_VIEW_TYPE_CUBE_ARRAY) {
    img_create_flags |= VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
  }

  VkImageCreateInfo info{.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
                         .flags = img_create_flags,
                         .imageType = vkviewtype_to_img_type(create_info.view_type),
                         .format = create_info.format,
                         .extent = create_info.extent,
                         .mipLevels = create_info.mip_levels,
                         .arrayLayers = create_info.array_layers,
                         .samples = create_info.samples,
                         .tiling = VK_IMAGE_TILING_OPTIMAL,
                         .usage = usage,
                         .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED};
  VK_CHECK(vmaCreateImage(vk2::get_device().allocator(), &info, &alloc_create_info, &image,
                          &allocation, nullptr));
  if (!image) {
    return;
  }

  VkImageAspectFlags aspect = VK_IMAGE_ASPECT_NONE;
  if (format_is_color(create_info.format)) {
    aspect |= VK_IMAGE_ASPECT_COLOR_BIT;
  }
  if (format_is_depth(create_info.format)) {
    aspect |= VK_IMAGE_ASPECT_DEPTH_BIT;
  }
  if (format_is_stencil(create_info.format)) {
    aspect |= VK_IMAGE_ASPECT_STENCIL_BIT;
  }

#ifndef NDEBUG
  if (create_info.name.size()) {
    VkDebugUtilsObjectNameInfoEXT name_info = {
        .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT,
        .objectType = VK_OBJECT_TYPE_IMAGE,
        .objectHandle = (u64)image,
        .pObjectName = create_info.name.c_str()};
    vkSetDebugUtilsObjectNameEXT(get_device().device(), &name_info);
  }
#endif
  if (create_info.make_view) {
    default_view = vk2::get_device().create_image_view_holder(
        *this, ImageViewCreateInfo{.format = create_info.format,
                                   .range =
                                       {
                                           .aspectMask = aspect,
                                           .baseMipLevel = 0,
                                           .levelCount = VK_REMAINING_MIP_LEVELS,
                                           .baseArrayLayer = 0,
                                           .layerCount = VK_REMAINING_ARRAY_LAYERS,
                                       },
                                   .components = {}});
  }
}

ImageViewHandle Device::create_image_view(const Image2& image, const ImageViewCreateInfo& info) {
  VkImageViewType view_type =
      info.view_type != VK_IMAGE_VIEW_TYPE_MAX_ENUM ? info.view_type : image.create_info.view_type;
  auto view_info = VkImageViewCreateInfo{.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                                         .image = image.image,
                                         .viewType = view_type,
                                         .format = info.format,
                                         .components = info.components,
                                         .subresourceRange = info.range};
  ImageView2 view;
  VK_CHECK(vkCreateImageView(vk2::get_device().device(), &view_info, nullptr, &view.view));

  // can only be a storage image if color and general usage
  if ((format_is_color(info.format) && (image.create_info.override_usage_flags == 0 &&
                                        image.create_info.usage == ImageUsage::General)) ||
      ((image.create_info.override_usage_flags & VK_IMAGE_USAGE_STORAGE_BIT) != 0)) {
    view.storage_image_resource_info =
        BindlessResourceAllocator::get().allocate_storage_img_descriptor(view.view,
                                                                         VK_IMAGE_LAYOUT_GENERAL);
  }
  if (image.usage & VK_IMAGE_USAGE_SAMPLED_BIT) {
    view.sampled_image_resource_info =
        BindlessResourceAllocator::get().allocate_sampled_img_descriptor(
            view.view, VK_IMAGE_LAYOUT_READ_ONLY_OPTIMAL);
  }
  return img_view_pool_.alloc(view);
}

void Device::destroy(ImageHandle handle) { img_pool_.destroy(handle); }

void Device::destroy(ImageViewHandle handle) { img_view_pool_.destroy(handle); }

Image2* Device::get_image(ImageHandle handle) { return img_pool_.get(handle); }
ImageView2* Device::get_image_view(ImageViewHandle handle) { return img_view_pool_.get(handle); }

Holder<ImageViewHandle> Device::create_image_view_holder(const Image2& image,
                                                         const ImageViewCreateInfo& info) {
  return Holder<ImageViewHandle>{this, create_image_view(image, info)};
}
Holder<ImageHandle> Device::create_image_holder(const ImageCreateInfo& info) {
  return Holder<ImageHandle>{this, create_image(info)};
}

Image2::Image2(Image2&& other) noexcept
    : create_info(std::move(other.create_info)),
      image(std::exchange(other.image, nullptr)),
      usage(other.usage),
      default_view(std::exchange(other.default_view, Holder<ImageViewHandle>{})),
      allocation(std::exchange(other.allocation, nullptr)) {}

Image2& Image2::operator=(Image2&& other) noexcept {
  if (&other != this) {
    create_info = std::move(other.create_info);
    image = std::exchange(other.image, nullptr);
    usage = other.usage;
    default_view = std::exchange(other.default_view, Holder<ImageViewHandle>{});
    allocation = std::exchange(other.allocation, nullptr);
  }
  return *this;
}

}  // namespace gfx::vk2
