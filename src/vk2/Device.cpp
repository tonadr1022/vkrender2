#include "Device.hpp"

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_vulkan.h>
#include <volk.h>
#include <vulkan/vulkan_core.h>

#include <cassert>
#include <tracy/Tracy.hpp>

#include "Common.hpp"
#include "GLFW/glfw3.h"
#include "Logger.hpp"
#include "Types.hpp"
#include "VkBootstrap.h"
#include "VkCommon.hpp"
#include "vk2/VkTypes.hpp"

namespace {

#ifdef DEBUG_CALLBACK_ENABLED
VKAPI_ATTR VkBool32 VKAPI_CALL
debug_callback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
               VkDebugUtilsMessageTypeFlagsEXT messageType,
               const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData, void*) {
  const auto* ms = vkb::to_string_message_severity(messageSeverity);
  const auto* mt = vkb::to_string_message_type(messageType);
  if (messageType & VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT) {
    LERROR("[{}: {}] - {}\n{}\n", ms, mt, pCallbackData->pMessageIdName, pCallbackData->pMessage);
  } else {
    LERROR("[{}: {}]\n{}\n", ms, mt, pCallbackData->pMessage);
  }

  // exit(1);
  return VK_FALSE;
}

#endif

}  // namespace

template <>
void destroy(gfx::ImageHandle data) {
  gfx::vk2::get_device().destroy(data);
}

template <>
void destroy(gfx::BufferHandle data) {
  gfx::vk2::get_device().destroy(data);
}

template <>
void destroy(gfx::ImageViewHandle data) {
  gfx::vk2::get_device().destroy(data);
}
namespace gfx {}  // namespace gfx

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
  VK_CHECK(volkInitialize());
  window_ = info.window;
  {
    ZoneScopedN("instance init");
    vkb::InstanceBuilder instance_builder;
    instance_builder
        .set_minimum_instance_version(vk2::min_api_version_major, vk2::min_api_version_minor, 0)
        .set_app_name(info.app_name)
        .require_api_version(vk2::min_api_version_major, vk2::min_api_version_minor, 0);

#ifdef DEBUG_CALLBACK_ENABLED
    instance_builder.set_debug_callback(debug_callback);
#endif
#ifdef VALIDATION_LAYERS_ENABLED
    instance_builder.request_validation_layers(true);
#endif

#if defined(__APPLE__)
    instance_builder.add_validation_feature_disable(VK_VALIDATION_FEATURE_DISABLE_SHADERS_EXT);
    instance_builder.add_validation_feature_disable(
        VK_VALIDATION_FEATURE_DISABLE_SHADER_VALIDATION_CACHE_EXT);
#endif
    auto instance_ret = instance_builder.build();
    if (!instance_ret) {
      LCRITICAL("Failed to acquire Vulkan Instance: {}", instance_ret.error().message());
      exit(1);
    }
    instance_ = instance_ret.value();
  }

  {
    ZoneScopedN("volk");
    volkLoadInstance(instance_.instance);
  }

  {
    ZoneScopedN("glfw init");
    glfwCreateWindowSurface(instance_.instance, info.window, nullptr, &surface_);
    if (!surface_) {
      LCRITICAL("Failed to create surface");
      exit(1);
    }
  }

  vkb::PhysicalDeviceSelector phys_selector(instance_, surface_);
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
  features.shaderInt64 = true;
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
  std::vector<const char*> extensions{
      {VK_KHR_FORMAT_FEATURE_FLAGS_2_EXTENSION_NAME, VK_KHR_COPY_COMMANDS_2_EXTENSION_NAME,
       VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME, VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME}};

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
      .physicalDevice = get_physical_device(),
      .device = device(),
      .pVulkanFunctions = &vma_vulkan_func,
      .instance = instance_,
  };

  VK_CHECK(vmaCreateAllocator(&allocator_info, &allocator_));

  volkLoadDevice(device_);

  for (u64 i = 0; i < vkb_device_.queue_families.size(); i++) {
    if (vkb_device_.queue_families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
      vkGetDeviceQueue(device_, i, 0, &queues_[(u32)QueueType::Graphics].queue);
      queues_[(u32)QueueType::Graphics].family_idx = i;
      break;
    }
  }

  // uint32_t extension_count = 0;
  // vkEnumerateDeviceExtensionProperties(vk2::get_device().phys_device(), nullptr,
  // &extension_count,
  //                                      nullptr);
  // std::vector<VkExtensionProperties> extensions(extension_count);
  // vkEnumerateDeviceExtensionProperties(vk2::get_device().phys_device(), nullptr,
  // &extension_count,
  //                                      extensions.data());
  //
  // for (const auto& ext : extensions) {
  //   LINFO("{} {}", ext.extensionName, ext.specVersion);
  // }

  for (u64 i = 0; i < vkb_device_.queue_families.size(); i++) {
    if (i == queues_[(u32)QueueType::Graphics].family_idx) continue;
    if (vkb_device_.queue_families[i].queueFlags & VK_QUEUE_TRANSFER_BIT) {
      vkGetDeviceQueue(device_, i, 0, &queues_[(u32)QueueType::Transfer].queue);
      queues_[(u32)QueueType::Transfer].family_idx = i;
      break;
    }
  }

#ifndef NDEBUG
  VkDebugUtilsObjectNameInfoEXT name_info = {
      .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT,
      .objectType = VK_OBJECT_TYPE_QUEUE,
      .objectHandle = reinterpret_cast<uint64_t>(queues_[(u32)QueueType::Graphics].queue),
      .pObjectName = "GraphicsQueue"};
  vkSetDebugUtilsObjectNameEXT(device_, &name_info);
#endif
  {
    int w, h;
    glfwGetWindowSize(window_, &w, &h);
    swapchain_.surface = surface_;
    create_swapchain(swapchain_, SwapchainDesc{.width = static_cast<u32>(w),
                                               .height = static_cast<u32>(h),
                                               .buffer_count = frames_in_flight,
                                               .vsync = info.vsync});
  }
  {
    for (u32 i = 0; i < frames_in_flight; i++) {
      auto& d = per_frame_data_.emplace_back();
      d.render_fence = create_fence();
      d.render_semaphore = create_semaphore();
    }
  }
}

void Device::destroy_impl() {
  for (auto& d : per_frame_data_) {
    vkDestroySemaphore(device_, d.render_semaphore, nullptr);
    vkDestroyFence(device_, d.render_fence, nullptr);
  }

  ImGui_ImplVulkan_Shutdown();
  vkDestroyDescriptorPool(device_, imgui_descriptor_pool_, nullptr);
  swapchain_.destroy(device_);
  vmaDestroyAllocator(allocator_);
  vkb::destroy_device(vkb_device_);
  vkb::destroy_surface(instance_, surface_);
  volkFinalize();
  vkb::destroy_instance(instance_);
}

Device& get_device() { return Device::get(); }

VkCommandPool Device::create_command_pool(QueueType type, VkCommandPoolCreateFlags flags) const {
  VkCommandPoolCreateInfo info{.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                               .flags = flags,
                               .queueFamilyIndex = queues_[(u32)type].family_idx};
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
  VkSemaphoreTypeCreateInfo cinfo{};
  cinfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
  cinfo.pNext = nullptr;
  cinfo.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
  cinfo.initialValue = 0;
  VkSemaphoreCreateInfo info{
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
      .pNext = timeline ? &cinfo : nullptr,
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
  return img_pool_.alloc(create_info);
}
ImageViewHandle Device::create_image_view(const Image& image, const ImageViewCreateInfo& info) {
  return img_view_pool_.alloc(image, info);
}

void Device::destroy(ImageHandle handle) { img_pool_.destroy(handle); }
void Device::destroy(ImageViewHandle handle) { img_view_pool_.destroy(handle); }
void Device::destroy(BufferHandle handle) { buffer_pool_.destroy(handle); }

Holder<ImageHandle> Device::create_image_holder(const ImageCreateInfo& info) {
  return Holder<ImageHandle>{create_image(info)};
}
Holder<ImageViewHandle> Device::create_image_view_holder(const Image& image,
                                                         const ImageViewCreateInfo& info) {
  return Holder<ImageViewHandle>{create_image_view(image, info)};
}

void Device::destroy_resources() {
  img_pool_.clear();
  img_view_pool_.clear();
  buffer_pool_.clear();
}

void Device::on_imgui() const {
  auto pool_stats = [](const char* pool_name, const auto& pool) {
    ImGui::Text("%s: \nActive: %u\nCreated: %zu\nDestroyed: %zu", pool_name, pool.size(),
                pool.get_num_created(), pool.get_num_destroyed());
  };
  pool_stats("Images", img_pool_);
  pool_stats("Image Views", img_view_pool_);
  pool_stats("Buffers", buffer_pool_);
}

Holder<BufferHandle> Device::create_buffer_holder(const BufferCreateInfo& info) {
  return Holder<BufferHandle>{buffer_pool_.alloc(info)};
}
BufferHandle Device::create_buffer(const BufferCreateInfo& info) {
  return buffer_pool_.alloc(info);
}

CopyAllocator::CopyCmd CopyAllocator::allocate(u64 size) {
  CopyCmd cmd;
  for (size_t i = 0; i < free_copy_cmds_.size(); i++) {
    auto& free_cmd = free_copy_cmds_[i];
    if (free_cmd.is_valid()) {
      auto* staging_buf = get_device().get_buffer(free_cmd.staging_buffer);
      assert(staging_buf);
      if (staging_buf->size() >= size) {
        cmd = free_copy_cmds_[i];
        std::swap(free_copy_cmds_[i], *free_copy_cmds_.end());
        free_copy_cmds_.pop_back();
        break;
      }
    }
  }
  // make new
  if (!cmd.is_valid()) {
    // cmd.transfer_cmd_pool = get_device().create_command_pool();
  }
  return cmd;
}

void CopyAllocator::submit(CopyCmd cmd) { free_copy_cmds_.emplace_back(cmd); }

void CopyAllocator::destroy() {
  std::scoped_lock lock(free_list_mtx_);
  for (auto& el : free_copy_cmds_) {
    vkDestroyFence(get_device().device(), el.fence, nullptr);
    vkDestroyCommandPool(get_device().device(), el.transfer_cmd_pool, nullptr);
  }
  free_copy_cmds_.clear();
}

void Device::queue_submit(QueueType type, std::span<VkSubmitInfo2> submits, VkFence fence) {
  VK_CHECK(vkQueueSubmit2KHR(queues_[(u32)type].queue, submits.size(), submits.data(), fence));
}
void Device::queue_submit(QueueType type, std::span<VkSubmitInfo2> submits) {
  VK_CHECK(vkQueueSubmit2KHR(queues_[(u32)type].queue, submits.size(), submits.data(),
                             curr_frame().render_fence));
}

void Device::init_imgui() {
  ZoneScopedN("init imgui");
  ImGui::CreateContext();
  ImGuiIO& io = ImGui::GetIO();
  (void)io;
  if (!ImGui_ImplGlfw_InitForVulkan(window_, true)) {
    LCRITICAL("ImGui_ImplGlfw_InitForVulkan failed");
    exit(1);
  }
  {
    VkInstance instance = vk2::get_device().get_instance();
    // TODO: remove
    ImGui_ImplVulkan_LoadFunctions(
        [](const char* functionName, void* vulkanInstance) {
          return vkGetInstanceProcAddr(*static_cast<VkInstance*>(vulkanInstance), functionName);
        },
        reinterpret_cast<void*>(&instance));
    // create descriptor pool
    VkDescriptorPoolSize pool_sizes[] = {{VK_DESCRIPTOR_TYPE_SAMPLER, 1000},
                                         {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1000},
                                         {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1000},
                                         {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1000},
                                         {VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 1000},
                                         {VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 1000},
                                         {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1000},
                                         {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1000},
                                         {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1000},
                                         {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 1000},
                                         {VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, 1000}};
    VkDescriptorPoolCreateInfo pool_info{};
    pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    pool_info.maxSets = 1000;
    pool_info.poolSizeCount = std::size(pool_sizes);
    pool_info.pPoolSizes = pool_sizes;

    VK_CHECK(vkCreateDescriptorPool(device_, &pool_info, nullptr, &imgui_descriptor_pool_));

    ImGui_ImplVulkan_InitInfo init_info{};
    init_info.Instance = instance;
    init_info.PhysicalDevice = vkb_phys_device_.physical_device;
    init_info.Device = device_;
    init_info.Queue = vk2::get_device().get_queue(vk2::QueueType::Graphics).queue;
    init_info.DescriptorPool = imgui_descriptor_pool_;
    init_info.MinImageCount = 3;
    init_info.ImageCount = 3;
    init_info.UseDynamicRendering = true;

    init_info.PipelineRenderingCreateInfo = {};
    init_info.PipelineRenderingCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    init_info.PipelineRenderingCreateInfo.colorAttachmentCount = 1;
    init_info.PipelineRenderingCreateInfo.pColorAttachmentFormats = &swapchain_.format;
    init_info.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
    if (!ImGui_ImplVulkan_Init(&init_info)) {
      LCRITICAL("ImGui_ImplVulkan_Init failed");
      exit(1);
    }
    ImGui_ImplVulkan_CreateFontsTexture();
  }
}
AttachmentInfo Device::get_swapchain_info() const {
  return AttachmentInfo{.dims = {swapchain_.dims.x, swapchain_.dims.y, 1},
                        .format = vkformat_to_format(swapchain_.format)};
}

VkImage Device::get_swapchain_img(u32 idx) const { return swapchain_.imgs[idx]; }

VkImage Device::acquire_next_image() {
  swapchain_.acquire_semaphore_idx =
      (swapchain_.acquire_semaphore_idx + 1) % swapchain_.imgs.size();
  VkResult acquire_next_image_result;
  // acquire next image
  do {
    acquire_next_image_result =
        vkAcquireNextImageKHR(device_, swapchain_.swapchain, 1000000000,
                              swapchain_.acquire_semaphores[swapchain_.acquire_semaphore_idx],
                              nullptr, &swapchain_.curr_swapchain_idx);
    if (acquire_next_image_result == VK_TIMEOUT) {
      LERROR("vkAcquireNextImageKHR resulted in VK_TIMEOUT, retring");
    }
  } while (acquire_next_image_result == VK_TIMEOUT);

  if (acquire_next_image_result != VK_SUCCESS) {
    // handle outdated error
    if (acquire_next_image_result == VK_SUBOPTIMAL_KHR ||
        acquire_next_image_result == VK_ERROR_OUT_OF_DATE_KHR) {
      // recreate new semaphore
      // need to make new semaphore since wsi doesn't unsignal it
      for (auto& sem : swapchain_.acquire_semaphores) {
        vkDestroySemaphore(device_, sem, nullptr);
      }
      LINFO("new semapohres create");
      swapchain_.acquire_semaphores.clear();
      int x, y;
      glfwGetWindowSize(window_, &x, &y);
      auto desc = swapchain_.desc;
      desc.width = x;
      desc.height = y;
      create_swapchain(swapchain_, desc);
      return acquire_next_image();
    }
    assert(0);
  }
  return swapchain_.imgs[swapchain_.curr_swapchain_idx];
}

void Device::submit_to_graphics_queue() {
  VkResult res;
  VkPresentInfoKHR info{.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
                        .waitSemaphoreCount = 1,
                        .pWaitSemaphores = &curr_frame().render_semaphore,
                        .swapchainCount = 1,
                        .pSwapchains = &swapchain_.swapchain,
                        .pImageIndices = &swapchain_.curr_swapchain_idx,
                        .pResults = &res};
  VkResult present_result = vkQueuePresentKHR(queues_[(u32)QueueType::Graphics].queue, &info);
  if (present_result == VK_ERROR_OUT_OF_DATE_KHR || present_result == VK_SUBOPTIMAL_KHR) {
  } else {
    VK_CHECK(present_result);
  }

  curr_frame_num_++;
}
void Device::begin_frame() {
  // wait for fence
  u64 timeout = 1000000000;
  vkWaitForFences(device_, 1, &curr_frame().render_fence, true, timeout);
  VK_CHECK(vkResetFences(device_, 1, &curr_frame().render_fence));
}

}  // namespace gfx::vk2
