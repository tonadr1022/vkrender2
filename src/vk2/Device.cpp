#include "Device.hpp"

#include <imgui/backends/imgui_impl_glfw.h>
#include <imgui/backends/imgui_impl_vulkan.h>
#include <imgui/imgui.h>
#include <volk.h>
#include <vulkan/vk_enum_string_helper.h>
#include <vulkan/vulkan_core.h>

#include <cassert>
#include <tracy/Tracy.hpp>

#include "BindlessResourceAllocator.hpp"
#include "Common.hpp"
#include "GLFW/glfw3.h"
#include "PipelineManager.hpp"
#include "Types.hpp"
#include "VkBootstrap.h"
#include "VkCommon.hpp"
#include "core/Logger.hpp"
#include "vk2/VkTypes.hpp"

namespace {

#ifndef NDEBUG
#define DEBUG_VK_OBJECT_NAMES 1
#endif
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

// TODO: let pool have ctx that can destroy
template <>
void destroy(gfx::ImageHandle data) {
  gfx::get_device().destroy(data);
}

template <>
void destroy(gfx::BufferHandle data) {
  gfx::get_device().destroy(data);
}

template <>
void destroy(gfx::ImageViewHandle data) {
  gfx::get_device().destroy(data);
}

template <>
void destroy(gfx::SamplerHandle data) {
  gfx::get_device().destroy(data);
}

namespace gfx {}  // namespace gfx

namespace gfx {

namespace {
Device* g_device{};
}

void Device::init(const CreateInfo& info) {
  g_device = new Device;
  g_device->init_impl(info);
}

void Device::destroy() {
  assert(g_device);
  delete g_device;
  g_device = nullptr;
}

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
    instance_builder.set_minimum_instance_version(1, 2, 0)
        .set_app_name(info.app_name)
        .require_api_version(1, 2, 0);

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

  vkb::PhysicalDeviceSelector phys_builder(instance_, surface_);
  supported_features12_.bufferDeviceAddress = true;
  supported_features12_.descriptorIndexing = true;
  supported_features12_.runtimeDescriptorArray = true;
  supported_features12_.shaderStorageImageArrayNonUniformIndexing = true;
  supported_features12_.shaderUniformBufferArrayNonUniformIndexing = true;
  supported_features12_.shaderSampledImageArrayNonUniformIndexing = true;
  supported_features12_.shaderStorageBufferArrayNonUniformIndexing = true;
  supported_features12_.shaderInputAttachmentArrayNonUniformIndexing = true;
  supported_features12_.shaderUniformTexelBufferArrayNonUniformIndexing = true;
  supported_features12_.descriptorBindingUniformBufferUpdateAfterBind = true;
  supported_features12_.descriptorBindingStorageImageUpdateAfterBind = true;
  supported_features12_.descriptorBindingSampledImageUpdateAfterBind = true;
  supported_features12_.descriptorBindingStorageBufferUpdateAfterBind = true;
  supported_features12_.descriptorBindingUpdateUnusedWhilePending = true;
  supported_features12_.descriptorBindingPartiallyBound = true;
  supported_features12_.descriptorBindingVariableDescriptorCount = true;
  supported_features12_.runtimeDescriptorArray = true;
  supported_features12_.timelineSemaphore = true;
  VkPhysicalDeviceFeatures features{};
  features.shaderStorageImageWriteWithoutFormat = true;
  features.depthClamp = true;
  features.shaderInt64 = true;
  features.multiDrawIndirect = true;
  VkPhysicalDeviceVulkan11Features features11{
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
  features11.shaderDrawParameters = true;

  VkPhysicalDeviceDynamicRenderingFeatures dynamic_rendering_features{};
  dynamic_rendering_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES;
  dynamic_rendering_features.dynamicRendering = VK_TRUE;
  VkPhysicalDeviceExtendedDynamicStateFeaturesEXT extended_dynamic_state_features{};
  extended_dynamic_state_features.sType =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_FEATURES_EXT;
  extended_dynamic_state_features.extendedDynamicState = VK_TRUE;
  VkPhysicalDeviceSynchronization2Features sync2_features{};
  sync2_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES;
  sync2_features.synchronization2 = VK_TRUE;
  std::vector<const char*> extensions{
      {VK_KHR_FORMAT_FEATURE_FLAGS_2_EXTENSION_NAME, VK_KHR_COPY_COMMANDS_2_EXTENSION_NAME,
       VK_EXT_EXTENDED_DYNAMIC_STATE_EXTENSION_NAME, VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME,
       VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME}};

  // features12.drawIndirectCount = true;

  auto pr = phys_builder.set_minimum_version(1, 2)
                .set_required_features_12(supported_features12_)
                .set_required_features_11(features11)
                .add_required_extensions(extensions)
                .allow_any_gpu_device_type(false)
                .prefer_gpu_device_type(vkb::PreferredDeviceType::discrete)
                .add_required_extension_features(dynamic_rendering_features)
                .add_required_extension_features(sync2_features)
                .add_required_extension_features(extended_dynamic_state_features)
                .set_required_features(features)
                .select();
  if (pr.has_value()) {
    vkb_phys_device_ = pr.value();
  } else {
    LCRITICAL("Failed to select physical device: {}", pr.error().message());
    exit(1);
  }
  {
    VkPhysicalDeviceVulkan12Features features{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    features.drawIndirectCount = true;
    if (vkb_phys_device_.enable_extension_features_if_present(features)) {
      supported_features12_.drawIndirectCount = true;
    }

    // fix validation error due to buffer device address use in spirv causing weird error
    vkb_phys_device_.enable_extension_if_present(
        VK_KHR_SHADER_RELAXED_EXTENDED_INSTRUCTION_EXTENSION_NAME);
  }

  {
    const auto& props = vkb_phys_device_.properties;
    LINFO("[Device Info]\nName: {}\nType: {}\nAPI Version {}.{}.{}", props.deviceName,
          props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU     ? "Discrete"
          : props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU ? "Integrated"
                                                                       : "CPU",
          VK_API_VERSION_MAJOR(props.apiVersion), VK_API_VERSION_MINOR(props.apiVersion),
          VK_API_VERSION_PATCH(props.apiVersion), props.driverVersion);
  }

  vkb::DeviceBuilder dev_builder(vkb_phys_device_);
  auto dev_ret = dev_builder.build();
  if (!dev_ret) {
    LCRITICAL("Failed to acquire logical device: {}", dev_ret.error().message());
    exit(1);
  }
  vkb_device_ = std::move(dev_ret.value());
  device_ = vkb_device_.device;
  assert(device_);

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

  // TODO: set debug name functions
#ifdef DEBUG_VK_OBJECT_NAMES
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
    create_swapchain(swapchain_, vk2::SwapchainDesc{.width = static_cast<u32>(w),
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
  assert(device_ && device());
  ResourceAllocator::init(device_, allocator_);
  null_sampler_ = get_or_create_sampler({.address_mode = AddressMode::MirroredRepeat});
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

void Device::destroy(SamplerHandle handle) {
  if (auto* samp = sampler_pool_.get(handle); samp) {
    vkDestroySampler(device_, samp->sampler_, nullptr);
  }
  sampler_pool_.destroy(handle);
}

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
    cmd.transfer_cmd_pool = device_->create_command_pool(QueueType::Graphics, 0);
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
    // TODO: remove
    ImGui_ImplVulkan_LoadFunctions(
        vkb_phys_device_.properties.apiVersion,
        [](const char* functionName, void* vulkanInstance) {
          return vkGetInstanceProcAddr(*static_cast<VkInstance*>(vulkanInstance), functionName);
        },
        reinterpret_cast<void*>(&instance_));
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
    init_info.Instance = instance_;
    init_info.PhysicalDevice = vkb_phys_device_.physical_device;
    init_info.Device = device_;
    init_info.Queue = get_queue(QueueType::Graphics).queue;
    init_info.DescriptorPool = imgui_descriptor_pool_;
    // TODO: fix
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
                        .format = vk2::vkformat_to_format(swapchain_.format)};
}

VkImage Device::get_swapchain_img(u32 idx) const { return swapchain_.imgs[idx]; }

VkImage Device::acquire_next_image() {
  ZoneScoped;
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
        ResourceAllocator::get().enqueue_delete_sempahore(sem);
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

VkFence Device::allocate_fence(bool reset) {
  if (!free_fences_.empty()) {
    VkFence f = free_fences_.back();
    free_fences_.pop_back();
    if (reset) {
      VK_CHECK(vkResetFences(device_, 1, &f));
    }
    return f;
  }

  VkFenceCreateInfo info{.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
                         .flags = VK_FENCE_CREATE_SIGNALED_BIT};
  VkFence fence;
  VK_CHECK(vkCreateFence(device_, &info, nullptr, &fence));
  // TODO: maybe not this
  if (reset) {
    VK_CHECK(vkResetFences(device_, 1, &fence));
  }
  return fence;
}

void Device::free_fence(VkFence fence) { free_fences_.push_back(fence); }

Device::~Device() {
  ZoneScoped;
  PipelineManager::shutdown();
  ResourceAllocator::get().set_frame_num(UINT32_MAX, 0);
  for (auto& it : sampler_cache_) {
    destroy(it.second.first);
  }
  ResourceAllocator::shutdown();

  for (auto& d : per_frame_data_) {
    vkDestroySemaphore(device_, d.render_semaphore, nullptr);
    vkDestroyFence(device_, d.render_fence, nullptr);
  }
  for (auto& f : free_fences_) {
    vkDestroyFence(device_, f, nullptr);
  }

  ImGui_ImplVulkan_Shutdown();
  vkDestroyDescriptorPool(device_, imgui_descriptor_pool_, nullptr);
  swapchain_.destroy(device_);
  {
    ZoneScopedN("shutdown base");
    vmaDestroyAllocator(allocator_);
    vkb::destroy_device(vkb_device_);
    vkb::destroy_surface(instance_, surface_);
    volkFinalize();
    vkb::destroy_instance(instance_);
  }
}

void Device::wait_idle() { vkDeviceWaitIdle(device_); }

bool Device::is_supported(DeviceFeature feature) const {
  switch (feature) {
    case DeviceFeature::DrawIndirectCount:
      return supported_features12_.drawIndirectCount;
  }
  return true;
}

void Device::set_name(const char* name, u64 handle, VkObjectType type) {
  VkDebugUtilsObjectNameInfoEXT name_info = {
      .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT,
      .objectType = type,
      .objectHandle = handle,
      .pObjectName = name};
  vkSetDebugUtilsObjectNameEXT(device_, &name_info);
}

void Device::set_name(VkPipeline pipeline, const char* name) {
#ifdef DEBUG_VK_OBJECT_NAMES
  if (pipeline) {
    set_name(name, reinterpret_cast<u64>(pipeline), VK_OBJECT_TYPE_PIPELINE);
  }
#else
  (void)pipeline;
  (void)name;
#endif
}

namespace {
VkFilter get_filter(FilterMode mode) {
  switch (mode) {
    case gfx::FilterMode::Linear:
      return VK_FILTER_LINEAR;
    case gfx::FilterMode::Nearest:
      return VK_FILTER_NEAREST;
    default:
      assert(0);
      return VK_FILTER_MAX_ENUM;
  }
}
VkSamplerMipmapMode get_mipmap_mode(FilterMode mode) {
  switch (mode) {
    default:
    case gfx::FilterMode::Linear:
      return VK_SAMPLER_MIPMAP_MODE_LINEAR;
    case gfx::FilterMode::Nearest:
      return VK_SAMPLER_MIPMAP_MODE_NEAREST;
  }
}

VkSamplerAddressMode get_address_mode(AddressMode mode) {
  switch (mode) {
    default:
    case AddressMode::Repeat:
      return VK_SAMPLER_ADDRESS_MODE_REPEAT;
    case AddressMode::MirroredRepeat:
      return VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
    case AddressMode::ClampToEdge:
      return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    case AddressMode::ClampToBorder:
      return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    case AddressMode::MirrorClampToEdge:
      return VK_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE;
  }
}

}  // namespace

SamplerHandle Device::get_or_create_sampler(const SamplerCreateInfo& info) {
  ZoneScoped;
  auto h =
      std::make_tuple(info.address_mode, info.min_filter, info.mag_filter, info.anisotropy_enable,
                      info.max_anisotropy, info.compare_enable, info.compare_op);
  auto hash = vk2::detail::hashing::hash<decltype(h)>{}(h);
  auto it = sampler_cache_.find(hash);
  if (it != sampler_cache_.end()) {
    // ref cnt
    it->second.second++;
    return it->second.first;
  }
  VkSamplerCreateInfo cinfo{.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
                            .magFilter = get_filter(info.mag_filter),
                            .minFilter = get_filter(info.min_filter),
                            .mipmapMode = get_mipmap_mode(info.mipmap_mode),
                            .addressModeU = get_address_mode(info.address_mode),
                            .addressModeV = get_address_mode(info.address_mode),
                            .addressModeW = get_address_mode(info.address_mode),
                            .anisotropyEnable = info.anisotropy_enable,
                            .maxAnisotropy = info.max_anisotropy,
                            .compareEnable = info.compare_enable,
                            .compareOp = static_cast<VkCompareOp>(info.compare_op),
                            .minLod = info.min_lod,
                            .maxLod = info.max_lod,
                            .borderColor = static_cast<VkBorderColor>(info.border_color)};
  auto handle = sampler_pool_.alloc();
  Sampler* sampler = sampler_pool_.get(handle);
  VK_CHECK(vkCreateSampler(device_, &cinfo, nullptr, &sampler->sampler_));
  assert(sampler->sampler_);
  sampler->bindless_info_ = ResourceAllocator::get().allocate_sampler_descriptor(sampler->sampler_);
  sampler_cache_.emplace(hash, std::make_pair(handle, 1));
  return handle;
}

u32 Device::get_bindless_idx(SamplerHandle sampler) {
  if (auto* samp = sampler_pool_.get(sampler); samp != nullptr) {
    return samp->bindless_info_.handle;
  }
  return 0;
}

VkSampler Device::get_sampler_vk(SamplerHandle sampler) {
  if (auto* samp = sampler_pool_.get(sampler); samp != nullptr) {
    return samp->sampler_;
  }
  return nullptr;
}

}  // namespace gfx
