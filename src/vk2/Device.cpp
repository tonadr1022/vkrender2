#include "Device.hpp"

#include <imgui/backends/imgui_impl_glfw.h>
#include <imgui/backends/imgui_impl_vulkan.h>
#include <imgui/imgui.h>
#include <volk.h>
#include <vulkan/vk_enum_string_helper.h>
#include <vulkan/vulkan_core.h>

#include <cassert>
#include <tracy/Tracy.hpp>

#include "CommandEncoder.hpp"
#include "Common.hpp"
#include "GLFW/glfw3.h"
#include "PipelineManager.hpp"
#include "Types.hpp"
#include "VkBootstrap.h"
#include "VkCommon.hpp"
#include "core/Logger.hpp"
#include "vk2/Resource.hpp"
#include "vk2/Texture.hpp"
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
    std::vector<const char*> exts{VK_KHR_SHADER_NON_SEMANTIC_INFO_EXTENSION_NAME,
                                  VK_KHR_SHADER_RELAXED_EXTENDED_INSTRUCTION_EXTENSION_NAME};
    vkb_phys_device_.enable_extensions_if_present(exts);
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
      .physicalDevice = vkb_phys_device_.physical_device,
      .device = device_,
      .pVulkanFunctions = &vma_vulkan_func,
      .instance = instance_,
  };

#ifndef NDEBUG
  allocator_info.flags |= VMA_ALLOCATOR_CREATE_EXT_MEMORY_BUDGET_BIT;
#endif
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

  init_bindless();
  null_sampler_ = get_or_create_sampler({.address_mode = AddressMode::MirroredRepeat});
  // default pipeline layout
  VkPushConstantRange default_range{.stageFlags = VK_SHADER_STAGE_ALL, .offset = 0, .size = 128};
  VkDescriptorSetLayout layouts[] = {main_set_layout_, main_set2_layout_};
  VkPipelineLayoutCreateInfo pipeline_info{.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                                           .setLayoutCount = COUNTOF(layouts),
                                           .pSetLayouts = layouts,
                                           .pushConstantRangeCount = 1,
                                           .pPushConstantRanges = &default_range};
  VK_CHECK(vkCreatePipelineLayout(device_, &pipeline_info, nullptr, &default_pipeline_layout_));
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

void Device::destroy(ImageHandle handle) {
  auto* img = img_pool_.get(handle);
  if (img) {
    destroy(*img);
    img_pool_.destroy(handle);
  }
}

void Device::destroy(SamplerHandle handle) {
  if (auto* samp = sampler_pool_.get(handle); samp) {
    vkDestroySampler(device_, samp->sampler_, nullptr);
  }
  sampler_pool_.destroy(handle);
}

void Device::destroy(BufferHandle handle) {
  storage_buffer_delete_q_.emplace_back(handle, curr_frame_num());
}

void Device::on_imgui() const {
  auto pool_stats = [](const char* pool_name, const auto& pool) {
    ImGui::Text("%s: \nActive: %u\nCreated: %zu\nDestroyed: %zu", pool_name, pool.size(),
                pool.get_num_created(), pool.get_num_destroyed());
  };
  pool_stats("Images", img_pool_);
  pool_stats("Buffers", buffer_pool_);
}

Holder<BufferHandle> Device::create_buffer_holder(const BufferCreateInfo& info) {
  return Holder<BufferHandle>{create_buffer(info)};
}

BufferHandle Device::create_buffer(const BufferCreateInfo& cinfo) {
  if (cinfo.size == 0) {
    return {};
  }
  // https://gpuopen-librariesandsdks.github.io/VulkanMemoryAllocator/html/usage_patterns.html
  VmaAllocationCreateInfo alloc_info{.usage = VMA_MEMORY_USAGE_AUTO};
  VkBufferUsageFlags usage{};
  // if no usage, it's 99% chance a staging buffer
  if (cinfo.usage == 0) {
    usage |= VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
  }
  if (cinfo.flags & BufferCreateFlags_HostVisible) {
    alloc_info.flags |= VMA_ALLOCATION_CREATE_MAPPED_BIT |
                        ((cinfo.flags & BufferCreateFlags_HostAccessRandom)
                             ? VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT
                             : VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);
  } else {
    // device
    usage |= VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
  }
  if (cinfo.usage & BufferUsage_Index) {
    usage |= VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
  }
  if (cinfo.usage & BufferUsage_Vertex) {
    usage |= VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
  }
  if (cinfo.usage & BufferUsage_Storage) {
    usage |= VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
             VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
  }
  if (cinfo.usage & BufferUsage_Indirect) {
    usage |= VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
  }

  auto handle = buffer_pool_.alloc();
  auto* buffer = buffer_pool_.get(handle);

  VkBufferCreateInfo buffer_create_info{
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = cinfo.size, .usage = usage};

  VK_CHECK(vmaCreateBuffer(allocator_, &buffer_create_info, &alloc_info, &buffer->buffer_,
                           &buffer->allocation_, &buffer->info_));
  if (buffer->info_.size == 0) {
    return {};
  }
  if (cinfo.usage & BufferUsage_Storage) {
    buffer->resource_info_ = allocate_storage_buffer_descriptor(buffer->buffer_);
  }
  if (usage & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT) {
    VkBufferDeviceAddressInfo info{.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
                                   .buffer = buffer->buffer_};
    buffer->buffer_address_ = vkGetBufferDeviceAddress(get_device().device(), &info);
    assert(buffer->buffer_address_);
  }

  buffer->size_ = cinfo.size;
  buffer->usage_ = cinfo.usage;

  return handle;
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

void CopyAllocator::submit(CopyCmd cmd) {
  // need to transfer ownership?
  free_copy_cmds_.emplace_back(cmd);
}

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
                        .format = vk2::convert_format(swapchain_.format)};
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
        enqueue_delete_sempahore(sem);
      }
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
  flush_deletions();
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
  curr_frame_num_ = UINT32_MAX;
  for (auto& it : sampler_cache_) {
    destroy(it.second.first);
  }

  flush_deletions();

  vkDestroyDescriptorPool(device_, main_pool_, nullptr);
  vkDestroyDescriptorSetLayout(device_, main_set_layout_, nullptr);
  vkDestroyDescriptorSetLayout(device_, main_set2_layout_, nullptr);

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

void Device::set_name(ImageHandle handle, const char* name) {
#ifdef DEBUG_VK_OBJECT_NAMES
  if (auto* img = get_image(handle); img) {
    set_name(name, reinterpret_cast<u64>(img->image()), VK_OBJECT_TYPE_IMAGE);
  }
#else
  (void)handle;
  (void)name;
#endif
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
  sampler->bindless_info_ = allocate_sampler_descriptor(sampler->sampler_);
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

// struct ImageDesc {
//   enum class Type : u8 { One, Two, Three };
//   Type type{Type::Two};
//   Format format{Format::Undefined};
//   uvec3 dims{};
//   u32 mip_levels{1};
//   u32 array_layers{1};
//   u32 sample_count{};
//   BindFlag bind_flags{};
// };

Holder<ImageHandle> Device::create_image_holder(const ImageDesc& desc) {
  return Holder<ImageHandle>{create_image(desc)};
}
ImageHandle Device::create_image(const ImageDesc& desc) {
  VkImageCreateInfo cinfo{.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
  VmaAllocationCreateInfo alloc_create_info{.usage = VMA_MEMORY_USAGE_AUTO};
  if (has_flag(desc.bind_flags, BindFlag::ColorAttachment)) {
    cinfo.usage |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    alloc_create_info.flags |= VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;
  }
  if (has_flag(desc.bind_flags, BindFlag::DepthStencilAttachment)) {
    cinfo.usage |= VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
    alloc_create_info.flags |= VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;
  }
  if (has_flag(desc.bind_flags, BindFlag::ShaderResource)) {
    cinfo.usage |= VK_IMAGE_USAGE_SAMPLED_BIT;
  }
  if (has_flag(desc.bind_flags, BindFlag::Storage)) {
    cinfo.usage |= VK_IMAGE_USAGE_STORAGE_BIT;
  }

  // always copy to/from
  cinfo.usage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;

  switch (desc.type) {
    case ImageDesc::Type::OneD:
      cinfo.imageType = VK_IMAGE_TYPE_1D;
      break;
    case ImageDesc::Type::TwoD:
      cinfo.imageType = VK_IMAGE_TYPE_2D;
      break;
    case ImageDesc::Type::ThreeD:
      cinfo.imageType = VK_IMAGE_TYPE_3D;
      break;
  }

  if (has_flag(desc.misc_flags, ResourceMiscFlag::ImageCube)) {
    cinfo.flags |= VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
  }
  cinfo.arrayLayers = desc.array_layers;
  cinfo.mipLevels = desc.mip_levels;
  cinfo.format = vk2::convert_format(desc.format);
  cinfo.extent = {desc.dims.x, desc.dims.y, desc.dims.z};
  cinfo.tiling = VK_IMAGE_TILING_OPTIMAL;
  cinfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  cinfo.samples = VK_SAMPLE_COUNT_1_BIT;
  if (desc.sample_count == 2) {
    cinfo.samples = VK_SAMPLE_COUNT_2_BIT;
  } else if (desc.sample_count == 4) {
    cinfo.samples = VK_SAMPLE_COUNT_4_BIT;
  } else if (desc.sample_count == 8) {
    cinfo.samples = VK_SAMPLE_COUNT_8_BIT;
  } else if (desc.sample_count == 16) {
    cinfo.samples = VK_SAMPLE_COUNT_16_BIT;
  } else if (desc.sample_count == 32) {
    cinfo.samples = VK_SAMPLE_COUNT_32_BIT;
  } else if (desc.sample_count == 64) {
    cinfo.samples = VK_SAMPLE_COUNT_64_BIT;
  }
  if (desc.usage == Usage::Default) {
    alloc_create_info.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
  }
  auto handle = img_pool_.alloc();
  Image* image = img_pool_.get(handle);
  image->desc_ = desc;
  VK_CHECK(vmaCreateImage(get_device().allocator(), &cinfo, &alloc_create_info, &image->image_,
                          &image->allocation_, nullptr));
  if (!image->image()) {
    return {};
  }

  if (desc.usage == Usage::Default) {
    // depth stencil also needs a subresource
    if (has_flag(desc.bind_flags, BindFlag::ColorAttachment | BindFlag::DepthStencilAttachment)) {
      // attachment view has one mip level/array layer for now
      VkImageViewCreateInfo view_info{.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                                      .image = image->image(),
                                      .format = vk2::convert_format(image->get_desc().format),
                                      .subresourceRange = {
                                          .baseMipLevel = 0,
                                          .levelCount = 1,
                                          .baseArrayLayer = 0,
                                          .layerCount = 1,
                                      }};
      if (format_is_color(image->get_desc().format)) {
        view_info.subresourceRange.aspectMask |= VK_IMAGE_ASPECT_COLOR_BIT;
      }
      if (format_is_depth(image->get_desc().format)) {
        view_info.subresourceRange.aspectMask |= VK_IMAGE_ASPECT_DEPTH_BIT;
      }
      if (format_is_stencil(image->get_desc().format)) {
        view_info.subresourceRange.aspectMask |= VK_IMAGE_ASPECT_STENCIL_BIT;
      }
      view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
      VK_CHECK(vkCreateImageView(device_, &view_info, nullptr, &image->attachment_view_));
    }

    if (has_flag(desc.bind_flags, BindFlag::ShaderResource)) {
      image->sampled_view_ = create_image_view2(handle, SubresourceType::Shader, 0, desc.mip_levels,
                                                0, desc.array_layers);
    }
    if (has_flag(desc.bind_flags, BindFlag::Storage)) {
      image->storage_view_ = create_image_view2(handle, SubresourceType::Storage, 0,
                                                desc.mip_levels, 0, desc.array_layers);
    }
  }

  return handle;
}

i32 Device::create_subresource(ImageHandle image_handle, u32 base_mip_level, u32 level_count,
                               u32 base_array_layer, u32 layer_count) {
  auto* img = get_image(image_handle);
  if (!img) {
    LCRITICAL("can't create subresource: no image found");
    return {};
  }

  VkImageViewCreateInfo view_info{.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                                  .image = img->image(),
                                  .format = vk2::convert_format(img->get_desc().format),
                                  .subresourceRange = {
                                      .baseMipLevel = base_mip_level,
                                      .levelCount = level_count,
                                      .baseArrayLayer = base_array_layer,
                                      .layerCount = layer_count,
                                  }};
  // make image view
  if (format_is_color(img->get_desc().format)) {
    view_info.subresourceRange.aspectMask |= VK_IMAGE_ASPECT_COLOR_BIT;
  }
  if (format_is_depth(img->get_desc().format)) {
    view_info.subresourceRange.aspectMask |= VK_IMAGE_ASPECT_DEPTH_BIT;
  }
  if (format_is_stencil(img->get_desc().format)) {
    view_info.subresourceRange.aspectMask |= VK_IMAGE_ASPECT_STENCIL_BIT;
  }

  const auto& desc = img->get_desc();
  if (layer_count > 1) {
    if (has_flag(desc.misc_flags, ResourceMiscFlag::ImageCube)) {
      if (layer_count > 6 && layer_count != constants::remaining_array_layers) {
        view_info.viewType = VK_IMAGE_VIEW_TYPE_CUBE_ARRAY;
      } else {
        view_info.viewType = VK_IMAGE_VIEW_TYPE_CUBE;
      }
    } else {
      if (desc.type == ImageDesc::Type::TwoD) {
        view_info.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
      } else if (desc.type == ImageDesc::Type::OneD) {
        view_info.viewType = VK_IMAGE_VIEW_TYPE_1D_ARRAY;
      }
    }
  } else {
    if (desc.type == ImageDesc::Type::TwoD) {
      view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
    } else if (desc.type == ImageDesc::Type::OneD) {
      view_info.viewType = VK_IMAGE_VIEW_TYPE_1D;
    }
  }

  i32 handle = img->subresources_.size();
  auto* view = &img->subresources_.emplace_back();

  VK_CHECK(vkCreateImageView(device_, &view_info, nullptr, &view->view));
  if (has_flag(img->get_desc().bind_flags, BindFlag::ShaderResource)) {
    assert(img->sampled_view_.is_valid());
    view->resource_info =
        allocate_sampled_img_descriptor(view->view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
  }
  if (has_flag(img->get_desc().bind_flags, BindFlag::Storage)) {
    assert(img->storage_view_.is_valid());
    view->resource_info = allocate_storage_img_descriptor(view->view, VK_IMAGE_LAYOUT_GENERAL);
  }
  return handle;
}

void Device::destroy(Image& img) {
  if (img.image_) {
    delete_texture(TextureDeleteInfo{img.image_, img.allocation_});
    if (img.sampled_view_.is_valid()) {
      texture_view_delete_q3_.emplace_back(img.sampled_view_);
    }
    if (img.storage_view_.is_valid()) {
      texture_view_delete_q3_.emplace_back(img.storage_view_);
    }
    if (img.attachment_view_) {
      texture_view_delete_q2_.emplace_back(img.attachment_view_);
    }
    for (auto& view : img.subresources_) {
      texture_view_delete_q3_.emplace_back(view);
    }
    img.image_ = nullptr;
  }
}

BindlessResourceInfo Device::allocate_sampled_img_descriptor(VkImageView view,
                                                             VkImageLayout layout) {
  u32 handle = sampled_image_allocator_.alloc();
  VkDescriptorImageInfo img{.sampler = nullptr, .imageView = view, .imageLayout = layout};
  allocate_bindless_resource(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &img, nullptr, handle,
                             bindless_sampled_image_binding);
  return {ResourceType::SampledImage, handle};
}

BindlessResourceInfo Device::allocate_storage_img_descriptor(VkImageView view,
                                                             VkImageLayout layout) {
  u32 handle = storage_image_allocator_.alloc();
  VkDescriptorImageInfo img{.sampler = nullptr, .imageView = view, .imageLayout = layout};
  allocate_bindless_resource(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &img, nullptr, handle,
                             resource_to_binding(ResourceType::StorageImage));
  return {ResourceType::StorageImage, handle};
}

BindlessResourceInfo Device::allocate_sampler_descriptor(VkSampler sampler) {
  u32 handle = sampler_allocator_.alloc();
  VkDescriptorImageInfo info{.sampler = sampler};
  VkWriteDescriptorSet write{.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                             .dstSet = main_set2_,
                             .dstBinding = bindless_sampler_binding,
                             .dstArrayElement = handle,
                             .descriptorCount = 1,
                             .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER,
                             .pImageInfo = &info,
                             .pBufferInfo = nullptr};
  vkUpdateDescriptorSets(device_, 1, &write, 0, nullptr);
  return {ResourceType::Sampler, handle};
}

void Device::allocate_bindless_resource(VkDescriptorType descriptor_type,
                                        VkDescriptorImageInfo* img, VkDescriptorBufferInfo* buffer,
                                        u32 idx, u32 binding) {
  VkWriteDescriptorSet write{.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                             .dstSet = main_set_,
                             .dstBinding = binding,
                             .dstArrayElement = idx,
                             .descriptorCount = 1,
                             .descriptorType = descriptor_type,
                             .pImageInfo = img,
                             .pBufferInfo = buffer};
  vkUpdateDescriptorSets(device_, 1, &write, 0, nullptr);
}

u32 Device::resource_to_binding(ResourceType type) {
  switch (type) {
    default:
    case ResourceType::StorageImage:
      return bindless_storage_image_binding;
    case ResourceType::StorageBuffer:
      return bindless_storage_buffer_binding;
    case ResourceType::SampledImage:
      return bindless_sampled_image_binding;
    case ResourceType::Sampler:
      return bindless_sampler_binding;
    case ResourceType::CombinedImageSampler:
      return bindless_combined_image_sampler_binding;
  }
}

void Device::delete_texture(const TextureDeleteInfo& img) {
  texture_delete_q_.emplace_back(img, curr_frame_num());
}

void Device::flush_deletions() {
  ZoneScoped;

  std::erase_if(texture_delete_q_, [this](const DeleteQEntry<TextureDeleteInfo>& entry) {
    if (entry.frame + frames_in_flight < curr_frame_num()) {
      vmaDestroyImage(allocator_, entry.data.img, entry.data.allocation);
      return true;
    }
    return false;
  });

  std::erase_if(semaphore_delete_q_, [this](const DeleteQEntry<VkSemaphore>& entry) {
    if (entry.frame + frames_in_flight < curr_frame_num()) {
      vkDestroySemaphore(device_, entry.data, nullptr);
      return true;
    }
    return false;
  });

  std::erase_if(pipeline_delete_q_, [this](const DeleteQEntry<VkPipeline>& entry) {
    if (entry.frame + frames_in_flight < curr_frame_num()) {
      vkDestroyPipeline(device_, entry.data, nullptr);
      return true;
    }
    return false;
  });

  std::erase_if(swapchain_delete_q_, [this](const DeleteQEntry<VkSwapchainKHR>& entry) {
    if (entry.frame + frames_in_flight < curr_frame_num()) {
      vkDestroySwapchainKHR(device_, entry.data, nullptr);
      return true;
    }
    return false;
  });

  std::erase_if(texture_view_delete_q3_, [this](const DeleteQEntry<ImageView2>& entry) {
    if (entry.frame + frames_in_flight < curr_frame_num()) {
      const auto& resource_info = entry.data.resource_info;
      vkDestroyImageView(device_, entry.data.view, nullptr);
      switch (resource_info.type) {
        case gfx::ResourceType::Sampler:
          sampler_allocator_.free(resource_info.handle);
          break;
        case gfx::ResourceType::SampledImage:
          sampled_image_allocator_.free(resource_info.handle);
          break;
        case gfx::ResourceType::StorageBuffer:
          storage_buffer_allocator_.free(resource_info.handle);
          break;
        case gfx::ResourceType::StorageImage:
          storage_image_allocator_.free(resource_info.handle);
          break;
        case gfx::ResourceType::CombinedImageSampler:
          LCRITICAL("not handled");
          exit(1);
      }
      return true;
    }
    return false;
  });

  std::erase_if(texture_view_delete_q2_, [this](const DeleteQEntry<VkImageView>& entry) {
    if (entry.frame + frames_in_flight < curr_frame_num()) {
      vkDestroyImageView(device_, entry.data, nullptr);
      return true;
    }
    return false;
  });
  std::erase_if(texture_view_delete_q_, [this](const DeleteQEntry<ImageView>& entry) {
    if (entry.frame + frames_in_flight < curr_frame_num()) {
      if (entry.data.sampled_image_resource_info_.is_valid()) {
        sampled_image_allocator_.free(entry.data.sampled_image_resource_info_.handle);
      }
      if (entry.data.storage_image_resource_info_.is_valid()) {
        storage_image_allocator_.free(entry.data.storage_image_resource_info_.handle);
      }
      vkDestroyImageView(device_, entry.data.view_, nullptr);
      return true;
    }
    return false;
  });

  std::erase_if(storage_buffer_delete_q_, [this](const DeleteQEntry<BufferHandle>& entry) {
    if (entry.frame + frames_in_flight < curr_frame_num()) {
      if (auto* buf = buffer_pool_.get(entry.data); buf) {
        if (buf->resource_info_->is_valid()) {
          storage_buffer_allocator_.free(buf->resource_info_->handle);
        }
        vmaDestroyBuffer(allocator_, buf->buffer_, buf->allocation_);
        buffer_pool_.destroy(entry.data);
        return true;
      }
    }
    return false;
  });
}

void Device::delete_texture_view(const ImageView& info) {
  texture_view_delete_q_.emplace_back(info, curr_frame_num_);
}

BindlessResourceInfo Device::allocate_storage_buffer_descriptor(VkBuffer buffer) {
  u32 handle = storage_buffer_allocator_.alloc();
  VkDescriptorBufferInfo buf{.buffer = buffer, .offset = 0, .range = VK_WHOLE_SIZE};
  allocate_bindless_resource(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &buf, handle,
                             resource_to_binding(ResourceType::StorageBuffer));
  return {ResourceType::StorageBuffer, handle};
}

void Device::enqueue_delete_swapchain(VkSwapchainKHR swapchain) {
  swapchain_delete_q_.emplace_back(swapchain, curr_frame_num_);
}

void Device::enqueue_delete_pipeline(VkPipeline pipeline) {
  pipeline_delete_q_.emplace_back(pipeline, curr_frame_num_);
}

void Device::enqueue_delete_sempahore(VkSemaphore semaphore) {
  semaphore_delete_q_.emplace_back(semaphore, curr_frame_num_);
}

u32 Device::IndexAllocator::alloc() {
  if (free_list_.empty()) {
    return next_index_++;
  }
  auto ret = free_list_.back();
  free_list_.pop_back();
  return ret;
}

void Device::IndexAllocator::free(u32 idx) {
  if (idx != UINT32_MAX) {
    free_list_.push_back(idx);
  }
}

Device::IndexAllocator::IndexAllocator(u32 size) { free_list_.reserve(size); }

void Device::init_bindless() {
  {
    VkDescriptorSetLayoutBinding bindings[] = {
        VkDescriptorSetLayoutBinding{
            .binding = bindless_storage_image_binding,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            .descriptorCount = max_resource_descriptors,
            .stageFlags = VK_SHADER_STAGE_ALL,
        },
        VkDescriptorSetLayoutBinding{
            .binding = bindless_combined_image_sampler_binding,
            .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .descriptorCount = max_sampler_descriptors,
            .stageFlags = VK_SHADER_STAGE_ALL,
        },
        VkDescriptorSetLayoutBinding{
            .binding = bindless_storage_buffer_binding,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = max_resource_descriptors,
            .stageFlags = VK_SHADER_STAGE_ALL,
        },
        VkDescriptorSetLayoutBinding{
            .binding = bindless_sampled_image_binding,
            .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
            .descriptorCount = max_resource_descriptors,
            .stageFlags = VK_SHADER_STAGE_ALL,
        },
    };

    std::array<VkDescriptorBindingFlags, COUNTOF(bindings)> flags;
    for (auto& f : flags) {
      f = VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT | VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT |
          VK_DESCRIPTOR_BINDING_UPDATE_UNUSED_WHILE_PENDING_BIT_EXT;
    }

    VkDescriptorSetLayoutBindingFlagsCreateInfo binding_flags{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO,
        .bindingCount = flags.size(),
        .pBindingFlags = flags.data()};
    VkDescriptorSetLayoutCreateInfo set_info{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .pNext = &binding_flags,
        .flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT,
        .bindingCount = COUNTOF(bindings),
        .pBindings = bindings};
    VK_CHECK(vkCreateDescriptorSetLayout(device_, &set_info, nullptr, &main_set_layout_));

    VkDescriptorPoolSize sizes[] = {
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, max_resource_descriptors},
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, max_resource_descriptors},
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_SAMPLER, max_sampler_descriptors},
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, max_resource_descriptors},
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, max_sampler_descriptors},
    };

    VkDescriptorPoolCreateInfo info{.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                                    .flags = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT,
                                    // TODO: fine tune this? just one?
                                    .maxSets = 10,
                                    .poolSizeCount = COUNTOF(sizes),
                                    .pPoolSizes = sizes};
    VK_CHECK(vkCreateDescriptorPool(device_, &info, nullptr, &main_pool_));
    assert(main_pool_);
    assert(main_set_layout_);
    VkDescriptorSetAllocateInfo set_layout_info{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = main_pool_,
        .descriptorSetCount = 1,
        .pSetLayouts = &main_set_layout_};
    VK_CHECK(vkAllocateDescriptorSets(device_, &set_layout_info, &main_set_));
    assert(main_set_);
  }
  VkDescriptorSetLayoutBinding bindings[] = {
      VkDescriptorSetLayoutBinding{
          .binding = bindless_sampler_binding,
          .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER,
          .descriptorCount = max_sampler_descriptors,
          .stageFlags = VK_SHADER_STAGE_ALL,
      },
  };

  std::array<VkDescriptorBindingFlags, COUNTOF(bindings)> flags;
  for (auto& f : flags) {
    f = VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT | VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT |
        VK_DESCRIPTOR_BINDING_UPDATE_UNUSED_WHILE_PENDING_BIT_EXT;
  }

  VkDescriptorSetLayoutBindingFlagsCreateInfo binding_flags{
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO,
      .bindingCount = flags.size(),
      .pBindingFlags = flags.data()};
  VkDescriptorSetLayoutCreateInfo set_info{
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .pNext = &binding_flags,
      .flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT,
      .bindingCount = COUNTOF(bindings),
      .pBindings = bindings};
  VK_CHECK(vkCreateDescriptorSetLayout(device_, &set_info, nullptr, &main_set2_layout_));
  assert(main_pool_);
  assert(main_set2_layout_);
  VkDescriptorSetAllocateInfo set_layout_info{
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .descriptorPool = main_pool_,
      .descriptorSetCount = 1,
      .pSetLayouts = &main_set2_layout_};
  VK_CHECK(vkAllocateDescriptorSets(device_, &set_layout_info, &main_set2_));
  assert(main_set2_);
}

void Device::bind_bindless_descriptors(CmdEncoder& cmd) {
  cmd.bind_descriptor_set(VK_PIPELINE_BIND_POINT_GRAPHICS, default_pipeline_layout_, &main_set_, 0);
  cmd.bind_descriptor_set(VK_PIPELINE_BIND_POINT_COMPUTE, default_pipeline_layout_, &main_set_, 0);
  cmd.bind_descriptor_set(VK_PIPELINE_BIND_POINT_GRAPHICS, default_pipeline_layout_, &main_set2_,
                          1);
  cmd.bind_descriptor_set(VK_PIPELINE_BIND_POINT_COMPUTE, default_pipeline_layout_, &main_set2_, 1);
}

u32 Device::get_bindless_idx(BufferHandle buffer) {
  auto* buf = buffer_pool_.get(buffer);
  if (buf) {
    return buf->resource_info_->handle;
  }
  return 0;
}

u32 Device::get_bindless_idx(ImageHandle img, SubresourceType type, int subresource) {
  auto* image = get_image(img);
  if (!image) {
    LCRITICAL("failed to get bindless index, image doesn't exist");
    return 0;
  }
  ImageView* view{};
  if (subresource == -1) {
    switch (type) {
      case gfx::SubresourceType::Shader:
        return image->sampled_view_.resource_info.handle;
      case gfx::SubresourceType::Storage:
        return image->storage_view_.resource_info.handle;
      case gfx::SubresourceType::Attachment:
        assert(0);
        return 0;
    }
    assert(0);
  } else {
    if (subresource < 0 || static_cast<size_t>(subresource) >= image->subresources_.size()) {
      LCRITICAL("invalid subresource index: {}", subresource);
      exit(1);
    }
    return image->subresources_[subresource].resource_info.handle;
  }
  assert(0);
  if (type == SubresourceType::Storage) {
    return view->storage_image_resource_info_.handle;
  }
  return view->sampled_image_resource_info_.handle;
}

u32 Device::get_bindless_idx(const Holder<ImageHandle>& img, SubresourceType type,
                             int subresource) {
  return get_bindless_idx(img.handle, type, subresource);
}

ImageView2 Device::create_image_view2(ImageHandle image_handle, SubresourceType type,
                                      u32 base_mip_level, u32 level_count, u32 base_array_layer,
                                      u32 layer_count) {
  auto* img = get_image(image_handle);
  if (!img) {
    LCRITICAL("can't create subresource: no image found");
    return {};
  }

  VkImageViewCreateInfo view_info{.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                                  .image = img->image(),
                                  .format = vk2::convert_format(img->get_desc().format),
                                  .subresourceRange = {
                                      .baseMipLevel = base_mip_level,
                                      .levelCount = level_count,
                                      .baseArrayLayer = base_array_layer,
                                      .layerCount = layer_count,
                                  }};
  // make image view
  if (format_is_color(img->get_desc().format)) {
    view_info.subresourceRange.aspectMask |= VK_IMAGE_ASPECT_COLOR_BIT;
  }
  if (format_is_depth(img->get_desc().format)) {
    view_info.subresourceRange.aspectMask |= VK_IMAGE_ASPECT_DEPTH_BIT;
  }
  if (format_is_stencil(img->get_desc().format)) {
    view_info.subresourceRange.aspectMask |= VK_IMAGE_ASPECT_STENCIL_BIT;
  }

  const auto& desc = img->get_desc();
  if (layer_count > 1) {
    if (has_flag(desc.misc_flags, ResourceMiscFlag::ImageCube)) {
      if (layer_count > 6 && layer_count != constants::remaining_array_layers) {
        view_info.viewType = VK_IMAGE_VIEW_TYPE_CUBE_ARRAY;
      } else {
        view_info.viewType = VK_IMAGE_VIEW_TYPE_CUBE;
      }
    } else {
      if (desc.type == ImageDesc::Type::TwoD) {
        view_info.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
      } else if (desc.type == ImageDesc::Type::OneD) {
        view_info.viewType = VK_IMAGE_VIEW_TYPE_1D_ARRAY;
      }
    }
  } else {
    if (desc.type == ImageDesc::Type::TwoD) {
      view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
    } else if (desc.type == ImageDesc::Type::OneD) {
      view_info.viewType = VK_IMAGE_VIEW_TYPE_1D;
    }
  }

  ImageView2 view;
  VK_CHECK(vkCreateImageView(device_, &view_info, nullptr, &view.view));

  if (type == SubresourceType::Shader) {
    if (!has_flag(desc.bind_flags, BindFlag::ShaderResource | BindFlag::ColorAttachment |
                                       BindFlag::DepthStencilAttachment)) {
      LCRITICAL(
          "cannot make sampled subresource when image was not created with "
          "BindFlag::ShaderResource");
      exit(1);
    }
    view.resource_info =
        allocate_sampled_img_descriptor(view.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
  } else if (type == SubresourceType::Storage) {
    if (!has_flag(desc.bind_flags, BindFlag::Storage)) {
      LCRITICAL(
          "cannot make storage subresource when image was not created with "
          "BindFlag::Storage");
      exit(1);
    }
    view.resource_info = allocate_storage_img_descriptor(view.view, VK_IMAGE_LAYOUT_GENERAL);
  }

  return view;
}

VkImageView Device::get_image_view(ImageHandle img, SubresourceType type, int subresource) {
  auto* image = get_image(img);
  if (!image) {
    LCRITICAL("failed to get bindless index, image doesn't exist");
    return VK_NULL_HANDLE;
  }
  if (subresource == -1) {
    switch (type) {
      case gfx::SubresourceType::Shader:
        assert(image->sampled_view_.view);
        return image->sampled_view_.view;
      case gfx::SubresourceType::Storage:
        assert(image->storage_view_.view);
        return image->storage_view_.view;
      case gfx::SubresourceType::Attachment:
        assert(image->attachment_view_);
        return image->attachment_view_;
    }
    assert(0);
  }
  if (subresource < 0 || static_cast<size_t>(subresource) >= image->subresources_.size()) {
    LCRITICAL("invalid subresource index: {}", subresource);
    exit(1);
  }
  return image->subresources_[subresource].view;

  assert(0);  // unreachable
  return VK_NULL_HANDLE;
}

}  // namespace gfx
