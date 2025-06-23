#include "Device.hpp"

#include <imgui/backends/imgui_impl_glfw.h>
#include <imgui/backends/imgui_impl_vulkan.h>
#include <imgui/imgui.h>
#include <volk.h>
#include <vulkan/vk_enum_string_helper.h>
#include <vulkan/vulkan_core.h>

#include <cassert>
#include <thread>
#include <tracy/Tracy.hpp>
#include <tracy/TracyVulkan.hpp>

#include "CommandEncoder.hpp"
#include "Common.hpp"
#include "GLFW/glfw3.h"
#include "Initializers.hpp"
#include "PipelineManager.hpp"
#include "Types.hpp"
#include "VkBootstrap.h"
#include "VkCommon.hpp"
#include "core/Logger.hpp"
#include "vk2/Buffer.hpp"
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
  supported_features12_.shaderFloat16 = true;
  VkPhysicalDeviceFeatures features{};
  features.shaderStorageImageWriteWithoutFormat = true;
  features.depthClamp = true;
  features.shaderInt64 = true;
  features.multiDrawIndirect = true;
  features.fragmentStoresAndAtomics = true;
  VkPhysicalDeviceVulkan11Features features11{
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
  features11.shaderDrawParameters = true;
  features11.storageBuffer16BitAccess = true;

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
      queue_family_indices_.emplace_back(i);
      queues_[(u32)QueueType::Graphics].family_idx = i;
      break;
    }
  }

  for (u64 i = 0; i < vkb_device_.queue_families.size(); i++) {
    if (i == queues_[(u32)QueueType::Graphics].family_idx) continue;
    if (vkb_device_.queue_families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
      vkGetDeviceQueue(device_, i, 0, &queues_[(u32)QueueType::Compute].queue);
      queues_[(u32)QueueType::Compute].family_idx = i;
      queue_family_indices_.emplace_back(i);
      break;
    }
  }

  for (u64 i = 0; i < vkb_device_.queue_families.size(); i++) {
    if (i == queues_[(u32)QueueType::Graphics].family_idx ||
        i == queues_[(u32)QueueType::Compute].family_idx) {
      continue;
    }
    if (vkb_device_.queue_families[i].queueFlags & VK_QUEUE_TRANSFER_BIT) {
      vkGetDeviceQueue(device_, i, 0, &queues_[(u32)QueueType::Transfer].queue);
      queues_[(u32)QueueType::Transfer].family_idx = i;
      queue_family_indices_.emplace_back(i);
      break;
    }
  }

  {
    int w, h;
    glfwGetWindowSize(window_, &w, &h);
    swapchain_.surface = surface_;
    create_swapchain(swapchain_, vk2::SwapchainDesc{.width = static_cast<u32>(w),
                                                    .height = static_cast<u32>(h),
                                                    .buffer_count = frames_in_flight,
                                                    .vsync = info.vsync});
  }

  // transition handler
  for (auto& transition_handler : transition_handlers_) {
    transition_handler.cmd_pool = create_command_pool(
        QueueType::Graphics, VK_COMMAND_POOL_CREATE_TRANSIENT_BIT, "transition handler pool");
    transition_handler.cmd_buf = create_command_buffer(transition_handler.cmd_pool);
    for (auto& semaphore : transition_handler.semaphores) {
      semaphore = create_semaphore(false, "transition handler");
    }
  }

  // frame resources
  for (u32 frame_i = 0; frame_i < frames_in_flight; frame_i++) {
    for (u32 queue_type = 0; queue_type < (u32)QueueType::Count; queue_type++) {
      auto& queue = queues_[queue_type];
      if (queue.queue == VK_NULL_HANDLE) {
        continue;
      }

      // create frame fences
      VkFence fence = create_fence(0);
      frame_fences_[frame_i][queue_type] = fence;
      switch ((QueueType)queue_type) {
        default:
        case QueueType::Compute:
          set_name(fence, "FrameFence[Compute]");
        case QueueType::Graphics:
          set_name(fence, "FrameFence[Graphics]");
        case QueueType::Transfer:
          set_name(fence, "FrameFence[Transfer]");
      }

      // create frame semaphores
      for (u32 other_queue_type = 0; other_queue_type < (u32)QueueType::Count; other_queue_type++) {
        if (other_queue_type == queue_type) {
          continue;
        }
        if (queues_[other_queue_type].queue == VK_NULL_HANDLE) {
          continue;
        }

        VkSemaphore semaphore = create_semaphore(false, "frame semaphore");
        queue.frame_semaphores[frame_i][other_queue_type] = semaphore;
        switch ((QueueType)queue_type) {
          default:
          case QueueType::Compute:
            set_name(semaphore, "FrameQueue[Compute]");
          case QueueType::Graphics:
            set_name(semaphore, "FrameQueue[Graphics]");
          case QueueType::Transfer:
            set_name(semaphore, "FrameQueue[Transfer]");
        }
      }
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

VkCommandPool Device::create_command_pool(QueueType type, VkCommandPoolCreateFlags flags,
                                          const char* name) const {
  VkCommandPoolCreateInfo info{.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                               .flags = flags,
                               .queueFamilyIndex = queues_[(u32)type].family_idx};
  VkCommandPool pool;
  VK_CHECK(vkCreateCommandPool(device_, &info, nullptr, &pool));
  set_name(pool, name);
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

VkSemaphore Device::create_semaphore(bool timeline, const char* name) const {
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
  set_name(semaphore, name);
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

Device::CopyAllocator::CopyCmd Device::CopyAllocator::allocate(u64 size) {
  CopyCmd cmd;
  {
    std::scoped_lock lock(free_list_mtx_);
    for (size_t i = 0; i < free_copy_cmds_.size(); i++) {
      auto& free_cmd = free_copy_cmds_[i];
      if (free_cmd.is_valid()) {
        auto* staging_buf = device_->get_buffer(free_cmd.staging_buffer);
        assert(staging_buf);
        if (staging_buf->size() >= size) {
          cmd = free_copy_cmds_[i];
          std::swap(free_copy_cmds_[i], *(free_copy_cmds_.end() - 1));
          free_copy_cmds_.pop_back();
          break;
        }
      }
    }
  }

  if (!cmd.is_valid()) {
    cmd.transfer_cmd_pool = device_->create_command_pool(type_, 0, "transfer cmd pool");
    cmd.transfer_cmd_buf = device_->create_command_buffer(cmd.transfer_cmd_pool);
    cmd.staging_buffer = device_->create_buffer(BufferCreateInfo{
        .size = std::max<u64>(size, 1024ul * 64), .flags = BufferCreateFlags_HostVisible});

    VkFenceCreateInfo info{.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, .flags = 0};
    VK_CHECK(vkCreateFence(device_->device_, &info, nullptr, &cmd.fence));
  }
  VK_CHECK(vkResetCommandPool(device_->device_, cmd.transfer_cmd_pool, 0));

  VkCommandBufferBeginInfo cmd_buf_begin_info{.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                                              .flags = 0};
  VK_CHECK(vkBeginCommandBuffer(cmd.transfer_cmd_buf, &cmd_buf_begin_info));
  VK_CHECK(vkResetFences(device_->device_, 1, &cmd.fence));
  return cmd;
}

void Device::CopyAllocator::submit(CopyCmd cmd) {
  ZoneScoped;
  // need to transfer ownership?
  VK_CHECK(vkEndCommandBuffer(cmd.transfer_cmd_buf));
  VkCommandBufferSubmitInfo cb_submit{.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
                                      .commandBuffer = cmd.transfer_cmd_buf};
  VkSubmitInfo2 submit_info{.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
                            .commandBufferInfoCount = 1,
                            .pCommandBufferInfos = &cb_submit};
  device_->get_queue(type_).submit(1, &submit_info, cmd.fence);
  VkResult res{};
  while ((res = vkWaitForFences(device_->device_, 1, &cmd.fence, true,
                                gfx::Device::timeout_value)) == VK_TIMEOUT) {
    LINFO("vkWaitForFences TIMEOUT, CopyAllocator: QueueType::Transfer");
    std::this_thread::yield();
  }

  std::scoped_lock lock(free_list_mtx_);
  free_copy_cmds_.emplace_back(cmd);
}

void Device::CopyAllocator::destroy() {
  std::scoped_lock lock(free_list_mtx_);
  for (auto& el : free_copy_cmds_) {
    vkDestroyFence(device_->device_, el.fence, nullptr);
    vkDestroyCommandPool(device_->device_, el.transfer_cmd_pool, nullptr);
    device_->destroy(el.staging_buffer);
  }
  free_copy_cmds_.clear();
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

void Device::acquire_next_image(CmdEncoder* cmd) {
  ZoneScoped;
  swapchain_.acquire_semaphore_idx =
      (swapchain_.acquire_semaphore_idx + 1) % swapchain_.imgs.size();
  VkResult acquire_next_image_result;
  // acquire next image
  do {
    acquire_next_image_result =
        vkAcquireNextImageKHR(device_, swapchain_.swapchain, timeout_value,
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
      glfwGetFramebufferSize(window_, &x, &y);
      auto desc = swapchain_.desc;
      desc.width = x;
      desc.height = y;
      create_swapchain(swapchain_, desc);
      acquire_next_image(cmd);
    }
  }
  // TODO: refactor this LOL
  if (cmd->submit_swapchains_.empty()) {
    cmd->submit_swapchains_.push_back(&swapchain_);
  }
  assert(cmd->submit_swapchains_.size() == 1);
}

void Device::begin_frame() { ZoneScoped; }

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
  vkDestroyPipelineLayout(device_, default_pipeline_layout_, nullptr);
  curr_frame_num_ = UINT32_MAX;
  for (auto& it : sampler_cache_) {
    destroy(it.second.first);
  }

  for (auto& t : transition_handlers_) {
    vkDestroyCommandPool(device_, t.cmd_pool, nullptr);
    for (auto& sem : t.semaphores) {
      vkDestroySemaphore(device_, sem, nullptr);
    }
  }

  for (auto& sem : free_semaphores_) {
    vkDestroySemaphore(device_, sem, nullptr);
  }
  for (auto& c : cmd_lists_) {
    if (c) {
      for (auto& sem : c->signal_semaphores_) {
        vkDestroySemaphore(device_, sem, nullptr);
      }
      for (auto& sem : c->wait_semaphores_) {
        vkDestroySemaphore(device_, sem, nullptr);
      }
      for (auto& pools : c->command_pools_) {
        for (auto& pool : pools) {
          vkDestroyCommandPool(device_, pool, nullptr);
        }
      }
    }
  }

  graphics_copy_allocator_.destroy();
  transfer_copy_allocator_.destroy();

  flush_deletions();
  assert(buffer_pool_.empty());
  assert(img_pool_.empty());
  assert(sampler_pool_.empty());

  vkDestroyDescriptorPool(device_, main_pool_, nullptr);
  vkDestroyDescriptorSetLayout(device_, main_set_layout_, nullptr);
  vkDestroyDescriptorSetLayout(device_, main_set2_layout_, nullptr);

  for (auto& f : free_fences_) {
    vkDestroyFence(device_, f, nullptr);
  }

  for (auto& frame_fence : frame_fences_) {
    for (VkFence fence : frame_fence) {
      vkDestroyFence(device_, fence, nullptr);
    }
  }

  for (Queue& queue : queues_) {
    for (auto& frame_semaphore : queue.frame_semaphores) {
      for (VkSemaphore sem : frame_semaphore) {
        vkDestroySemaphore(device_, sem, nullptr);
      }
    }
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

void Device::set_name(VkSemaphore semaphore, const char* name) const {
#ifdef DEBUG_VK_OBJECT_NAMES
  set_name(name, reinterpret_cast<u64>(semaphore), VK_OBJECT_TYPE_SEMAPHORE);
#else
  (void)semaphore;
  (void)name;
#endif
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

void Device::set_name(const char* name, u64 handle, VkObjectType type) const {
  VkDebugUtilsObjectNameInfoEXT name_info = {
      .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT,
      .objectType = type,
      .objectHandle = handle,
      .pObjectName = name};
  vkSetDebugUtilsObjectNameEXT(device_, &name_info);
}

void Device::set_name(VkFence fence, const char* name) {
#ifdef DEBUG_VK_OBJECT_NAMES
  if (fence) {
    set_name(name, reinterpret_cast<u64>(fence), VK_OBJECT_TYPE_FENCE);
  }
#else
  (void)fence;
  (void)name;
#endif
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

Holder<ImageHandle> Device::create_image_holder(const ImageDesc& desc, void* initial_data) {
  return Holder<ImageHandle>{create_image(desc, initial_data)};
}

ImageHandle Device::create_image(const ImageDesc& desc, void*) {
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
  if (queue_family_indices_.size() > 1) {
    cinfo.sharingMode = VK_SHARING_MODE_CONCURRENT;
    cinfo.queueFamilyIndexCount = queue_family_indices_.size();
    cinfo.pQueueFamilyIndices = queue_family_indices_.data();
  }
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

  // if (initial_data != nullptr) {
  //   // for each mip level, copy the buffer data to  each row
  //   u32 width = desc.dims.x, height = desc.dims.y, depth = desc.dims.z;
  //   for (u32 layer = 0; layer < desc.array_layers; layer++) {
  //     for (u32 mip = 0; mip < desc.mip_levels;mip++) {
  //
  //     }
  //
  //   }
  // }

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
  if (subresource == -1) {
    switch (type) {
      case gfx::SubresourceType::Shader:
        return image->sampled_view_.resource_info.handle;
      case gfx::SubresourceType::Storage:
        return image->storage_view_.resource_info.handle;
      case gfx::SubresourceType::Attachment:
        assert(0 && "can't access attachment view bindlessly");
        return 0;
    }
    assert(0);  // unreachable
    return 0;
  }

  assert(subresource >= 0 && static_cast<size_t>(subresource) < image->subresources_.size());
  return image->subresources_[subresource].resource_info.handle;
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

void Device::render_imgui(CmdEncoder& cmd) {
  ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmd.cmd());
}

void Device::new_imgui_frame() {
  ImGui_ImplVulkan_NewFrame();
  ImGui_ImplGlfw_NewFrame();
  ImGui::NewFrame();
}

Device::Device()
    : graphics_copy_allocator_(this, QueueType::Graphics),
      transfer_copy_allocator_(this, QueueType::Transfer) {}

void Device::enqueue_delete_texture_view(VkImageView view) {
  texture_view_delete_q2_.emplace_back(view, curr_frame_num());
}

void Device::Queue::wait(VkSemaphore semaphore) {
  if (!queue) {
    return;
  }
  wait_semaphores_infos.emplace_back(VkSemaphoreSubmitInfo{
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
      .semaphore = semaphore,
      .value = 0,  // no timeline
      .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
      .deviceIndex = 0,
  });
}

void Device::Queue::signal(VkSemaphore semaphore) {
  if (!queue) {
    return;
  }
  assert(semaphore != VK_NULL_HANDLE);
  signal_semaphore_infos.emplace_back(VkSemaphoreSubmitInfo{
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
      .semaphore = semaphore,
      .value = 0,  // no timeline
      .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
      .deviceIndex = 0,
  });
}

void Device::Queue::clear() {
  signal_semaphore_infos.clear();
  signal_semaphores.clear();
  wait_semaphores_infos.clear();
  submit_cmds.clear();
  swapchain_updates.clear();
  submit_swapchains.clear();
  submit_swapchain_img_indices.clear();
}

void Device::Queue::submit(Device* device, VkFence fence) {
  if (queue == VK_NULL_HANDLE) {
    return;
  }
  {
    // main submit
    if (fence != VK_NULL_HANDLE) {
      // end of frame submit, signal the semaphores so future submits can wait
      for (u32 queue_i = 0; queue_i < (u32)QueueType::Count; queue_i++) {
        VkSemaphore semaphore = frame_semaphores[device->curr_frame_in_flight()][queue_i];
        if (semaphore) signal(semaphore);
      }
    }

    // submit
    VkSubmitInfo2 queue_submit_info{
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
        .waitSemaphoreInfoCount = static_cast<u32>(wait_semaphores_infos.size()),
        .pWaitSemaphoreInfos = wait_semaphores_infos.data(),
        .commandBufferInfoCount = static_cast<u32>(submit_cmds.size()),
        .pCommandBufferInfos = submit_cmds.data(),
        .signalSemaphoreInfoCount = static_cast<u32>(signal_semaphore_infos.size()),
        .pSignalSemaphoreInfos = signal_semaphore_infos.data()};
    submit(1, &queue_submit_info, fence);
    wait_semaphores_infos.clear();
    signal_semaphore_infos.clear();
    submit_cmds.clear();
    // present swapchain results
    if (submit_swapchains.size()) {
      VkPresentInfoKHR info{.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
                            .waitSemaphoreCount = 1,
                            .pWaitSemaphores = signal_semaphores.data(),
                            .swapchainCount = static_cast<u32>(submit_swapchains.size()),
                            .pSwapchains = submit_swapchains.data(),
                            .pImageIndices = submit_swapchain_img_indices.data()};
      VkResult present_result{};
      {
        std::scoped_lock lock(mtx_);
        present_result = vkQueuePresentKHR(queue, &info);
      }
      // out of date == recreate swapchain
      if (present_result == VK_ERROR_OUT_OF_DATE_KHR || present_result == VK_SUBOPTIMAL_KHR) {
        for (auto& swapchain : swapchain_updates) {
          create_swapchain(*swapchain, swapchain->desc);
        }
      } else {
        VK_CHECK(present_result);
      }

      swapchain_updates.clear();
      submit_swapchain_img_indices.clear();
      signal_semaphores.clear();
      submit_swapchains.clear();
    }
  }
}

void Device::submit_commands() {
  // transition resources (images) to graphics queue
  if (!init_transitions_.empty()) {
    // place barriers and submit to grpahics queue
    auto& transition_handler = transition_handlers_[curr_frame_in_flight()];
    VK_CHECK(vkResetCommandPool(device_, transition_handler.cmd_pool, 0));
    VkCommandBufferBeginInfo cmd_buf_begin_info{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, .flags = 0};
    VK_CHECK(vkBeginCommandBuffer(transition_handler.cmd_buf, &cmd_buf_begin_info));
    VkDependencyInfo dep_info = vk2::init::dependency_info({}, init_transitions_);
    vkCmdPipelineBarrier2KHR(transition_handler.cmd_buf, &dep_info);
    VK_CHECK(vkEndCommandBuffer(transition_handler.cmd_buf));
    Queue& graphics_q = queues_[(u32)QueueType::Graphics];
    graphics_q.submit_cmds.emplace_back(
        vk2::init::command_buffer_submit_info(transition_handler.cmd_buf));
    // graphics queue transitions should complete before other operations
    for (u32 queue_type = 1; queue_type < (u32)QueueType::Count; queue_type++) {
      if (queues_[queue_type].queue == VK_NULL_HANDLE) {
        continue;
      }
      graphics_q.signal(transition_handler.semaphores[queue_type]);
      queues_[queue_type].wait(transition_handler.semaphores[queue_type]);
    }
    graphics_q.submit(this, VK_NULL_HANDLE);
    init_transitions_.clear();
  }

  // submit frame cmd lists
  {
    u32 last_cmd_idx = cmd_buf_count_;
    cmd_buf_count_ = 0;
    for (u32 cmd_i = 0; cmd_i < last_cmd_idx; cmd_i++) {
      CmdEncoder* cmd_list = cmd_lists_[cmd_i].get();
      VK_CHECK(vkEndCommandBuffer(cmd_list->get_cmd_buf()));
      Queue& queue = get_queue(cmd_list->queue_);
      // if the command list has dependencies, previous must be submitted first
      bool has_dependency =
          cmd_list->signal_semaphores_.size() || cmd_list->wait_semaphores_.size();
      if (has_dependency) {
        queue.submit(this, VK_NULL_HANDLE);
      }

      // submit cmd buf to the queue
      queue.submit_cmds.emplace_back(
          vk2::init::command_buffer_submit_info(cmd_list->get_cmd_buf()));

      // swapchain submits
      queue.swapchain_updates = cmd_list->submit_swapchains_;
      for (auto& swapchain : cmd_list->submit_swapchains_) {
        queue.submit_swapchains.push_back(swapchain->swapchain);
        queue.submit_swapchain_img_indices.push_back(swapchain->curr_swapchain_idx);
        // queue needs to wait for ready
        queue.wait_semaphores_infos.emplace_back(vk2::init::semaphore_submit_info(
            swapchain->acquire_semaphores[swapchain->acquire_semaphore_idx],
            VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_2_BLIT_BIT, 0));
        // signals the release for swapchain
        VkSemaphore sem = swapchain->release_semaphores[swapchain->acquire_semaphore_idx];
        assert(sem != VK_NULL_HANDLE);
        queue.signal_semaphores.emplace_back(sem);
        queue.signal_semaphore_infos.emplace_back(
            vk2::init::semaphore_submit_info(sem, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, 0));
      }

      // handle dependencies
      if (has_dependency) {
        for (VkSemaphore semaphore : cmd_list->wait_semaphores_) {
          queue.wait(semaphore);
        }
        cmd_list->wait_semaphores_.clear();

        {
          std::scoped_lock lock(semaphore_pool_mtx_);
          for (VkSemaphore semaphore : cmd_list->signal_semaphores_) {
            queue.signal(semaphore);
            free_semaphore_unsafe(semaphore);
          }
          cmd_list->signal_semaphores_.clear();
        }

        queue.submit(this, VK_NULL_HANDLE);
      }
    }

    // final submit with fence
    for (u32 queue_type = 0; queue_type < (u32)QueueType::Count; queue_type++) {
      queues_[queue_type].submit(this, frame_fences_[curr_frame_in_flight()][queue_type]);
    }
  }

  // sync queues at end of frame, no overlap going into next frame
  for (u32 queue1 = 0; queue1 < (u32)QueueType::Count; queue1++) {
    if (!queues_[queue1].queue) continue;
    for (u32 queue2 = 0; queue2 < (u32)QueueType::Count; queue2++) {
      if (!queues_[queue2].queue || queue1 == queue2) continue;
      VkSemaphore semaphore = queues_[queue2].frame_semaphores[curr_frame_in_flight()][queue1];
      if (semaphore) {
        queues_[queue1].wait(semaphore);
      }
    }
  }

  curr_frame_num_++;
  if (curr_frame_num_ >= frames_in_flight) {
    // wait for fences and reset
    VkFence wait_fences[(u32)QueueType::Count]{};
    VkFence reset_fences[(u32)QueueType::Count]{};
    u32 wait_fence_cnt{}, reset_fence_cnt{};

    for (VkFence fence : frame_fences_[curr_frame_in_flight()]) {
      if (!fence) continue;
      reset_fences[reset_fence_cnt++] = fence;
      if (vkGetFenceStatus(device_, fence) != VK_SUCCESS) {
        wait_fences[wait_fence_cnt++] = fence;
      }
    }
    if (wait_fence_cnt > 0) {
      while (true) {
        VkResult res =
            vkWaitForFences(device_, wait_fence_cnt, wait_fences, VK_TRUE, timeout_value);
        if (res == VK_TIMEOUT) {
          LERROR(
              "vkWaitForFences resulted in VK_TIMEOUT. Statuses:\nGraphics fence: {}\nCompute "
              "fence: "
              "{}\nTransfer fence: {}",
              string_VkResult(vkGetFenceStatus(
                  device_, frame_fences_[curr_frame_in_flight()][(u32)QueueType::Graphics])),
              string_VkResult(vkGetFenceStatus(
                  device_, frame_fences_[curr_frame_in_flight()][(u32)QueueType::Compute])),
              string_VkResult(vkGetFenceStatus(
                  device_, frame_fences_[curr_frame_in_flight()][(u32)QueueType::Transfer])));
        } else if (res != VK_SUCCESS) {
          LCRITICAL("vkWaitForFences failed, exiting");
          exit(1);
        } else {
          break;
        }
      }
    }

    if (reset_fence_cnt > 0) {
      VK_CHECK(vkResetFences(device_, reset_fence_cnt, reset_fences));
    }
  }

  for (auto& q : queues_) {
    if (!q.queue) q.clear();
  }
  flush_deletions();
}

CmdEncoder* Device::begin_command_list(QueueType queue_type) {
  u32 curr_cmd_idx = cmd_buf_count_++;
  if (curr_cmd_idx >= cmd_lists_.size()) {
    cmd_lists_.emplace_back(std::make_unique<CmdEncoder>(this, default_pipeline_layout_));
  }

  CmdEncoder* cmd = cmd_lists_[curr_cmd_idx].get();
  cmd->queue_ = queue_type;
  cmd->id_ = curr_cmd_idx;
  cmd->reset(curr_frame_in_flight());
  if (cmd->get_cmd_buf() == VK_NULL_HANDLE) {
    for (u32 frame_i = 0; frame_i < frames_in_flight; frame_i++) {
      cmd->command_pools_[frame_i][(u32)queue_type] = create_command_pool(
          queue_type, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, "begincmdlist pool");
      cmd->command_bufs_[frame_i][(u32)queue_type] =
          create_command_buffer(cmd->command_pools_[frame_i][(u32)queue_type]);
    }
  }

  VK_CHECK(vkResetCommandPool(device_, cmd->get_cmd_pool(), 0));
  auto cmd_begin_info =
      vk2::init::command_buffer_begin_info(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
  VK_CHECK(vkBeginCommandBuffer(cmd->get_cmd_buf(), &cmd_begin_info));

  return cmd;
}

void Device::begin_swapchain_blit(CmdEncoder* cmd) {
  ZoneScopedN("blit to swapchain");
  {
    // make swapchain img writeable in blit stage
    VkImageMemoryBarrier2 img_barriers[] = {
        VkImageMemoryBarrier2{
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
            .srcStageMask =
                VK_PIPELINE_STAGE_2_TRANSFER_BIT | VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
            .srcAccessMask = 0,
            .dstStageMask = VK_PIPELINE_STAGE_2_BLIT_BIT,
            .dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            .image = swapchain_.imgs[swapchain_.curr_swapchain_idx],
            .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
        },
    };
    VkDependencyInfo info{.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                          .imageMemoryBarrierCount = COUNTOF(img_barriers),
                          .pImageMemoryBarriers = img_barriers};
    vkCmdPipelineBarrier2KHR(cmd->cmd(), &info);
  }
}

void Device::blit_to_swapchain(CmdEncoder* cmd, const Image& img, uvec2 dims, uvec2 dst_dims) {
  assert(dst_dims.x > 0 && dst_dims.y > 0);
  assert(dims.x > 0 && dims.y > 0);
  VkImageBlit2 region{
      .sType = VK_STRUCTURE_TYPE_IMAGE_BLIT_2,
      .srcSubresource = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                         .mipLevel = 0,
                         .baseArrayLayer = 0,
                         .layerCount = 1},
      .srcOffsets = {{}, {static_cast<i32>(dims.x), static_cast<i32>(dims.y), 1}},
      .dstSubresource = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                         .mipLevel = 0,
                         .baseArrayLayer = 0,
                         .layerCount = 1},
      .dstOffsets = {{}, {static_cast<i32>(dst_dims.x), static_cast<i32>(dst_dims.y), 1}}};
  VkBlitImageInfo2 blit_info{.sType = VK_STRUCTURE_TYPE_BLIT_IMAGE_INFO_2,
                             .srcImage = img.image(),
                             .srcImageLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                             .dstImage = swapchain_.imgs[swapchain_.curr_swapchain_idx],
                             .dstImageLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                             .regionCount = 1,
                             .pRegions = &region,
                             .filter = VK_FILTER_NEAREST};
  vkCmdBlitImage2KHR(cmd->cmd(), &blit_info);
}

VkSemaphore Device::new_semaphore() {
  std::scoped_lock lock(semaphore_pool_mtx_);
  if (free_semaphores_.empty()) {
    return create_semaphore(false, "semaphore pool");
  }
  VkSemaphore sem = free_semaphores_.back();
  free_semaphores_.pop_back();
  return sem;
}

void Device::free_semaphore(VkSemaphore semaphore) {
  std::scoped_lock lock(semaphore_pool_mtx_);
  free_semaphore_unsafe(semaphore);
}
void Device::free_semaphore_unsafe(VkSemaphore semaphore) { free_semaphores_.push_back(semaphore); }

void Device::cmd_list_wait(CmdEncoder* cmd_list, CmdEncoder* wait_for) {
  assert(cmd_list != wait_for && wait_for->id_ < cmd_list->id_);
  VkSemaphore semaphore = new_semaphore();
  cmd_list->wait_semaphores_.emplace_back(semaphore);
  wait_for->signal_semaphores_.emplace_back(semaphore);
}

void Device::CopyAllocator::CopyCmd::copy_buffer(Device* device, const Buffer& dst, u64 src_offset,
                                                 u64 dst_offset, u64 size) const {
  VkBufferCopy2KHR copy{.sType = VK_STRUCTURE_TYPE_BUFFER_COPY_2_KHR,
                        .srcOffset = src_offset,
                        .dstOffset = dst_offset,
                        .size = size};

  VkCopyBufferInfo2KHR copy_info{.sType = VK_STRUCTURE_TYPE_COPY_BUFFER_INFO_2,
                                 .srcBuffer = device->get_buffer(staging_buffer)->buffer(),
                                 .dstBuffer = dst.buffer(),
                                 .regionCount = 1,
                                 .pRegions = &copy};
  vkCmdCopyBuffer2KHR(transfer_cmd_buf, &copy_info);
}

void Device::set_name(VkCommandPool pool, const char* name) const {
#ifdef DEBUG_VK_OBJECT_NAMES
  set_name(name, reinterpret_cast<u64>(pool), VK_OBJECT_TYPE_COMMAND_POOL);
#else
  (void)pool;
  (void)name;
#endif
}

void Device::Queue::submit(u32 submit_count, const VkSubmitInfo2* submits, VkFence fence) {
  std::scoped_lock lock(mtx_);
  VK_CHECK(vkQueueSubmit2KHR(queue, submit_count, submits, fence));
}

BufferHandle Device::create_staging_buffer(u64 size) {
  return create_buffer(BufferCreateInfo{.size = std::max<u64>(size, 1024ul * 64),
                                        .flags = BufferCreateFlags_HostVisible});
}

VkImage Device::get_curr_swapchain_img() const {
  return swapchain_.imgs[swapchain_.curr_swapchain_idx];
}
}  // namespace gfx
