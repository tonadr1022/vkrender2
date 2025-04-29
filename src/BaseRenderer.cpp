#include "BaseRenderer.hpp"

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_vulkan.h>
#include <volk.h>
#include <vulkan/vk_enum_string_helper.h>
#include <vulkan/vulkan_core.h>

#include <algorithm>
#include <tracy/TracyVulkan.hpp>
#include <utility>

#include "GLFW/glfw3.h"
#include "Logger.hpp"
#include "VkBootstrap.h"
#include "tracy/Tracy.hpp"
#include "vk2/BindlessResourceAllocator.hpp"
#include "vk2/Device.hpp"
#include "vk2/Fence.hpp"
#include "vk2/Initializers.hpp"
#include "vk2/SamplerCache.hpp"
#include "vk2/Swapchain.hpp"
#include "vk2/VkCommon.hpp"

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

namespace gfx {

BaseRenderer::BaseRenderer(const InitInfo& info, const BaseInitInfo& base_info)
    : window_(info.window),
      resource_dir_(info.resource_dir),
      on_gui_callback_(info.on_gui_callback) {
  assert(resource_dir_.string().length());
  if (!info.window) {
    LCRITICAL("cannot initialize renderer, window not provided");
    exit(1);
  }

  ZoneScoped;
  frames_in_flight_ = std::min(base_info.frames_in_flight, max_frames_in_flight);
  {
    ZoneScopedN("volk init");
    VK_CHECK(volkInitialize());
  }
  {
    ZoneScopedN("instance init");
    vkb::InstanceBuilder instance_builder;
    instance_builder
        .set_minimum_instance_version(vk2::min_api_version_major, vk2::min_api_version_minor, 0)
        .set_app_name(info.name)
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
    app_del_queue_.push([this]() { vkb::destroy_instance(instance_); });
  }

  {
    ZoneScopedN("volk");
    volkLoadInstance(instance_.instance);
    app_del_queue_.push([]() { volkFinalize(); });
  }

  {
    ZoneScopedN("glfw init");
    glfwCreateWindowSurface(instance_.instance, window_, nullptr, &surface_);
    if (!surface_) {
      LCRITICAL("Failed to create surface");
      exit(1);
    }
  }

  app_del_queue_.push([this]() { vkb::destroy_surface(instance_, surface_); });

  vk2::Device::init({.instance = instance_, .surface = surface_});
  app_del_queue_.push([]() { vk2::Device::destroy(); });
  device_ = vk2::get_device().device();
  volkLoadDevice(device_);

  const auto& phys = vk2::get_device().vkb_device();
  for (u64 i = 0; i < phys.queue_families.size(); i++) {
    if (phys.queue_families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
      vkGetDeviceQueue(device_, i, 0, &queues_.graphics_queue);
      queues_.graphics_queue_idx = i;
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

  for (u64 i = 0; i < phys.queue_families.size(); i++) {
    if (i == queues_.graphics_queue_idx) continue;
    if (phys.queue_families[i].queueFlags & VK_QUEUE_TRANSFER_BIT) {
      vkGetDeviceQueue(device_, i, 0, &queues_.transfer_queue);
      queues_.transfer_queue_idx = i;
      break;
    }
  }

  queues_.is_unified_graphics_transfer = queues_.graphics_queue_idx == queues_.transfer_queue_idx;

#ifndef NDEBUG
  VkDebugUtilsObjectNameInfoEXT name_info = {
      .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT,
      .objectType = VK_OBJECT_TYPE_QUEUE,
      .objectHandle = reinterpret_cast<uint64_t>(queues_.graphics_queue),
      .pObjectName = "GraphicsQueue"};
  vkSetDebugUtilsObjectNameEXT(device_, &name_info);
#endif

  {
    ZoneScopedN("init swapchain");
#if defined(VK_USE_PLATFORM_WAYLAND_KHR)
    VkPresentModeKHR presentMode =
        info.vsync ? VK_PRESENT_MODE_FIFO_KHR : VK_PRESENT_MODE_MAILBOX_KHR;
#else
    VkPresentModeKHR present_mode =
        info.vsync ? VK_PRESENT_MODE_FIFO_KHR : VK_PRESENT_MODE_IMMEDIATE_KHR;
#endif
    swapchain_.init({.phys_device = vk2::get_device().phys_device(),
                     .device = vk2::get_device().device(),
                     .surface = surface_,
                     .present_mode = present_mode,
                     .dims = window_dims(),
                     .queue_idx = queues_.graphics_queue_idx,
                     .requested_resize = false},
                    vk2::get_device().get_swapchain_format());
    swapchain_.recreate_img_views(vk2::get_device().device());
  }
  app_del_queue_.push([this]() { swapchain_.destroy(vk2::get_device().device()); });

  vk2::FencePool::init(device_);
  app_del_queue_.push([]() { vk2::FencePool::destroy(); });

  {
    ZoneScopedN("init per frame");
    per_frame_data_.resize(frames_in_flight_);
    for (auto& frame : per_frame_data_) {
      frame.cmd_pool = vk2::get_device().create_command_pool(queues_.graphics_queue_idx);
      frame.main_cmd_buffer = vk2::get_device().create_command_buffer(frame.cmd_pool);
      frame.render_fence = vk2::get_device().create_fence();
      frame.swapchain_semaphore = vk2::get_device().create_semaphore();
      frame.render_semaphore = vk2::get_device().create_semaphore();
      frame.tracy_vk_ctx =
          TracyVkContext(vk2::get_device().phys_device(), vk2::get_device().device(),
                         queues_.graphics_queue, frame.main_cmd_buffer);
    }
  }

  app_del_queue_.push([this]() {
    for (auto& frame : per_frame_data_) {
      auto& d = vk2::get_device();
      d.destroy_fence(frame.render_fence);
      d.destroy_semaphore(frame.render_semaphore);
      d.destroy_semaphore(frame.swapchain_semaphore);
      d.destroy_command_pool(frame.cmd_pool);
      TracyVkDestroy(frame.tracy_vk_ctx);
    }
  });
  vk2::SamplerCache::init(device_);
  transfer_queue_manager_ = std::make_unique<QueueManager>(queues_.transfer_queue_idx, 1);
  app_del_queue_.push([this]() {
    vk2::SamplerCache::destroy();
    transfer_queue_manager_ = nullptr;
  });
  {
    ZoneScopedN("init imgui");
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    (void)io;
    if (!ImGui_ImplGlfw_InitForVulkan(window_, true)) {
      LCRITICAL("ImGui_ImplGlfw_InitForVulkan failed");
      exit(1);
    }
    {
      ImGui_ImplVulkan_LoadFunctions(
          [](const char* functionName, void* vulkanInstance) {
            return vkGetInstanceProcAddr(*static_cast<VkInstance*>(vulkanInstance), functionName);
          },
          &instance_);
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

      VkDescriptorPool imgui_pool;
      VK_CHECK(vkCreateDescriptorPool(device_, &pool_info, nullptr, &imgui_pool));

      ImGui_ImplVulkan_InitInfo init_info{};
      init_info.Instance = instance_;
      init_info.PhysicalDevice = vk2::get_device().phys_device();
      init_info.Device = device_;
      init_info.Queue = queues_.graphics_queue;
      init_info.DescriptorPool = imgui_pool;
      init_info.MinImageCount = 3;
      init_info.ImageCount = 3;
      init_info.UseDynamicRendering = true;

      init_info.PipelineRenderingCreateInfo = {};
      init_info.PipelineRenderingCreateInfo.sType =
          VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
      init_info.PipelineRenderingCreateInfo.colorAttachmentCount = 1;
      init_info.PipelineRenderingCreateInfo.pColorAttachmentFormats = &swapchain_.format;
      init_info.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
      if (!ImGui_ImplVulkan_Init(&init_info)) {
        LCRITICAL("ImGui_ImplVulkan_Init failed");
        exit(1);
      }
      ImGui_ImplVulkan_CreateFontsTexture();
      app_del_queue_.push([device = device_, imgui_pool]() {
        ImGui_ImplVulkan_Shutdown();
        vkDestroyDescriptorPool(device, imgui_pool, nullptr);
      });
    }
  }

  initialized_ = true;
}

BaseRenderer::~BaseRenderer() {
  vkDeviceWaitIdle(device_);
  vk2::get_device().destroy_resources();
  vk2::BindlessResourceAllocator::get().set_frame_num(curr_frame_num() + 100);
  vk2::BindlessResourceAllocator::get().flush_deletions();
  vk2::BindlessResourceAllocator::shutdown();
  app_del_queue_.flush();
}
void BaseRenderer::on_draw(const SceneDrawInfo&) {}

void BaseRenderer::on_imgui() {}

void BaseRenderer::on_update() {}

void BaseRenderer::draw(const SceneDrawInfo& info) {
  ImGui_ImplVulkan_NewFrame();
  ImGui_ImplGlfw_NewFrame();
  ImGui::NewFrame();

  if (on_gui_callback_) on_gui_callback_();
  on_imgui();
  ImGuiIO& io = ImGui::GetIO();
  ImGui::Render();
  if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable) {
    ImGui::UpdatePlatformWindows();
    ImGui::RenderPlatformWindowsDefault();
  }

  curr_frame_swapchain_status_ = swapchain_.update({.phys_device = vk2::get_device().phys_device(),
                                                    .device = vk2::get_device().device(),
                                                    .surface = surface_,
                                                    .present_mode = swapchain_.present_mode,
                                                    .dims = window_dims(),
                                                    .queue_idx = queues_.graphics_queue_idx,
                                                    .requested_resize = resize_swapchain_req_});
  resize_swapchain_req_ = false;
  if (curr_frame_swapchain_status_ == vk2::Swapchain::Status::NotReady) {
    return;
  }
  if (curr_frame_swapchain_status_ == vk2::Swapchain::Status::Resized) {
    swapchain_.recreate_img_views(device_);
    on_resize();
  }

  // wait for fence
  u64 timeout = 1000000000;
  vkWaitForFences(device_, 1, &curr_frame().render_fence, true, timeout);

  vk2::BindlessResourceAllocator::get().set_frame_num(curr_frame_num());
  vk2::BindlessResourceAllocator::get().flush_deletions();

  // acquire next image
  VkResult get_swapchain_img_res =
      vkAcquireNextImageKHR(device_, swapchain_.swapchain, timeout,
                            curr_frame().swapchain_semaphore, nullptr, &curr_swapchain_img_idx_);
  if (get_swapchain_img_res == VK_ERROR_OUT_OF_DATE_KHR) {
    return;
  }

  // reset fence once drawing, else return
  VK_CHECK(vkResetFences(device_, 1, &curr_frame().render_fence));

  on_draw(info);

  {
    ZoneScopedN("presentation");
    VkResult res;
    VkPresentInfoKHR info{.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
                          .waitSemaphoreCount = 1,
                          .pWaitSemaphores = &curr_frame().render_semaphore,
                          .swapchainCount = 1,
                          .pSwapchains = &swapchain_.swapchain,
                          .pImageIndices = &curr_swapchain_img_idx_,
                          .pResults = &res};
    VkResult present_result = vkQueuePresentKHR(queues_.graphics_queue, &info);
    if (present_result == VK_ERROR_OUT_OF_DATE_KHR || present_result == VK_SUBOPTIMAL_KHR) {
    } else {
      VK_CHECK(present_result);
    }
    curr_frame_num_++;
  }
}

vk2::Swapchain::Status BaseRenderer::curr_frame_swapchain_status() const {
  return curr_frame_swapchain_status_;
}
PerFrameData& BaseRenderer::curr_frame() { return per_frame_data_[curr_frame_num_ % 2]; }

void BaseRenderer::submit_single_command_buf_to_graphics(VkCommandBuffer cmd) {
  ZoneScoped;
  {
    ZoneScopedN("queue submit");
    // wait for swapchain to be ready
    auto wait_info = vk2::init::semaphore_submit_info(
        curr_frame().swapchain_semaphore, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT);
    // signal the render semaphore so presentation can wait on it
    auto signal_info = vk2::init::semaphore_submit_info(curr_frame().render_semaphore,
                                                        VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT);
    auto cmd_buf_submit_info = vk2::init::command_buffer_submit_info(cmd);
    auto submit = vk2::init::queue_submit_info(SPAN1(cmd_buf_submit_info), SPAN1(wait_info),
                                               SPAN1(signal_info));
    VK_CHECK(vkQueueSubmit2KHR(queues_.graphics_queue, 1, &submit, curr_frame().render_fence));
  }
}

uvec2 BaseRenderer::window_dims() const {
  int x, y;
  glfwGetFramebufferSize(window_, &x, &y);
  return {x, y};
}

void BaseRenderer::on_resize() {}

// TODO: refactor
QueueManager::QueueManager(u32 queue_idx, u32 cmd_buffer_cnt)
    : submit_semaphore_(vk2::get_device().create_semaphore(true)),
      cmd_pool_(vk2::get_device().create_command_pool(queue_idx)) {
  free_cmd_buffers_.resize(cmd_buffer_cnt);
  vk2::get_device().create_command_buffers(cmd_pool_.pool(), free_cmd_buffers_);
}

QueueManager::~QueueManager() { vk2::get_device().destroy_semaphore(submit_semaphore_); }

VkCommandBuffer QueueManager::get_cmd_buffer() {
  VkCommandBuffer buf;
  if (free_cmd_buffers_.size()) {
    buf = free_cmd_buffers_.back();
    free_cmd_buffers_.pop_back();
  } else {
    buf = vk2::get_device().create_command_buffer(cmd_pool_.pool());
  }
  return buf;
}

CmdPool::~CmdPool() {
  if (pool_) {
    vk2::get_device().destroy_command_pool(pool_);
    pool_ = nullptr;
  }
}

CmdPool::CmdPool(CmdPool&& old) noexcept : pool_(std::exchange(old.pool_, nullptr)) {}

CmdPool& CmdPool::operator=(CmdPool&& old) noexcept {
  if (&old == this) {
    return *this;
  }
  this->~CmdPool();
  pool_ = std::exchange(old.pool_, nullptr);
  return *this;
}

void BaseRenderer::render_imgui(VkCommandBuffer cmd, uvec2 draw_extent,
                                VkImageView target_img_view) {
  VkRenderingAttachmentInfo color_attachment = vk2::init::rendering_attachment_info(
      target_img_view, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, nullptr);
  VkRenderingInfo render_info = vk2::init::rendering_info({draw_extent.x, draw_extent.y},
                                                          &color_attachment, nullptr, nullptr);
  vkCmdBeginRenderingKHR(cmd, &render_info);
  ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmd);
  vkCmdEndRenderingKHR(cmd);
}
float BaseRenderer::aspect_ratio() const {
  auto dims = window_dims();
  return (float)dims.x / (float)dims.y;
}
}  // namespace gfx
