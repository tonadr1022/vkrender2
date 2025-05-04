#include "BaseRenderer.hpp"

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_vulkan.h>
#include <volk.h>
#include <vulkan/vk_enum_string_helper.h>
#include <vulkan/vulkan_core.h>

#include <tracy/TracyVulkan.hpp>
#include <utility>

#include "GLFW/glfw3.h"
#include "Logger.hpp"
#include "tracy/Tracy.hpp"
#include "vk2/BindlessResourceAllocator.hpp"
#include "vk2/Device.hpp"
#include "vk2/Fence.hpp"
#include "vk2/Initializers.hpp"
#include "vk2/SamplerCache.hpp"

namespace gfx {

BaseRenderer::BaseRenderer(const InitInfo& info)
    : window_(info.window), resource_dir_(info.resource_dir) {
  assert(resource_dir_.string().length());
  if (!info.window) {
    LCRITICAL("cannot initialize renderer, window not provided");
    exit(1);
  }

  vk2::Device::init({info.name, window_, info.vsync});

  vk2::FencePool::init(device_);
  app_del_queue_.push([]() { vk2::FencePool::destroy(); });

  {
    ZoneScopedN("init per frame");
    auto& device = vk2::get_device();
    per_frame_data_.resize(device.get_frames_in_flight());
    for (auto& frame : per_frame_data_) {
      frame.cmd_pool = device.create_command_pool(vk2::QueueType::Graphics);
      frame.main_cmd_buffer = device.create_command_buffer(frame.cmd_pool);
      frame.tracy_vk_ctx =
          TracyVkContext(device.get_physical_device(), device.device(),
                         device.get_queue(vk2::QueueType::Graphics).queue, frame.main_cmd_buffer);
    }
  }

  app_del_queue_.push([this]() {
    for (auto& frame : per_frame_data_) {
      auto& d = vk2::get_device();
      d.destroy_command_pool(frame.cmd_pool);
      TracyVkDestroy(frame.tracy_vk_ctx);
    }
  });
  vk2::SamplerCache::init(device_);
  transfer_queue_manager_ = std::make_unique<QueueManager>(vk2::QueueType::Graphics, 1);
  app_del_queue_.push([this]() {
    vk2::SamplerCache::destroy();
    transfer_queue_manager_ = nullptr;
  });

  vk2::get_device().init_imgui();
  initialized_ = true;
}

BaseRenderer::~BaseRenderer() {
  vkDeviceWaitIdle(device_);
  vk2::get_device().destroy_resources();
  vk2::ResourceAllocator::get().set_frame_num(UINT32_MAX, 0);
  vk2::ResourceAllocator::get().flush_deletions();
  vk2::ResourceAllocator::shutdown();
  app_del_queue_.flush();
}
void BaseRenderer::on_draw(const SceneDrawInfo&) {}

void BaseRenderer::on_imgui() {}

void BaseRenderer::on_update() {}

void BaseRenderer::draw(const SceneDrawInfo& info) { on_draw(info); }

PerFrameData& BaseRenderer::curr_frame() {
  return per_frame_data_[curr_frame_num() % vk2::get_device().get_frames_in_flight()];
}

uvec2 BaseRenderer::window_dims() const {
  int x, y;
  glfwGetFramebufferSize(window_, &x, &y);
  return {x, y};
}

void BaseRenderer::on_resize() {}

// TODO: refactor
QueueManager::QueueManager(vk2::QueueType type, u32 cmd_buffer_cnt)
    : submit_semaphore_(vk2::get_device().create_semaphore(true)),
      cmd_pool_(vk2::get_device().create_command_pool(
          type, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT)) {
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

void BaseRenderer::new_frame() {
  ImGui_ImplVulkan_NewFrame();
  ImGui_ImplGlfw_NewFrame();
  ImGui::NewFrame();
}

u64 BaseRenderer::curr_frame_in_flight_num() const {
  return vk2::get_device().curr_frame_num() % vk2::get_device().get_frames_in_flight();
}
}  // namespace gfx
