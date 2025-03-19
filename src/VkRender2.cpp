#include "VkRender2.hpp"

#include <volk.h>
#include <vulkan/vulkan_core.h>

#include <cassert>
#include <filesystem>
#include <tracy/Tracy.hpp>

#include "GLFW/glfw3.h"
#include "Logger.hpp"
#include "SceneLoader.hpp"
#include "StateTracker.hpp"
#include "vk2/BindlessResourceAllocator.hpp"
#include "vk2/Device.hpp"
#include "vk2/Fence.hpp"
#include "vk2/Initializers.hpp"
#include "vk2/PipelineManager.hpp"
#include "vk2/StagingBufferPool.hpp"
#include "vk2/Texture.hpp"
#include "vk2/VkCommon.hpp"

namespace {
VkRender2* instance{};
}

VkRender2& VkRender2::get() {
  assert(instance);
  return *instance;
}

void VkRender2::init(const InitInfo& info) {
  assert(!instance);
  instance = new VkRender2{info};
}

void VkRender2::shutdown() {
  assert(instance);
  delete instance;
}

namespace {

std::optional<std::filesystem::path> get_resource_dir() {
  auto curr_path = std::filesystem::current_path();
  while (curr_path.has_parent_path()) {
    auto resource_path = curr_path / "resources";
    if (std::filesystem::exists(resource_path)) {
      return resource_path;
    }
    curr_path = curr_path.parent_path();
  }
  return std::nullopt;
}

}  // namespace

using namespace vk2;

VkRender2::VkRender2(const InitInfo& info)
    : BaseRenderer(info, BaseRenderer::BaseInitInfo{.frames_in_flight = 2}) {
  auto resource_dir_result = get_resource_dir();
  if (!resource_dir_result.has_value()) {
    LCRITICAL("unable to locate 'resources' directory from current path: {}",
              std::filesystem::current_path().string());
    exit(1);
  }
  resource_dir_ = resource_dir_result.value();
  shader_dir_ = resource_dir_ / "shaders";
  allocator_ = vk2::get_device().allocator();

  vk2::BindlessResourceAllocator::init(device_, vk2::get_device().allocator());
  vk2::StagingBufferPool::init();

  main_del_q_.push([]() {
    vk2::PipelineManager::shutdown();
    vk2::StagingBufferPool::destroy();
  });

  vk2::PipelineManager::init(device_);

  VkPushConstantRange default_range{.stageFlags = VK_SHADER_STAGE_ALL, .offset = 0, .size = 128};
  // TODO: refactor
  VkDescriptorSetLayout main_set_layout = vk2::BindlessResourceAllocator::get().main_set_layout();
  main_set_ = vk2::BindlessResourceAllocator::get().main_set();

  VkPipelineLayoutCreateInfo pipeline_info{.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                                           .setLayoutCount = 1,
                                           .pSetLayouts = &main_set_layout,
                                           .pushConstantRangeCount = 1,
                                           .pPushConstantRanges = &default_range};
  VK_CHECK(vkCreatePipelineLayout(device_, &pipeline_info, nullptr, &default_pipeline_layout_));
  main_del_q_.push(
      [this]() { vkDestroyPipelineLayout(device_, default_pipeline_layout_, nullptr); });

  create_attachment_imgs();

  img_pipeline_ = PipelineManager::get().load_compute_pipeline(
      {get_shader_path("debug/clear_img.comp"), default_pipeline_layout_});

  draw_pipeline_ = PipelineManager::get().load_graphics_pipeline(GraphicsPipelineCreateInfo{
      .vertex_path = get_shader_path("debug/basic.vert"),
      .fragment_path = get_shader_path("debug/basic.frag"),
      .layout = default_pipeline_layout_,
      .rendering = {{{img_->format()}}, depth_img_->format()},
      .depth_stencil = GraphicsPipelineCreateInfo::depth_enable(true, CompareOp::Always),
      // .rendering = {{{img->format()}}},
      // .depth_stencil = GraphicsPipelineCreateInfo::depth_disable(),
  });

  // auto res =
  // std::move(gfx::load_gltf("/home/tony/models/Models/Sponza/glTF/Sponza.gltf").value());
  auto res = std::move(gfx::load_gltf(resource_dir_ / "models/Cube/glTF/Cube.gltf").value());
  cube_ = LoadedScene{
      std::move(res.scene_graph_data),
      vk2::Buffer{vk2::BufferCreateInfo{
          .size = res.vertices_size,
          .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
          .alloc_flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT,
          .buffer_device_address = true,
      }},
      vk2::Buffer{vk2::BufferCreateInfo{
          .size = res.indices_size,
          .usage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
      }},
      std::move(res.samplers)};

  {
    VkCommandBuffer cmd = transfer_queue_manager_->get_cmd_buffer();
    {
      auto info = vk2::init::command_buffer_begin_info();
      VK_CHECK(vkResetCommandBuffer(cmd, 0));
      VK_CHECK(vkBeginCommandBuffer(cmd, &info));
      {
        VkBufferCopy2KHR copy{.sType = VK_STRUCTURE_TYPE_BUFFER_COPY_2_KHR,
                              .srcOffset = 0,
                              .dstOffset = 0,
                              .size = res.vertices_size};
        VkCopyBufferInfo2KHR copy_info{.sType = VK_STRUCTURE_TYPE_COPY_BUFFER_INFO_2,
                                       .srcBuffer = res.vert_idx_staging->buffer(),
                                       .dstBuffer = cube_->vertex_buffer.buffer(),
                                       .regionCount = 1,
                                       .pRegions = &copy};
        vkCmdCopyBuffer2KHR(cmd, &copy_info);
        {
          VkBufferCopy2KHR copy{.sType = VK_STRUCTURE_TYPE_BUFFER_COPY_2_KHR,
                                .srcOffset = res.vertices_size,
                                .dstOffset = 0,
                                .size = res.indices_size};
          VkCopyBufferInfo2KHR copy_info{.sType = VK_STRUCTURE_TYPE_COPY_BUFFER_INFO_2,
                                         .srcBuffer = res.vert_idx_staging->buffer(),
                                         .dstBuffer = cube_->index_buffer.buffer(),
                                         .regionCount = 1,
                                         .pRegions = &copy};
          vkCmdCopyBuffer2KHR(cmd, &copy_info);
        }
        transfer_q_state_.reset(cmd)
            .queue_transfer_buffer(state_, VK_PIPELINE_STAGE_2_VERTEX_INPUT_BIT,
                                   VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT,
                                   cube_->vertex_buffer.buffer(), queues_.transfer_queue_idx,
                                   queues_.graphics_queue_idx)
            .queue_transfer_buffer(state_, VK_PIPELINE_STAGE_2_INDEX_INPUT_BIT,
                                   VK_ACCESS_2_INDEX_READ_BIT, cube_->index_buffer.buffer(),
                                   queues_.transfer_queue_idx, queues_.graphics_queue_idx)
            .flush_barriers();
      }
      VK_CHECK(vkEndCommandBuffer(cmd));

      VkFence transfer_fence = vk2::FencePool::get().allocate(true);
      auto cmd_buf_submit_info = vk2::init::command_buffer_submit_info(cmd);

      auto wait_info = vk2::init::semaphore_submit_info(transfer_queue_manager_->submit_semaphore_,
                                                        VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT);
      auto submit = init::queue_submit_info(SPAN1(cmd_buf_submit_info), {}, SPAN1(wait_info));
      VK_CHECK(vkQueueSubmit2KHR(queues_.transfer_queue, 1, &submit, transfer_fence));
      transfer_queue_manager_->submit_signaled_ = true;
      in_flight_staging_buffers_.emplace(res.vert_idx_staging, transfer_fence);
    }
    // staging buffer deleted here but too early
  }
}

void VkRender2::on_draw(const SceneDrawInfo& info) {
  while (!in_flight_staging_buffers_.empty()) {
    auto& f = in_flight_staging_buffers_.front();
    if (vkGetFenceStatus(device_, f.fence) == VK_SUCCESS) {
      StagingBufferPool::get().free(f.data);
      FencePool::get().free(f.fence);
      in_flight_staging_buffers_.pop();
    } else {
      break;
    }
  }
  VkCommandBuffer cmd = curr_frame().main_cmd_buffer;
  state_.reset(cmd);

  CmdEncoder ctx{cmd};
  auto cmd_begin_info = vk2::init::command_buffer_begin_info();
  VK_CHECK(vkResetCommandBuffer(cmd, 0));
  VK_CHECK(vkBeginCommandBuffer(cmd, &cmd_begin_info));
  ctx.bind_descriptor_set(VK_PIPELINE_BIND_POINT_GRAPHICS, default_pipeline_layout_, &main_set_, 0);
  ctx.bind_descriptor_set(VK_PIPELINE_BIND_POINT_COMPUTE, default_pipeline_layout_, &main_set_, 0);
  state_.flush_transfers(queues_.graphics_queue_idx);

  vk2::BindlessResourceAllocator::get().set_frame_num(curr_frame_num());
  vk2::BindlessResourceAllocator::get().flush_deletions();

  state_.transition(img_->image(), VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                    VK_ACCESS_2_MEMORY_WRITE_BIT, VK_IMAGE_LAYOUT_GENERAL);
  state_.flush_barriers();

  {
    struct {
      uint idx;
      float t;
    } pc{img_->view().storage_img_resource().handle, static_cast<f32>(glfwGetTime())};
    ctx.bind_compute_pipeline(PipelineManager::get().get(img_pipeline_)->pipeline);
    ctx.push_constants(default_pipeline_layout_, sizeof(pc), &pc);
    ctx.dispatch((img_->extent().width + 16) / 16, (img_->extent().height + 16) / 16, 1);
  }

  // draw cube
  {
    state_.transition(
        img_->image(), VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
        VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_ASPECT_COLOR_BIT);
    state_.transition(
        depth_img_->image(),
        VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
        VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
        VK_IMAGE_ASPECT_DEPTH_BIT);
    state_.flush_barriers();
    auto dims = window_dims();
    // VkClearValue clear{.color = {{.1, .1, .1, 1.}}};
    VkClearValue depth_clear{.depthStencil = {.depth = 0.f}};
    VkRenderingAttachmentInfo color_attachment =
        init::rendering_attachment_info(img_->view(), VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    auto depth_att = init::rendering_attachment_info(
        depth_img_->view(), VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, &depth_clear);
    auto rendering_info = init::rendering_info({dims.x, dims.y}, &color_attachment, &depth_att);
    vkCmdBeginRenderingKHR(cmd, &rendering_info);
    set_viewport_and_scissor(cmd, {dims.x, dims.y});

    mat4 proj = info.proj;
    proj[1][1] *= -1;

    mat4 vp = proj * info.view;
    struct {
      mat4 vp;
      u64 vbaddr;
    } pc{vp, cube_->vertex_buffer.device_addr()};
    ctx.push_constants(default_pipeline_layout_, sizeof(pc), &pc);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                      PipelineManager::get().get(draw_pipeline_)->pipeline);
    vkCmdBindIndexBuffer(cmd, cube_->index_buffer.buffer(), 0, VK_INDEX_TYPE_UINT32);
    vkCmdDrawIndexed(cmd, cube_->index_buffer.size() / sizeof(u32), 1, 0, 0, 0);
    vkCmdEndRenderingKHR(cmd);
  }

  state_.transition(img_->image(), VK_PIPELINE_STAGE_2_BLIT_BIT, VK_ACCESS_2_MEMORY_READ_BIT,
                    VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_ASPECT_COLOR_BIT);

  state_.transition(depth_img_->image(), VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, VK_ACCESS_2_NONE,
                    VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, VK_IMAGE_ASPECT_DEPTH_BIT);

  auto& swapchain_img = swapchain_.imgs[curr_swapchain_img_idx()];
  state_.transition(swapchain_img, VK_PIPELINE_STAGE_2_BLIT_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_ASPECT_COLOR_BIT);

  state_.flush_barriers();

  blit_img(cmd, img_->image(), swapchain_img, img_->extent(), VK_IMAGE_ASPECT_COLOR_BIT);

  state_.transition(swapchain_img, VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT_KHR,
                    VK_ACCESS_2_MEMORY_READ_BIT, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);
  state_.flush_barriers();
  VK_CHECK(vkEndCommandBuffer(cmd));

  // wait for swapchain to be ready
  std::array<VkSemaphoreSubmitInfo, 10> wait_semaphores{};
  u32 next_wait_sem_idx{0};
  wait_semaphores[next_wait_sem_idx++] = vk2::init::semaphore_submit_info(
      curr_frame().swapchain_semaphore, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT);
  // signal the render semaphore so presentation can wait on it
  auto signal_info = vk2::init::semaphore_submit_info(curr_frame().render_semaphore,
                                                      VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT);
  if (transfer_queue_manager_->submit_signaled_) {
    wait_semaphores[next_wait_sem_idx++] = vk2::init::semaphore_submit_info(
        transfer_queue_manager_->submit_semaphore_, VK_PIPELINE_STAGE_2_TRANSFER_BIT);
    transfer_queue_manager_->submit_signaled_ = false;
  }
  auto cmd_buf_submit_info = vk2::init::command_buffer_submit_info(cmd);
  auto submit = vk2::init::queue_submit_info(SPAN1(cmd_buf_submit_info),
                                             std::span(wait_semaphores.data(), next_wait_sem_idx),
                                             SPAN1(signal_info));
  VK_CHECK(vkQueueSubmit2KHR(queues_.graphics_queue, 1, &submit, curr_frame().render_fence));
}

void VkRender2::on_gui() {}

std::string VkRender2::get_shader_path(const std::string& path) const { return shader_dir_ / path; }

VkRender2::~VkRender2() { vkDeviceWaitIdle(device_); }

void CmdEncoder::dispatch(u32 work_groups_x, u32 work_groups_y, u32 work_groups_z) {
  vkCmdDispatch(cmd_, work_groups_x, work_groups_y, work_groups_z);
}

void CmdEncoder::bind_compute_pipeline(VkPipeline pipeline) {
  vkCmdBindPipeline(cmd_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
}
void CmdEncoder::bind_descriptor_set(VkPipelineBindPoint bind_point, VkPipelineLayout layout,
                                     VkDescriptorSet* set, u32 idx) {
  vkCmdBindDescriptorSets(cmd_, bind_point, layout, idx, 1, set, 0, nullptr);
}

void CmdEncoder::push_constants(VkPipelineLayout layout, u32 size, void* data) {
  vkCmdPushConstants(cmd_, layout, VK_SHADER_STAGE_ALL, 0, size, data);
}

void VkRender2::on_resize() { create_attachment_imgs(); }

void VkRender2::create_attachment_imgs() {
  auto win_dims = window_dims();
  uvec3 dims{win_dims, 1};

  img_ = create_texture_2d(VK_FORMAT_R8G8B8A8_UNORM, dims, TextureUsage::General);
  depth_img_ = create_texture_2d(VK_FORMAT_D32_SFLOAT, dims, TextureUsage::AttachmentReadOnly);
}

void VkRender2::set_viewport_and_scissor(VkCommandBuffer cmd, VkExtent2D extent) {
  VkViewport viewport{.x = 0,
                      .y = 0,
                      .width = static_cast<float>(extent.width),
                      .height = static_cast<float>(extent.height),
                      .minDepth = 0.f,
                      .maxDepth = 1.f};

  vkCmdSetViewport(cmd, 0, 1, &viewport);
  VkRect2D scissor{.offset = VkOffset2D{.x = 0, .y = 0},
                   .extent = VkExtent2D{.width = extent.width, .height = extent.height}};
  vkCmdSetScissor(cmd, 0, 1, &scissor);
}
