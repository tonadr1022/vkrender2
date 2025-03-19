#include "VkRender2.hpp"

#include <volk.h>
#include <vulkan/vulkan_core.h>

#include <cassert>
#include <filesystem>
#include <tracy/Tracy.hpp>

#include "GLFW/glfw3.h"
#include "Logger.hpp"
#include "SceneLoader.hpp"
#include "glm/ext/matrix_clip_space.hpp"
#include "glm/ext/matrix_transform.hpp"
#include "vk2/BindlessResourceAllocator.hpp"
#include "vk2/Device.hpp"
#include "vk2/Fence.hpp"
#include "vk2/Initializers.hpp"
#include "vk2/PipelineManager.hpp"
#include "vk2/Texture.hpp"
#include "vk2/VkCommon.hpp"

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
  resource_dir = resource_dir_result.value();
  shader_dir = resource_dir / "shaders";
  allocator_ = vk2::get_device().allocator();

  vk2::BindlessResourceAllocator::init(device_, vk2::get_device().allocator());

  main_del_q.push([]() { vk2::PipelineManager::shutdown(); });

  vk2::PipelineManager::init(device_);

  VkPushConstantRange default_range{.stageFlags = VK_SHADER_STAGE_ALL, .offset = 0, .size = 128};
  // TODO: refactor
  VkDescriptorSetLayout main_set_layout = vk2::BindlessResourceAllocator::get().main_set_layout();
  main_set = vk2::BindlessResourceAllocator::get().main_set();

  VkPipelineLayoutCreateInfo pipeline_info{.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                                           .setLayoutCount = 1,
                                           .pSetLayouts = &main_set_layout,
                                           .pushConstantRangeCount = 1,
                                           .pPushConstantRanges = &default_range};
  VK_CHECK(vkCreatePipelineLayout(device_, &pipeline_info, nullptr, &default_pipeline_layout));
  main_del_q.push([this]() { vkDestroyPipelineLayout(device_, default_pipeline_layout, nullptr); });

  create_attachment_imgs();

  img_pipeline = PipelineManager::get().load_compute_pipeline(
      {get_shader_path("debug/clear_img.comp"), default_pipeline_layout});

  draw_pipeline = PipelineManager::get().load_graphics_pipeline(GraphicsPipelineCreateInfo{
      .vertex_path = get_shader_path("debug/basic.vert"),
      .fragment_path = get_shader_path("debug/basic.frag"),
      .layout = default_pipeline_layout,
      .rendering = {{{img->format()}}, depth_img->format()},
      .depth_stencil = GraphicsPipelineCreateInfo::depth_enable(true, CompareOp::Always),
      // .rendering = {{{img->format()}}},
      // .depth_stencil = GraphicsPipelineCreateInfo::depth_disable(),
  });

  auto res = std::move(gfx::load_gltf(resource_dir / "models/Cube/glTF/Cube.gltf").value());

  cube = LoadedScene{
      std::move(res.scene_graph_data),
      vk2::Buffer{vk2::BufferCreateInfo{
          .size = res.vertex_staging.size(),
          .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
          .alloc_flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT,
          .buffer_device_address = true,
      }},
      vk2::Buffer{vk2::BufferCreateInfo{
          .size = res.index_staging.size(),
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
                              .size = res.vertex_staging.size()};
        VkCopyBufferInfo2KHR copy_info{.sType = VK_STRUCTURE_TYPE_COPY_BUFFER_INFO_2,
                                       .srcBuffer = res.vertex_staging.buffer(),
                                       .dstBuffer = cube->vertex_buffer.buffer(),
                                       .regionCount = 1,
                                       .pRegions = &copy};
        vkCmdCopyBuffer2KHR(cmd, &copy_info);
        {
          VkBufferCopy2KHR copy{.sType = VK_STRUCTURE_TYPE_BUFFER_COPY_2_KHR,
                                .srcOffset = 0,
                                .dstOffset = 0,
                                .size = res.index_staging.size()};
          VkCopyBufferInfo2KHR copy_info{.sType = VK_STRUCTURE_TYPE_COPY_BUFFER_INFO_2,
                                         .srcBuffer = res.index_staging.buffer(),
                                         .dstBuffer = cube->index_buffer.buffer(),
                                         .regionCount = 1,
                                         .pRegions = &copy};
          vkCmdCopyBuffer2KHR(cmd, &copy_info);
        }

        // TODO: extract
        VkBufferMemoryBarrier2 barriers[] = {
            VkBufferMemoryBarrier2{.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
                                   .srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                                   .srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT,
                                   .srcQueueFamilyIndex = queues_.transfer_queue_idx,
                                   .dstQueueFamilyIndex = queues_.graphics_queue_idx,
                                   .buffer = cube->vertex_buffer.buffer(),
                                   .offset = 0,
                                   .size = VK_WHOLE_SIZE},
            VkBufferMemoryBarrier2{.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
                                   .srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                                   .srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT,
                                   .srcQueueFamilyIndex = queues_.transfer_queue_idx,
                                   .dstQueueFamilyIndex = queues_.graphics_queue_idx,
                                   .buffer = cube->index_buffer.buffer(),
                                   .offset = 0,
                                   .size = VK_WHOLE_SIZE}};

        VkDependencyInfo info{.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                              .bufferMemoryBarrierCount = COUNTOF(barriers),
                              .pBufferMemoryBarriers = barriers,
                              .imageMemoryBarrierCount = 0,
                              .pImageMemoryBarriers = nullptr};
        vkCmdPipelineBarrier2KHR(cmd, &info);
      }
      VK_CHECK(vkEndCommandBuffer(cmd));

      VkFence transfer_fence = vk2::FencePool::get().allocate(true);
      auto cmd_buf_submit_info = vk2::init::command_buffer_submit_info(cmd);

      auto wait_info = vk2::init::semaphore_submit_info(transfer_queue_manager_->submit_semaphore_,
                                                        VK_PIPELINE_STAGE_2_TRANSFER_BIT);
      auto submit = init::queue_submit_info(SPAN1(cmd_buf_submit_info), {}, SPAN1(wait_info));
      VK_CHECK(vkQueueSubmit2KHR(queues_.transfer_queue, 1, &submit, transfer_fence));
      transfer_queue_manager_->submit_signaled_ = true;
      in_flight_vertex_index_staging_buffers_.emplace(
          std::make_pair(std::move(res.vertex_staging), std::move(res.index_staging)),
          transfer_fence);
    }
    // staging buffer deleted here but too early
  }
}

void VkRender2::on_update() {}

void VkRender2::on_draw() {
  while (!in_flight_vertex_index_staging_buffers_.empty()) {
    auto& f = in_flight_vertex_index_staging_buffers_.front();
    if (vkGetFenceStatus(device_, f.fence) == VK_SUCCESS) {
      in_flight_vertex_index_staging_buffers_.pop();
      FencePool::get().free(f.fence);
    } else {
      break;
    }
  }
  VkCommandBuffer cmd = curr_frame().main_cmd_buffer;
  state.reset(cmd);

  CmdEncoder ctx{cmd};
  auto info = vk2::init::command_buffer_begin_info();
  VK_CHECK(vkResetCommandBuffer(cmd, 0));
  VK_CHECK(vkBeginCommandBuffer(cmd, &info));

  vk2::BindlessResourceAllocator::get().set_frame_num(curr_frame_num());
  vk2::BindlessResourceAllocator::get().flush_deletions();

  state.transition(img->image(), VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                   VK_ACCESS_2_MEMORY_WRITE_BIT, VK_IMAGE_LAYOUT_GENERAL);
  state.barrier();

  {
    struct {
      uint idx;
      float t;
    } pc{img->view().storage_img_resource().handle, static_cast<f32>(glfwGetTime())};
    ctx.bind_compute_pipeline(PipelineManager::get().get(img_pipeline)->pipeline);
    ctx.push_constants(default_pipeline_layout, sizeof(pc), &pc);
    ctx.bind_descriptor_set(VK_PIPELINE_BIND_POINT_COMPUTE, default_pipeline_layout, &main_set, 0);
    ctx.dispatch((img->extent().width + 16) / 16, (img->extent().height + 16) / 16, 1);
  }

  // draw cube
  {
    state.transition(img->image(), VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                     VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                     VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_ASPECT_COLOR_BIT);
    state.transition(depth_img->image(), VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT,
                     VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                     VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, VK_IMAGE_ASPECT_DEPTH_BIT);
    state.barrier();
    auto dims = window_dims();
    // VkClearValue clear{.color = {{.1, .1, .1, 1.}}};
    VkClearValue depth_clear{.depthStencil = {.depth = 0.f}};
    VkRenderingAttachmentInfo color_attachment =
        init::rendering_attachment_info(img->view(), VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    auto depth_att = init::rendering_attachment_info(
        depth_img->view(), VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, &depth_clear);
    auto info = init::rendering_info({dims.x, dims.y}, &color_attachment, &depth_att);
    vkCmdBeginRenderingKHR(cmd, &info);
    set_viewport_and_scissor(cmd, {dims.x, dims.y});
    float aspect_ratio = (float)dims.x / (float)dims.y;
    mat4 view = glm::lookAt(vec3{1, 2, 5}, vec3{0, 0, 0}, {0, 1, 0});
    mat4 proj = glm::perspective(glm::radians(70.f), aspect_ratio, 0.1f, 1000.f);
    mat4 vp = proj * view;
    struct {
      mat4 vp;
      u64 vbaddr;
    } pc{vp, cube->vertex_buffer.device_addr()};
    ctx.push_constants(default_pipeline_layout, sizeof(pc), &pc);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                      PipelineManager::get().get(draw_pipeline)->pipeline);
    // TODO: bind once?
    ctx.bind_descriptor_set(VK_PIPELINE_BIND_POINT_GRAPHICS, default_pipeline_layout, &main_set, 0);
    vkCmdBindIndexBuffer(cmd, cube->index_buffer.buffer(), 0, VK_INDEX_TYPE_UINT32);
    vkCmdDrawIndexed(cmd, cube->index_buffer.size() / sizeof(u32), 1, 0, 0, 0);
    vkCmdEndRenderingKHR(cmd);
  }

  state.transition(img->image(), VK_PIPELINE_STAGE_2_BLIT_BIT, VK_ACCESS_2_TRANSFER_READ_BIT,
                   VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_ASPECT_COLOR_BIT);

  auto& swapchain_img = swapchain_.imgs[curr_swapchain_img_idx()];
  state.transition(swapchain_img, VK_PIPELINE_STAGE_2_BLIT_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_ASPECT_COLOR_BIT);
  state.barrier();

  blit_img(cmd, img->image(), swapchain_img, img->extent(), VK_IMAGE_ASPECT_COLOR_BIT);

  state.transition(swapchain_img, VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT_KHR,
                   VK_ACCESS_2_MEMORY_READ_BIT, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);
  state.barrier();
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

std::string VkRender2::get_shader_path(const std::string& path) const { return shader_dir / path; }

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

  img = create_texture_2d(VK_FORMAT_R8G8B8A8_UNORM, dims, TextureUsage::General);
  depth_img = create_texture_2d(VK_FORMAT_D32_SFLOAT, dims, TextureUsage::AttachmentReadOnly);
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
