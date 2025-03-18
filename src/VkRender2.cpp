#include "VkRender2.hpp"

#include <volk.h>
#include <vulkan/vulkan_core.h>

#include <cassert>
#include <filesystem>
#include <tracy/Tracy.hpp>

#include "GLFW/glfw3.h"
#include "Logger.hpp"
#include "SceneLoader.hpp"
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

  img_pipeline = PipelineManager::get().load_compute_pipeline(
      {get_shader_path("debug/clear_img.comp"), default_pipeline_layout});

  draw_pipeline = PipelineManager::get().load_graphics_pipeline(GraphicsPipelineCreateInfo{
      .vertex_path = get_shader_path("debug/basic.vert"),
      .fragment_path = get_shader_path("debug/basic.frag"),
      .layout = default_pipeline_layout,
      .depth_stencil = GraphicsPipelineCreateInfo::depth_enable(true, CompareOp::Less),
  });

  auto res = std::move(gfx::load_gltf(resource_dir / "models/Cube/glTF/Cube.gltf").value());

  cube = LoadedScene{
      std::move(res.scene_graph_data),
      vk2::Buffer{vk2::BufferCreateInfo{
          .size = res.vertex_staging.size(),
          .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
          .alloc_flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT}},
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
        // TODO: extract
        VkBufferMemoryBarrier2 barrier{.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
                                       .srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                                       .srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT,
                                       .srcQueueFamilyIndex = queues_.transfer_queue_idx,
                                       .dstQueueFamilyIndex = queues_.graphics_queue_idx,
                                       .buffer = cube->vertex_buffer.buffer(),
                                       .offset = 0,
                                       .size = VK_WHOLE_SIZE};
      }
      VK_CHECK(vkEndCommandBuffer(cmd));

      VkFence transfer_fence = vk2::FencePool::get().allocate(true);
      auto cmd_buf_submit_info = vk2::init::command_buffer_submit_info(cmd);
      auto submit = init::queue_submit_info(SPAN1(cmd_buf_submit_info), {}, {});
      VK_CHECK(vkQueueSubmit2KHR(queues_.transfer_queue, 1, &submit, transfer_fence));
      in_flight_vertex_index_staging_buffers_.emplace(
          std::make_pair(std::move(res.vertex_staging), std::move(res.index_staging)),
          transfer_fence);
    }
    // staging buffer deleted here but too early
  }
  create_attachment_imgs();
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
  state.queue_transition(img->image(), VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                         VK_ACCESS_2_MEMORY_WRITE_BIT, VK_IMAGE_LAYOUT_GENERAL);
  state.barrier();

  {
    struct {
      uint idx;
      float t;
    } pc{img->view().storage_img_resource().handle, static_cast<f32>(glfwGetTime())};
    ctx.push_constants(default_pipeline_layout, sizeof(pc), &pc);
    ctx.bind_compute_pipeline(PipelineManager::get().get(img_pipeline)->pipeline);
    ctx.bind_descriptor_set(VK_PIPELINE_BIND_POINT_COMPUTE, default_pipeline_layout, &main_set, 0);
    ctx.dispatch((img->extent().width + 16) / 16, (img->extent().height + 16) / 16, 1);
  }

  state.queue_transition(img->image(), VK_PIPELINE_STAGE_2_BLIT_BIT, VK_ACCESS_2_TRANSFER_READ_BIT,
                         VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_ASPECT_COLOR_BIT);

  auto& swapchain_img = swapchain_.imgs[curr_swapchain_img_idx()];
  state.queue_transition(swapchain_img, VK_PIPELINE_STAGE_2_BLIT_BIT,
                         VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                         VK_IMAGE_ASPECT_COLOR_BIT);
  state.barrier();

  blit_img(cmd, img->image(), swapchain_img, img->extent(), VK_IMAGE_ASPECT_COLOR_BIT);

  state.queue_transition(swapchain_img, VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT_KHR,
                         VK_ACCESS_2_MEMORY_READ_BIT, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);
  state.barrier();
  VK_CHECK(vkEndCommandBuffer(cmd));

  submit_single_command_buf_to_graphics(cmd);
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
  auto dims = window_dims();
  img = create_texture_2d(VK_FORMAT_R8G8B8A8_UNORM, uvec3{dims, 1}, TextureUsage::General);
}
