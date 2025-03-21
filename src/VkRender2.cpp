#include "VkRender2.hpp"

#include <volk.h>
#include <vulkan/vulkan_core.h>

#include <cassert>
#include <filesystem>
#include <tracy/Tracy.hpp>
#include <tracy/TracyVulkan.hpp>

#include "GLFW/glfw3.h"
#include "SceneLoader.hpp"
#include "StateTracker.hpp"
#include "glm/packing.hpp"
#include "vk2/BindlessResourceAllocator.hpp"
#include "vk2/Device.hpp"
#include "vk2/Fence.hpp"
#include "vk2/Initializers.hpp"
#include "vk2/PipelineManager.hpp"
#include "vk2/SamplerCache.hpp"
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

using namespace vk2;

VkRender2::VkRender2(const InitInfo& info)
    : BaseRenderer(info, BaseRenderer::BaseInitInfo{.frames_in_flight = 2}) {
  shader_dir_ = resource_dir_ / "shaders";
  allocator_ = vk2::get_device().allocator();

  vk2::BindlessResourceAllocator::init(device_, vk2::get_device().allocator());
  vk2::StagingBufferPool::init();

  imm_cmd_pool_ = vk2::get_device().create_command_pool(queues_.graphics_queue_idx);
  imm_cmd_buf_ = vk2::get_device().create_command_buffer(imm_cmd_pool_);
  main_del_q_.push([this]() { vk2::get_device().destroy_command_pool(imm_cmd_pool_); });

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
      .depth_stencil = GraphicsPipelineCreateInfo::depth_enable(true, CompareOp::Less),
  });

  uint32_t white = glm::packUnorm4x8(glm::vec4(1, 1, 1, 1));
  vk2::Buffer* staging = vk2::StagingBufferPool::get().acquire(32);
  memcpy((char*)staging->mapped_data(), (void*)&white, sizeof(u32));
  default_data_.white_img =
      vk2::create_texture_2d(VK_FORMAT_R8G8B8A8_SRGB, {1, 1, 1}, TextureUsage::ReadOnly);
  immediate_submit([this, staging](VkCommandBuffer cmd) {
    // TODO: extract
    state_.reset(cmd);
    state_.transition(default_data_.white_img->image(), VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                      VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    state_.flush_barriers();
    vkCmdCopyBufferToImage2KHR(cmd, vk2::addr(VkCopyBufferToImageInfo2{
                                        .sType = VK_STRUCTURE_TYPE_COPY_BUFFER_TO_IMAGE_INFO_2_KHR,
                                        .srcBuffer = staging->buffer(),
                                        .dstImage = default_data_.white_img->image(),
                                        .dstImageLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                        .regionCount = 1,
                                        .pRegions = vk2::addr(VkBufferImageCopy2{
                                            .sType = VK_STRUCTURE_TYPE_BUFFER_IMAGE_COPY_2,
                                            .bufferOffset = 0,
                                            .bufferRowLength = 0,
                                            .bufferImageHeight = 0,
                                            .imageSubresource =
                                                {
                                                    .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                                                    .mipLevel = 0,
                                                    .layerCount = 1,
                                                },
                                            .imageExtent = VkExtent3D{1, 1, 1}})}));
    state_.transition(default_data_.white_img->image(), VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                      VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                      VK_IMAGE_LAYOUT_READ_ONLY_OPTIMAL);
    state_.flush_barriers();
  });
  vk2::StagingBufferPool::get().free(staging);

  // TODO: this is cringe
  SamplerCache::get().get_or_create_sampler(VkSamplerCreateInfo{
      .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
      .magFilter = VK_FILTER_NEAREST,
      .minFilter = VK_FILTER_NEAREST,
      .mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR,
      .addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT,
      .addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT,
      .addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT,
      .minLod = -1000,
      .maxLod = 100,
      .borderColor = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK,

  });
}

void VkRender2::on_draw(const SceneDrawInfo& info) {
  while (!pending_buffer_transfers_.empty()) {
    auto& f = pending_buffer_transfers_.front();
    if (!f.fence) {
      StagingBufferPool::get().free(f.data);
      pending_buffer_transfers_.pop();
      continue;
    }

    if (vkGetFenceStatus(device_, f.fence) == VK_SUCCESS) {
      StagingBufferPool::get().free(f.data);
      FencePool::get().free(f.fence);
      pending_buffer_transfers_.pop();
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
  {
    TracyVkZone(curr_frame().tracy_vk_ctx, cmd, "draw objects");
    ctx.bind_descriptor_set(VK_PIPELINE_BIND_POINT_GRAPHICS, default_pipeline_layout_, &main_set_,
                            0);
    ctx.bind_descriptor_set(VK_PIPELINE_BIND_POINT_COMPUTE, default_pipeline_layout_, &main_set_,
                            0);
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

    // draw
    {
      state_.transition(
          img_->image(), VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
          VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
          VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_ASPECT_COLOR_BIT);
      state_.transition(depth_img_->image(),
                        VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT |
                            VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
                        VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                        VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, VK_IMAGE_ASPECT_DEPTH_BIT);
      state_.flush_barriers();
      VkClearValue depth_clear{.depthStencil = {.depth = 1.f}};
      VkRenderingAttachmentInfo color_attachment =
          init::rendering_attachment_info(img_->view(), VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
      auto depth_att = init::rendering_attachment_info(
          depth_img_->view(), VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, &depth_clear);
      auto rendering_info = init::rendering_info(img_->extent_2d(), &color_attachment, &depth_att);
      vkCmdBeginRenderingKHR(cmd, &rendering_info);
      set_viewport_and_scissor(cmd, img_->extent_2d());

      mat4 proj = info.proj;
      proj[1][1] *= -1;

      mat4 vp = proj * info.view;
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                        PipelineManager::get().get(draw_pipeline_)->pipeline);

      // TODO: uniform buffer lol
      for (auto& scene : loaded_scenes_) {
        struct {
          mat4 vp;
          u64 vbaddr;
          u32 instance_buffer;
          u32 materials_buffer;
        } pc{vp, scene.resources->vertex_buffer.device_addr(),
             scene.resources->instance_buffer.resource_info_->handle,
             scene.resources->materials_buffer.resource_info_->handle};
        ctx.push_constants(default_pipeline_layout_, sizeof(pc), &pc);
        vkCmdBindIndexBuffer(cmd, scene.resources->index_buffer.buffer(), 0, VK_INDEX_TYPE_UINT32);
        vkCmdDrawIndexedIndirect(cmd, scene.resources->draw_indirect_buffer.buffer(), 0,
                                 scene.resources->draw_cnt, sizeof(VkDrawIndexedIndirectCommand));
      }
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

    VkExtent3D dims{glm::min(img_->extent().width, swapchain_.dims.x),
                    glm::min(img_->extent().height, swapchain_.dims.y), 1};
    blit_img(cmd, img_->image(), swapchain_img, dims, VK_IMAGE_ASPECT_COLOR_BIT);

    if (draw_imgui) {
      state_.transition(swapchain_img, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                        VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
      state_.flush_barriers();
      render_imgui(cmd, {swapchain_.dims.x, swapchain_.dims.y},
                   swapchain_.img_views[curr_swapchain_img_idx()]);
    }

    state_.transition(swapchain_img, VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT_KHR,
                      VK_ACCESS_2_MEMORY_READ_BIT, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);
    state_.flush_barriers();
    TracyVkCollect(curr_frame().tracy_vk_ctx, cmd);
  }
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

SceneHandle VkRender2::load_scene(const std::filesystem::path& path) {
  auto ret = gfx::load_gltf(path);
  if (!ret.has_value()) {
    return {};
  }
  auto res = std::move(ret.value());

  std::vector<VkDrawIndexedIndirectCommand> cmds;
  std::vector<InstanceData> transforms;
  for (auto& node : res.scene_graph_data.node_datas) {
    for (auto& mesh_indices : node.meshes) {
      auto& mesh = res.mesh_draw_infos[mesh_indices.mesh_idx];
      cmds.emplace_back(
          VkDrawIndexedIndirectCommand{.indexCount = mesh.index_count,
                                       .instanceCount = 1,
                                       .firstIndex = mesh.first_index,
                                       .vertexOffset = static_cast<i32>(mesh.first_vertex),
                                       .firstInstance = 0});
      transforms.emplace_back(node.world_transform, mesh_indices.material_id);
    }
  }
  u64 draw_indirect_buf_size = cmds.size() * sizeof(VkDrawIndexedIndirectCommand);
  u64 instance_buf_size = transforms.size() * sizeof(InstanceData);
  SceneHandle handle{loaded_scenes_.size()};
  loaded_scenes_.emplace_back(LoadedScene{
      std::move(res.scene_graph_data),
      std::make_unique<SceneGPUResources>(
          vk2::Buffer{vk2::BufferCreateInfo{
              .size = res.vertices_size,
              .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
              .buffer_device_address = true,
          }},
          vk2::Buffer{vk2::BufferCreateInfo{
              .size = res.indices_size,
              .usage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
          }},
          vk2::Buffer{vk2::BufferCreateInfo{
              .size = res.materials.size() * sizeof(gfx::Material),
              .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
          }},
          vk2::Buffer{vk2::BufferCreateInfo{
              .size = draw_indirect_buf_size,
              .usage = VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
          }},
          vk2::Buffer{vk2::BufferCreateInfo{
              .size = instance_buf_size,
              .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
          }},
          std::move(res.samplers), std::move(res.textures), cmds.size()),
  });

  auto& gltf_result = loaded_scenes_[handle.get()];

  auto* resources = gltf_result.resources.get();

  auto copy_buffer = [](VkCommandBuffer cmd, VkBuffer src_buffer, VkBuffer dst_buffer,
                        u64 src_offset, u64 dst_offset, u64 size) {
    VkBufferCopy2KHR copy{.sType = VK_STRUCTURE_TYPE_BUFFER_COPY_2_KHR,
                          .srcOffset = src_offset,
                          .dstOffset = dst_offset,
                          .size = size};
    VkCopyBufferInfo2KHR copy_info{.sType = VK_STRUCTURE_TYPE_COPY_BUFFER_INFO_2,
                                   .srcBuffer = src_buffer,
                                   .dstBuffer = dst_buffer,
                                   .regionCount = 1,
                                   .pRegions = &copy};
    vkCmdCopyBuffer2KHR(cmd, &copy_info);
  };

  u64 materials_size = res.materials.size() * sizeof(gfx::Material);
  vk2::Buffer* staging = vk2::StagingBufferPool::get().acquire(draw_indirect_buf_size +
                                                               instance_buf_size + materials_size);
  memcpy(staging->mapped_data(), cmds.data(), draw_indirect_buf_size);
  memcpy((char*)staging->mapped_data() + draw_indirect_buf_size, transforms.data(),
         instance_buf_size);
  memcpy((char*)staging->mapped_data() + draw_indirect_buf_size + instance_buf_size,
         res.materials.data(), materials_size);
  {
    immediate_submit([&](VkCommandBuffer cmd) {
      copy_buffer(cmd, res.vert_idx_staging->buffer(), resources->vertex_buffer.buffer(), 0, 0,
                  res.vertices_size);
      copy_buffer(cmd, res.vert_idx_staging->buffer(), resources->index_buffer.buffer(),
                  res.vertices_size, 0, res.indices_size);
      copy_buffer(cmd, staging->buffer(), resources->draw_indirect_buffer.buffer(), 0, 0,
                  draw_indirect_buf_size);
      copy_buffer(cmd, staging->buffer(), resources->instance_buffer.buffer(),
                  draw_indirect_buf_size, 0, instance_buf_size);
      copy_buffer(cmd, staging->buffer(), resources->materials_buffer.buffer(),
                  draw_indirect_buf_size + instance_buf_size, 0, materials_size);
    });
  }

  // {
  //   VkCommandBuffer cmd = transfer_queue_manager_->get_cmd_buffer();
  //   {
  //     auto info = vk2::init::command_buffer_begin_info();
  //     VK_CHECK(vkResetCommandBuffer(cmd, 0));
  //     VK_CHECK(vkBeginCommandBuffer(cmd, &info));
  //     copy_buffer(cmd, res.vert_idx_staging->buffer(), resources->vertex_buffer.buffer(), 0, 0,
  //                 res.vertices_size);
  //     copy_buffer(cmd, res.vert_idx_staging->buffer(), resources->index_buffer.buffer(),
  //                 res.vertices_size, 0, res.indices_size);
  //     copy_buffer(cmd, staging->buffer(), resources->draw_indirect_buffer.buffer(), 0, 0,
  //                 draw_indirect_buf_size);
  //     copy_buffer(cmd, staging->buffer(), resources->instance_buffer.buffer(),
  //                 draw_indirect_buf_size, 0, instance_buf_size);
  //     // copy_buffer(cmd, staging->buffer(), resources->materials_buffer.buffer(),
  //     //             draw_indirect_buf_size + instance_buf_size, 0, materials_size);
  //     {
  //       transfer_q_state_.reset(cmd)
  //           .queue_transfer_buffer(state_, VK_PIPELINE_STAGE_2_VERTEX_INPUT_BIT,
  //                                  VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT,
  //                                  resources->vertex_buffer.buffer(), queues_.transfer_queue_idx,
  //                                  queues_.graphics_queue_idx)
  //           .queue_transfer_buffer(state_, VK_PIPELINE_STAGE_2_INDEX_INPUT_BIT,
  //                                  VK_ACCESS_2_INDEX_READ_BIT, resources->index_buffer.buffer(),
  //                                  queues_.transfer_queue_idx, queues_.graphics_queue_idx)
  //           .queue_transfer_buffer(state_, VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT,
  //                                  VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT,
  //                                  resources->draw_indirect_buffer.buffer(),
  //                                  queues_.transfer_queue_idx, queues_.graphics_queue_idx)
  //           .queue_transfer_buffer(state_, VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT,
  //                                  VK_ACCESS_2_SHADER_READ_BIT,
  //                                  resources->instance_buffer.buffer(),
  //                                  queues_.transfer_queue_idx, queues_.graphics_queue_idx)
  //           // .queue_transfer_buffer(state_, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
  //           //                        VK_ACCESS_2_SHADER_READ_BIT,
  //           //                        resources->materials_buffer.buffer(),
  //           //                        queues_.transfer_queue_idx, queues_.graphics_queue_idx)
  //           .flush_barriers();
  //     }
  //     VK_CHECK(vkEndCommandBuffer(cmd));
  //
  //     VkFence transfer_fence = vk2::FencePool::get().allocate(true);
  //     auto cmd_buf_submit_info = vk2::init::command_buffer_submit_info(cmd);
  //
  //     auto wait_info =
  //     vk2::init::semaphore_submit_info(transfer_queue_manager_->submit_semaphore_,
  //                                                       VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT);
  //     auto submit = init::queue_submit_info(SPAN1(cmd_buf_submit_info), {}, SPAN1(wait_info));
  //     LINFO("submit main graphics queue");
  //     VK_CHECK(vkQueueSubmit2KHR(queues_.transfer_queue, 1, &submit, transfer_fence));
  //     transfer_queue_manager_->submit_signaled_ = true;
  //     pending_buffer_transfers_.emplace(res.vert_idx_staging, transfer_fence);
  //     pending_buffer_transfers_.emplace(staging, nullptr);
  //   }
  // }
  return handle;
}

void VkRender2::submit_static(SceneHandle, mat4) {}

void VkRender2::immediate_submit(std::function<void(VkCommandBuffer cmd)>&& function) {
  VkFence imm_fence = FencePool::get().allocate(true);
  VK_CHECK(vkResetCommandBuffer(imm_cmd_buf_, 0));
  auto info = vk2::init::command_buffer_begin_info();
  VK_CHECK(vkBeginCommandBuffer(imm_cmd_buf_, &info));
  function(imm_cmd_buf_);
  VK_CHECK(vkEndCommandBuffer(imm_cmd_buf_));
  VkCommandBufferSubmitInfo cmd_info = init::command_buffer_submit_info(imm_cmd_buf_);
  VkSubmitInfo2 submit = init::queue_submit_info(SPAN1(cmd_info), {}, {});
  VK_CHECK(vkQueueSubmit2KHR(queues_.graphics_queue, 1, &submit, imm_fence));
  VK_CHECK(vkWaitForFences(device_, 1, &imm_fence, true, 99999999999));
  FencePool::get().free(imm_fence);
}
