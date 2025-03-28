#include "VkRender2.hpp"

// clang-format off
#include <volk.h>
#include <vulkan/vulkan_core.h>
#include <imgui_impl_vulkan.h>
// clang-format on

#include <cassert>
#include <filesystem>
#include <tracy/Tracy.hpp>
#include <tracy/TracyVulkan.hpp>

#include "GLFW/glfw3.h"
#include "Logger.hpp"
#include "SceneLoader.hpp"
#include "StateTracker.hpp"
#include "glm/packing.hpp"
#include "imgui.h"
#include "shaders/debug/basic_common.h.glsl"
#include "util/CVar.hpp"
#include "vk2/BindlessResourceAllocator.hpp"
#include "vk2/Buffer.hpp"
#include "vk2/Device.hpp"
#include "vk2/Fence.hpp"
#include "vk2/Hash.hpp"
#include "vk2/Initializers.hpp"
#include "vk2/PipelineManager.hpp"
#include "vk2/SamplerCache.hpp"
#include "vk2/StagingBufferPool.hpp"
#include "vk2/Texture.hpp"
#include "vk2/VkCommon.hpp"

namespace {
VkRender2* instance{};

AutoCVarInt ao_map_enabled{"renderer.ao_map", "AO Map", 1, CVarFlags::EditCheckbox};

// AutoCVarInt vsync{"renderer.vsync", "display vsync", 1, CVarFlags::EditCheckbox};

}  // namespace

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
  VkDescriptorSetLayout layouts[] = {main_set_layout,
                                     vk2::BindlessResourceAllocator::get().main_set2_layout_};
  main_set_ = vk2::BindlessResourceAllocator::get().main_set();
  main_set2_ = vk2::BindlessResourceAllocator::get().main_set2_;

  VkPipelineLayoutCreateInfo pipeline_info{.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                                           .setLayoutCount = COUNTOF(layouts),
                                           .pSetLayouts = layouts,
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
  assert(draw_pipeline_);

  uint32_t white = glm::packUnorm4x8(glm::vec4(1, 1, 1, 1));
  vk2::Buffer* staging = vk2::StagingBufferPool::get().acquire(32);
  memcpy((char*)staging->mapped_data(), (void*)&white, sizeof(u32));
  default_data_.white_img =
      vk2::create_texture_2d(VK_FORMAT_R8G8B8A8_SRGB, {1, 1, 1}, TextureUsage::ReadOnly);
  LINFO("defualt handle: {}", default_data_.white_img->view().sampled_img_resource().handle);
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
  linear_sampler_ = SamplerCache::get().get_or_create_sampler(VkSamplerCreateInfo{
      .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
      .magFilter = VK_FILTER_LINEAR,
      .minFilter = VK_FILTER_LINEAR,
      .mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR,
      .addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT,
      .addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT,
      .addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT,
      .minLod = -1000,
      .maxLod = 1000,
      .borderColor = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK,
  });
  default_mat_data_.white_img_handle =
      default_data_.white_img->view().sampled_img_resource().handle;

  {
    // per frame scene uniforms
    per_frame_data_2_.resize(frames_in_flight_);
    for (u32 i = 0; i < frames_in_flight_; i++) {
      auto& d = per_frame_data_2_[i];
      d.scene_uniform_buf = vk2::Buffer{vk2::BufferCreateInfo{
          .size = sizeof(SceneUniforms),
          .usage = VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT,
          .alloc_flags = VMA_ALLOCATION_CREATE_MAPPED_BIT |
                         VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT}};
    }
  }

  static_vertex_buf_ = LinearBuffer{create_storage_buffer(10'00'000 * sizeof(gfx::Vertex))};
  static_index_buf_ = LinearBuffer{BufferCreateInfo{
      .size = 10'000'00 * sizeof(u32),
      .usage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
  }};
  static_materials_buf_ = LinearBuffer{create_storage_buffer(10'000 * sizeof(gfx::Material))};
  u64 num_static_draws = 10'0'000;
  static_material_indices_buf_ =
      LinearBuffer{create_storage_buffer(num_static_draws * sizeof(u32))};
  static_transforms_buf_ = LinearBuffer{create_storage_buffer(num_static_draws * sizeof(mat4))};
  static_draw_cmds_buf_ = LinearBuffer{BufferCreateInfo{
      .size = num_static_draws * sizeof(VkDrawIndexedIndirectCommand),
      .usage = VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
  }};
}

void VkRender2::on_draw(const SceneDrawInfo& info) {
  ZoneScoped;
  {
    ZoneScopedN("scene uniform buffer");
    auto& d = curr_frame_2();
    SceneUniforms data;
    mat4 proj = info.proj;
    proj[1][1] *= -1;
    mat4 vp = proj * info.view;
    data.view_proj = vp;
    data.debug_flags = uvec4{};
    if (ao_map_enabled.get()) {
      data.debug_flags.x |= AO_ENABLED_BIT;
    }
    data.view_pos = info.view_pos;
    memcpy(d.scene_uniform_buf->mapped_data(), &data, sizeof(SceneUniforms));
  }

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
    ctx.bind_descriptor_set(VK_PIPELINE_BIND_POINT_GRAPHICS, default_pipeline_layout_, &main_set2_,
                            1);
    ctx.bind_descriptor_set(VK_PIPELINE_BIND_POINT_COMPUTE, default_pipeline_layout_, &main_set2_,
                            1);
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

      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                        PipelineManager::get().get(draw_pipeline_)->pipeline);

      auto draw_objects = [&](const Buffer& vertex_buffer, const Buffer& index_buffer,
                              const Buffer& instance_buffer, const Buffer& material_data_buffer,
                              const Buffer& material_indices_buffer,
                              const Buffer& draw_indirect_buffer, u64 draw_cnt) {
        BasicPushConstants pc{curr_frame_2().scene_uniform_buf->resource_info_->handle,
                              vertex_buffer.resource_info_->handle,
                              instance_buffer.resource_info_->handle,
                              material_data_buffer.resource_info_->handle,
                              material_indices_buffer.resource_info_->handle,
                              linear_sampler_->resource_info.handle};
        ctx.push_constants(default_pipeline_layout_, sizeof(pc), &pc);
        vkCmdBindIndexBuffer(cmd, index_buffer.buffer(), 0, VK_INDEX_TYPE_UINT32);
        vkCmdDrawIndexedIndirect(cmd, draw_indirect_buffer.buffer(), 0, draw_cnt,
                                 sizeof(VkDrawIndexedIndirectCommand));
      };

      for (auto& scene : loaded_dynamic_scenes_) {
        draw_objects(scene.resources->vertex_buffer, scene.resources->index_buffer,
                     scene.resources->instance_buffer, scene.resources->materials_buffer,
                     scene.resources->material_indices, scene.resources->draw_indirect_buffer,
                     scene.resources->draw_cnt);
      }
      if (draw_cnt_) {
        draw_objects(static_vertex_buf_->buffer, static_index_buf_->buffer,
                     static_transforms_buf_->buffer, static_materials_buf_->buffer,
                     static_material_indices_buf_->buffer, static_draw_cmds_buf_->buffer,
                     draw_cnt_);
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
      state_.transition(
          swapchain_img, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
          VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT,
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

  std::array<VkSemaphoreSubmitInfo, 10> wait_semaphores{};
  u32 next_wait_sem_idx{0};
  wait_semaphores[next_wait_sem_idx++] = vk2::init::semaphore_submit_info(
      curr_frame().swapchain_semaphore, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT);
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

void VkRender2::on_gui() {
  if (ImGui::Begin("Renderer")) {
    if (ImGui::CollapsingHeader("textures")) {
      for (const auto& obj : loaded_dynamic_scenes_) {
        for (const auto& t : obj.resources->textures) {
          ImGui::PushID(&t);
          if (ImGui::CollapsingHeader("a")) {
            ImVec2 window_size = ImGui::GetContentRegionAvail();
            float scale_width = window_size.x / t.extent_2d().width;
            float scaled_height = t.extent_2d().height * scale_width;
            ImGui::Image(reinterpret_cast<ImTextureID>(
                             get_imgui_set(linear_sampler_->sampler, t.view().view())),
                         ImVec2(window_size.x, scaled_height));
          }
          ImGui::PopID();
        }
      }
    }
  }
  ImGui::End();
}

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

SceneHandle VkRender2::load_scene(const std::filesystem::path& path, bool dynamic) {
  auto ret = gfx::load_gltf(path, default_mat_data_);
  if (!ret.has_value()) {
    return {};
  }
  auto res = std::move(ret.value());

  std::vector<mat4> transforms;
  std::vector<u32> material_ids;
  transforms.reserve(res.scene_graph_data.mesh_node_indices.size());
  material_ids.reserve(res.scene_graph_data.mesh_node_indices.size());
  for (auto& node : res.scene_graph_data.node_datas) {
    for (auto& mesh_indices : node.meshes) {
      transforms.emplace_back(node.world_transform);
      material_ids.emplace_back(mesh_indices.material_id);
    }
  }
  u64 transforms_size = transforms.size() * sizeof(mat4);
  u64 material_indices_size = material_ids.size() * sizeof(u32);
  u64 material_data_size = res.materials.size() * sizeof(gfx::Material);
  u64 vertices_size = res.vertices.size() * sizeof(gfx::Vertex);
  u64 indices_size = res.indices.size() * sizeof(u32);

  std::vector<VkDrawIndexedIndirectCommand> cmds;
  cmds.reserve(res.scene_graph_data.mesh_node_indices.size());

  if (!dynamic) {
    u64 vertices_gpu_offset = static_vertex_buf_->alloc(vertices_size);
    u64 indices_gpu_offset = static_index_buf_->alloc(indices_size);
    u32 i = 0;
    for (auto& node : res.scene_graph_data.node_datas) {
      for (auto& mesh_indices : node.meshes) {
        auto& mesh = res.mesh_draw_infos[mesh_indices.mesh_idx];
        cmds.emplace_back(VkDrawIndexedIndirectCommand{
            .indexCount = mesh.index_count,
            .instanceCount = 1,
            .firstIndex = static_cast<u32>((indices_gpu_offset / sizeof(u32)) + mesh.first_index),
            .vertexOffset = static_cast<i32>(vertices_gpu_offset + mesh.first_vertex),
            .firstInstance = i++});
      }
    }
    u64 cmds_buf_size = cmds.size() * sizeof(VkDrawIndexedIndirectCommand);
    // add transforms to static gpu buffer

    auto staging = LinearStagingBuffer{vk2::StagingBufferPool::get().acquire(
        cmds_buf_size + transforms_size + material_indices_size + material_data_size +
        vertices_size + indices_size)};
    u64 cmds_staging_offset = staging.copy(cmds.data(), cmds_buf_size);
    u64 transforms_staging_offset = staging.copy(transforms.data(), transforms_size);
    u64 material_ids_staging_offset = staging.copy(material_ids.data(), material_indices_size);
    u64 material_data_staging_offset = staging.copy(res.materials.data(), material_data_size);
    u64 vertices_staging_offset = staging.copy(res.vertices.data(), vertices_size);
    u64 indices_staging_offset = staging.copy(res.indices.data(), indices_size);

    auto copy_buffer = [](VkCommandBuffer cmd, const Buffer& src_buffer, const Buffer& dst_buffer,
                          u64 src_offset, u64 dst_offset, u64 size) {
      VkBufferCopy2KHR copy{.sType = VK_STRUCTURE_TYPE_BUFFER_COPY_2_KHR,
                            .srcOffset = src_offset,
                            .dstOffset = dst_offset,
                            .size = size};
      VkCopyBufferInfo2KHR copy_info{.sType = VK_STRUCTURE_TYPE_COPY_BUFFER_INFO_2,
                                     .srcBuffer = src_buffer.buffer(),
                                     .dstBuffer = dst_buffer.buffer(),
                                     .regionCount = 1,
                                     .pRegions = &copy};
      vkCmdCopyBuffer2KHR(cmd, &copy_info);
    };

    immediate_submit([&, this](VkCommandBuffer cmd) {
      u64 cmds_gpu_offset = static_draw_cmds_buf_->alloc(cmds_buf_size);
      copy_buffer(cmd, *staging.get_buffer(), static_draw_cmds_buf_->buffer, cmds_staging_offset,
                  cmds_gpu_offset, cmds_buf_size);
      u64 transforms_gpu_offset = static_transforms_buf_->alloc(transforms_size);
      copy_buffer(cmd, *staging.get_buffer(), static_transforms_buf_->buffer,
                  transforms_staging_offset, transforms_gpu_offset, transforms_size);
      u64 material_ids_gpu_offset = static_material_indices_buf_->alloc(material_indices_size);
      copy_buffer(cmd, *staging.get_buffer(), static_material_indices_buf_->buffer,
                  material_ids_staging_offset, material_ids_gpu_offset, material_indices_size);
      u64 material_data_gpu_offset = static_materials_buf_->alloc(material_data_size);
      copy_buffer(cmd, *staging.get_buffer(), static_materials_buf_->buffer,
                  material_data_staging_offset, material_data_gpu_offset, material_data_size);
      copy_buffer(cmd, *staging.get_buffer(), static_vertex_buf_->buffer, vertices_staging_offset,
                  vertices_gpu_offset, vertices_size);
      copy_buffer(cmd, *staging.get_buffer(), static_index_buf_->buffer, indices_staging_offset,
                  indices_gpu_offset, indices_size);
    });
    draw_cnt_ += cmds.size();
    static_textures_.reserve(static_textures_.size() + res.textures.size());
    for (auto& t : res.textures) {
      static_textures_.emplace_back(std::move(t));
    }
    StagingBufferPool::get().free(staging.get_buffer());

    return {};
  }

  u32 i = 0;
  for (auto& node : res.scene_graph_data.node_datas) {
    for (auto& mesh_indices : node.meshes) {
      auto& mesh = res.mesh_draw_infos[mesh_indices.mesh_idx];
      cmds.emplace_back(
          VkDrawIndexedIndirectCommand{.indexCount = mesh.index_count,
                                       .instanceCount = 1,
                                       .firstIndex = mesh.first_index,
                                       .vertexOffset = static_cast<i32>(mesh.first_vertex),
                                       .firstInstance = i++});
    }
  }
  u64 draw_indirect_buf_size = cmds.size() * sizeof(VkDrawIndexedIndirectCommand);
  SceneHandle handle{loaded_dynamic_scenes_.size()};
  loaded_dynamic_scenes_.emplace_back(LoadedScene{
      std::move(res.scene_graph_data),
      std::make_unique<SceneGPUResources>(
          create_storage_buffer(vertices_size),
          Buffer{BufferCreateInfo{
              .size = indices_size,
              .usage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
          }},
          create_storage_buffer(res.materials.size() * sizeof(gfx::Material)),
          Buffer{BufferCreateInfo{
              .size = draw_indirect_buf_size,
              .usage = VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
          }},
          create_storage_buffer(material_indices_size), create_storage_buffer(transforms_size),
          std::move(res.textures), cmds.size()),
  });

  auto& gltf_result = loaded_dynamic_scenes_[handle.get()];

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

  vk2::Buffer* staging = vk2::StagingBufferPool::get().acquire(
      draw_indirect_buf_size + transforms_size + material_data_size + material_indices_size +
      vertices_size + indices_size);
  u64 offset = 0;
  memcpy(staging->mapped_data(), cmds.data(), draw_indirect_buf_size);
  offset += draw_indirect_buf_size;
  memcpy((char*)staging->mapped_data() + offset, transforms.data(), transforms_size);
  offset += transforms_size;
  memcpy((char*)staging->mapped_data() + offset, res.materials.data(), material_data_size);
  offset += material_data_size;
  memcpy((char*)staging->mapped_data() + offset, material_ids.data(), material_indices_size);
  offset += material_indices_size;
  u64 vertices_staging_offset = offset;
  memcpy((char*)staging->mapped_data() + offset, res.vertices.data(), vertices_size);
  offset += vertices_size;
  u64 indices_staging_offset = offset;
  memcpy((char*)staging->mapped_data() + offset, res.indices.data(), indices_size);

  {
    VkCommandBuffer cmd = transfer_queue_manager_->get_cmd_buffer();
    {
      auto info = vk2::init::command_buffer_begin_info();
      VK_CHECK(vkResetCommandBuffer(cmd, 0));
      VK_CHECK(vkBeginCommandBuffer(cmd, &info));
      copy_buffer(cmd, staging->buffer(), resources->vertex_buffer.buffer(),
                  vertices_staging_offset, 0, vertices_size);
      copy_buffer(cmd, staging->buffer(), resources->index_buffer.buffer(), indices_staging_offset,
                  0, indices_size);
      copy_buffer(cmd, staging->buffer(), resources->draw_indirect_buffer.buffer(), 0, 0,
                  draw_indirect_buf_size);
      copy_buffer(cmd, staging->buffer(), resources->instance_buffer.buffer(),
                  draw_indirect_buf_size, 0, transforms_size);
      copy_buffer(cmd, staging->buffer(), resources->materials_buffer.buffer(),
                  draw_indirect_buf_size + transforms_size, 0, material_data_size);
      copy_buffer(cmd, staging->buffer(), resources->material_indices.buffer(),
                  draw_indirect_buf_size + transforms_size + material_data_size, 0,
                  material_indices_size);
      {
        transfer_q_state_.reset(cmd)
            .queue_transfer_buffer(state_, VK_PIPELINE_STAGE_2_VERTEX_INPUT_BIT,
                                   VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT,
                                   resources->vertex_buffer.buffer(), queues_.transfer_queue_idx,
                                   queues_.graphics_queue_idx)
            .queue_transfer_buffer(state_, VK_PIPELINE_STAGE_2_INDEX_INPUT_BIT,
                                   VK_ACCESS_2_INDEX_READ_BIT, resources->index_buffer.buffer(),
                                   queues_.transfer_queue_idx, queues_.graphics_queue_idx)
            .queue_transfer_buffer(state_, VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT,
                                   VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT,
                                   resources->draw_indirect_buffer.buffer(),
                                   queues_.transfer_queue_idx, queues_.graphics_queue_idx)
            .queue_transfer_buffer(state_, VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT,
                                   VK_ACCESS_2_SHADER_READ_BIT, resources->instance_buffer.buffer(),
                                   queues_.transfer_queue_idx, queues_.graphics_queue_idx)
            .queue_transfer_buffer(state_, VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT,
                                   VK_ACCESS_2_SHADER_READ_BIT,
                                   resources->materials_buffer.buffer(), queues_.transfer_queue_idx,
                                   queues_.graphics_queue_idx)
            .queue_transfer_buffer(state_, VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT,
                                   VK_ACCESS_2_SHADER_READ_BIT,
                                   resources->material_indices.buffer(), queues_.transfer_queue_idx,
                                   queues_.graphics_queue_idx)
            .flush_barriers();
      }
      VK_CHECK(vkEndCommandBuffer(cmd));

      VkFence transfer_fence = vk2::FencePool::get().allocate(true);
      auto cmd_buf_submit_info = vk2::init::command_buffer_submit_info(cmd);

      auto wait_info = vk2::init::semaphore_submit_info(transfer_queue_manager_->submit_semaphore_,
                                                        VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT);
      auto submit = init::queue_submit_info(SPAN1(cmd_buf_submit_info), {}, SPAN1(wait_info));
      LINFO("submit main graphics queue");
      VK_CHECK(vkQueueSubmit2KHR(queues_.transfer_queue, 1, &submit, transfer_fence));
      transfer_queue_manager_->submit_signaled_ = true;
      pending_buffer_transfers_.emplace(staging, nullptr);
    }
  }
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
VkDescriptorSet VkRender2::get_imgui_set(VkSampler sampler, VkImageView view) {
  auto tup = std::make_tuple(sampler, view);
  auto hash = vk2::detail::hashing::hash<decltype(tup)>{}(tup);
  auto it = imgui_desc_sets_.find(hash);
  if (it != imgui_desc_sets_.end()) {
    return it->second;
  }
  VkDescriptorSet imgui_set =
      ImGui_ImplVulkan_AddTexture(sampler, view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
  return imgui_desc_sets_.emplace(hash, imgui_set).first->second;
}
