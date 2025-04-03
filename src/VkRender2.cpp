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
#include "shaders/common.h.glsl"
#include "shaders/debug/basic_common.h.glsl"
#include "shaders/shadow_depth_common.h.glsl"
#include "util/CVar.hpp"
#include "util/IndexAllocator.hpp"
#include "vk2/BindlessResourceAllocator.hpp"
#include "vk2/Buffer.hpp"
#include "vk2/Device.hpp"
#include "vk2/Fence.hpp"
#include "vk2/Hash.hpp"
#include "vk2/Initializers.hpp"
#include "vk2/PipelineManager.hpp"
#include "vk2/Rendering.hpp"
#include "vk2/SamplerCache.hpp"
#include "vk2/StagingBufferPool.hpp"
#include "vk2/Texture.hpp"
#include "vk2/VkCommon.hpp"

namespace {
VkRender2* instance{};

AutoCVarInt ao_map_enabled{"renderer.ao_map", "AO Map", 1, CVarFlags::EditCheckbox};
AutoCVarInt normal_map_enabled{"renderer.normal_map", "Normal Map", 1, CVarFlags::EditCheckbox};

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
  allocator_ = vk2::get_device().allocator();

  vk2::StagingBufferPool::init();
  vk2::BindlessResourceAllocator::init(device_, vk2::get_device().allocator());
  main_set_ = vk2::BindlessResourceAllocator::get().main_set();
  main_set2_ = vk2::BindlessResourceAllocator::get().main_set2_;

  imm_cmd_pool_ = vk2::get_device().create_command_pool(queues_.graphics_queue_idx);
  imm_cmd_buf_ = vk2::get_device().create_command_buffer(imm_cmd_pool_);
  main_del_q_.push([this]() { vk2::get_device().destroy_command_pool(imm_cmd_pool_); });
  main_del_q_.push([]() {
    vk2::PipelineManager::shutdown();
    vk2::StagingBufferPool::destroy();
  });

  VkPushConstantRange default_range{.stageFlags = VK_SHADER_STAGE_ALL, .offset = 0, .size = 128};
  // TODO: refactor
  VkDescriptorSetLayout main_set_layout = vk2::BindlessResourceAllocator::get().main_set_layout();
  VkDescriptorSetLayout layouts[] = {main_set_layout,
                                     vk2::BindlessResourceAllocator::get().main_set2_layout_};
  VkPipelineLayoutCreateInfo pipeline_info{.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                                           .setLayoutCount = COUNTOF(layouts),
                                           .pSetLayouts = layouts,
                                           .pushConstantRangeCount = 1,
                                           .pPushConstantRanges = &default_range};
  VK_CHECK(vkCreatePipelineLayout(device_, &pipeline_info, nullptr, &default_pipeline_layout_));
  main_del_q_.push(
      [this]() { vkDestroyPipelineLayout(device_, default_pipeline_layout_, nullptr); });
  vk2::PipelineManager::init(device_, resource_dir_ / "shaders", true, default_pipeline_layout_);

  create_attachment_imgs();

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
      .maxLod = 1,
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

  static_vertex_buf_ = LinearBuffer{create_storage_buffer(10'000'000 * sizeof(gfx::Vertex))};
  static_index_buf_ = LinearBuffer{BufferCreateInfo{
      .size = 10'000'000 * sizeof(u32),
      .usage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
  }};
  static_materials_buf_ = LinearBuffer{create_storage_buffer(10'000 * sizeof(gfx::Material))};
  // TODO: tune and/or resizable buffers
  u64 max_static_draws = 100'000;
  static_instance_data_buf_ =
      LinearBuffer{create_storage_buffer(max_static_draws * sizeof(InstanceData))};
  static_object_data_buf_ = SlotBuffer<gfx::ObjectData>{
      create_storage_buffer(max_static_draws * sizeof(gfx::ObjectData))};
  static_draw_info_buf_ = SlotBuffer<DrawInfo>{BufferCreateInfo{
      .size = max_static_draws * sizeof(DrawInfo),
      .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT |
               VK_BUFFER_USAGE_TRANSFER_DST_BIT,
  }};

  init_pipelines();

  init_indirect_drawing();
  csm_ = CSM(default_pipeline_layout_);
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
    data.view = info.view;
    data.debug_flags = uvec4{};
    if (ao_map_enabled.get()) {
      data.debug_flags.x |= AO_ENABLED_BIT;
    }
    if (normal_map_enabled.get()) {
      data.debug_flags.x |= NORMAL_MAPS_ENABLED_BIT;
    }
    data.light_color = info.light_color;
    data.light_dir = glm::normalize(info.light_dir);
    data.debug_flags.w = debug_mode_;
    data.view_pos = info.view_pos;
    data.ambient_intensity = info.ambient_intensity;
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

  // TODO: better management lol
  bool portable = true;
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
      TracyVkZone(curr_frame().tracy_vk_ctx, cmd, "img comp");
      struct {
        uint idx;
        float t;
      } pc{img_->view().storage_img_resource().handle, static_cast<f32>(glfwGetTime())};
      ctx.bind_compute_pipeline(PipelineManager::get().get(img_pipeline_)->pipeline);
      ctx.push_constants(default_pipeline_layout_, sizeof(pc), &pc);
      ctx.dispatch((img_->extent().width + 16) / 16, (img_->extent().height + 16) / 16, 1);
    }

    {
      TracyVkZone(curr_frame().tracy_vk_ctx, cmd, "Cull Objects");
      state_
          .buffer_barrier(draw_cnt_buf_.value(), VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                          VK_ACCESS_2_TRANSFER_WRITE_BIT)
          .flush_barriers();
      vkCmdFillBuffer(cmd, draw_cnt_buf_->buffer(), 0, sizeof(u32), 0);
      // clear final buffer if we can't use drawindirectcount
      if (portable) {
        state_
            .buffer_barrier(final_draw_cmd_buf_.value(), VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                            VK_ACCESS_2_TRANSFER_WRITE_BIT)
            .flush_barriers();
        vkCmdFillBuffer(cmd, final_draw_cmd_buf_->buffer(), 0, final_draw_cmd_buf_->size(), 0);
      }

      // cull
      state_
          .buffer_barrier(static_draw_info_buf_->buffer, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                          VK_ACCESS_2_SHADER_READ_BIT)
          .buffer_barrier(static_object_data_buf_->buffer, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                          VK_ACCESS_2_SHADER_READ_BIT)
          .buffer_barrier(final_draw_cmd_buf_.value(), VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                          VK_ACCESS_2_SHADER_WRITE_BIT)
          .buffer_barrier(draw_cnt_buf_.value(), VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                          VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT)
          .flush_barriers();

      PipelineManager::get().bind_compute(cmd, cull_objs_pipeline_);
      struct {
        u32 num_objs;
        u32 in_draw_cmds_buf;
        u32 out_draw_cmds_buf;
        u32 draw_cnt_buf;
        u32 object_bounds_buf;
      } pc{static_cast<u32>(draw_cnt_), static_draw_info_buf_->buffer.resource_info_->handle,
           final_draw_cmd_buf_->resource_info_->handle, draw_cnt_buf_->resource_info_->handle,
           static_object_data_buf_->buffer.resource_info_->handle};
      ctx.push_constants(default_pipeline_layout_, sizeof(pc), &pc);
      vkCmdDispatch(cmd, (draw_cnt_ + 256) / 256, 1, 1);
    }
    state_
        .buffer_barrier(draw_cnt_buf_.value(), VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                        VK_ACCESS_2_MEMORY_READ_BIT)
        .buffer_barrier(final_draw_cmd_buf_.value(), VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT,
                        VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT)
        .flush_barriers();

    {
      TracyVkZone(curr_frame().tracy_vk_ctx, cmd, "CascadeShadowPass");
      csm_->render(state_, cmd, curr_frame_num(), info.view, info.light_dir, aspect_ratio(),
                   info.fov_degrees, [&, this](const mat4& vp_matrix) {
                     ShadowDepthPushConstants pc{
                         vp_matrix,
                         static_vertex_buf_->buffer.resource_info_->handle,
                         static_instance_data_buf_->buffer.resource_info_->handle,
                         static_object_data_buf_->buffer.resource_info_->handle,
                     };
                     ctx.push_constants(default_pipeline_layout_, sizeof(pc), &pc);
                     vkCmdBindIndexBuffer(cmd, static_index_buf_->buffer.buffer(), 0,
                                          VK_INDEX_TYPE_UINT32);
                     if (portable) {
                       vkCmdDrawIndexedIndirect(cmd, final_draw_cmd_buf_->buffer(), 0, draw_cnt_,
                                                sizeof(VkDrawIndexedIndirectCommand));
                     } else {
                       vkCmdDrawIndexedIndirectCount(cmd, final_draw_cmd_buf_->buffer(), 0,
                                                     draw_cnt_buf_->buffer(), 0, max_draws,
                                                     sizeof(VkDrawIndexedIndirectCommand));
                     }
                     // vkCmdDrawIndexedIndirect(cmd, static_draw_cmds_buf_->buffer.buffer(), 0,
                     // draw_cnt_,
                     //                          sizeof(VkDrawIndexedIndirectCommand));
                   });
    }

    csm_->debug_shadow_pass(state_, cmd, linear_sampler_.value());

    // draw
    {
      state_
          .transition(
              img_->image(), VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
              VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
              VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_ASPECT_COLOR_BIT)
          .buffer_barrier(csm_->get_shadow_data_buffer(curr_frame_num()).buffer(),
                          VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT)
          .transition(csm_->get_shadow_img().image(), VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                      VK_ACCESS_2_SHADER_READ_BIT, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                      VK_IMAGE_ASPECT_DEPTH_BIT)
          .transition(depth_img_->image(),
                      VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT |
                          VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
                      VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                      VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, VK_IMAGE_ASPECT_DEPTH_BIT)
          .flush_barriers();
      VkClearValue depth_clear{};
      // VkClearValue depth_clear{.depthStencil = {.depth = 1.f}};
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
                              const Buffer& object_data_buffer, const Buffer&, u64) {
        BasicPushConstants pc{curr_frame_2().scene_uniform_buf->resource_info_->handle,
                              vertex_buffer.resource_info_->handle,
                              instance_buffer.resource_info_->handle,
                              object_data_buffer.resource_info_->handle,
                              material_data_buffer.resource_info_->handle,
                              linear_sampler_->resource_info.handle,
                              csm_->get_shadow_data_buffer(curr_frame_num()).resource_info_->handle,
                              csm_->get_shadow_sampler().resource_info.handle,
                              csm_->get_shadow_img().view().sampled_img_resource().handle};
        ctx.push_constants(default_pipeline_layout_, sizeof(pc), &pc);

        vkCmdBindIndexBuffer(cmd, index_buffer.buffer(), 0, VK_INDEX_TYPE_UINT32);
        if (portable) {
          vkCmdDrawIndexedIndirect(cmd, final_draw_cmd_buf_->buffer(), 0, draw_cnt_,
                                   sizeof(VkDrawIndexedIndirectCommand));
        } else {
          vkCmdDrawIndexedIndirectCount(cmd, final_draw_cmd_buf_->buffer(), 0,
                                        draw_cnt_buf_->buffer(), 0, max_draws,
                                        sizeof(VkDrawIndexedIndirectCommand));
        }
        // vkCmdDrawIndexedIndirect(cmd, static_draw_cmds_buf_->buffer.buffer(), 0, draw_cnt_,
        //                          sizeof(VkDrawIndexedIndirectCommand));
      };

      TracyVkZone(curr_frame().tracy_vk_ctx, cmd, "Final Draw Pass");
      if (draw_cnt_) {
        draw_objects(static_vertex_buf_->buffer, static_index_buf_->buffer,
                     static_instance_data_buf_->buffer, static_materials_buf_->buffer,
                     static_object_data_buf_->buffer, static_draw_info_buf_->buffer, draw_cnt_);
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
        transfer_queue_manager_->submit_semaphore_, VK_PIPELINE_STAGE_2_TRANSFER_BIT,
        transfer_queue_manager_->semaphore_value_);
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
    if (ImGui::TreeNodeEx("stats", ImGuiTreeNodeFlags_DefaultOpen)) {
      if (ImGui::TreeNodeEx("static geometry", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Text("Total vertices: %lu", (size_t)static_draw_stats_.total_vertices);
        ImGui::Text("Total indices: %lu", (size_t)static_draw_stats_.total_indices);
        ImGui::Text("Total triangles: %lu", (size_t)static_draw_stats_.total_vertices / 3);
        ImGui::Text("Vertices %u", static_draw_stats_.vertices);
        ImGui::Text("Indices: %u", static_draw_stats_.indices);
        ImGui::Text("Draw Cmds: %u", static_draw_stats_.draw_cmds);
        ImGui::Text("Materials: %u", static_draw_stats_.materials);
        ImGui::Text("Textures: %u", static_draw_stats_.textures);
        ImGui::TreePop();
      }
      ImGui::TreePop();
    }
    if (ImGui::TreeNode("CSM")) {
      csm_->on_imgui(linear_sampler_->sampler);
      ImGui::TreePop();
    }

    if (ImGui::BeginCombo("Debug Mode", debug_mode_to_string(debug_mode_))) {
      for (u32 mode = 0; mode < DEBUG_MODE_COUNT; mode++) {
        if (ImGui::Selectable(debug_mode_to_string(mode), mode == debug_mode_)) {
          debug_mode_ = mode;
        }
      }
      ImGui::EndCombo();
    }
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

SceneHandle VkRender2::load_scene(const std::filesystem::path& path, bool dynamic,
                                  const mat4& transform) {
  ZoneScoped;
  if (!dynamic) {
    ZoneScopedN("load_scene");
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
    auto it = static_scenes_.find(path);
    StaticSceneGPUResources* resources{};
    if (it != static_scenes_.end()) {
      resources = &it->second;
    } else {
      auto ret = gfx::load_gltf(path, default_mat_data_);
      if (!ret.has_value()) {
        return {};
      }
      auto res = std::move(ret.value());

      u64 material_data_size = res.materials.size() * sizeof(gfx::Material);
      u64 vertices_size = res.vertices.size() * sizeof(gfx::Vertex);
      u64 indices_size = res.indices.size() * sizeof(u32);
      u64 vertices_gpu_offset = static_vertex_buf_->alloc(vertices_size);
      u64 indices_gpu_offset = static_index_buf_->alloc(indices_size);

      auto staging = LinearStagingBuffer{
          vk2::StagingBufferPool::get().acquire(material_data_size + vertices_size + indices_size)};
      u64 material_data_staging_offset = staging.copy(res.materials.data(), material_data_size);
      u64 vertices_staging_offset = staging.copy(res.vertices.data(), vertices_size);
      u64 indices_staging_offset = staging.copy(res.indices.data(), indices_size);

      static_draw_stats_.vertices += res.vertices.size();
      static_draw_stats_.indices += res.indices.size();
      static_draw_stats_.textures += res.textures.size();
      static_draw_stats_.materials += res.materials.size();
      u64 material_data_gpu_offset = static_materials_buf_->alloc(material_data_size);

      transfer_submit([&, this](VkCommandBuffer cmd, VkFence fence,
                                std::queue<InFlightResource<vk2::Buffer*>>& transfers) {
        copy_buffer(cmd, *staging.get_buffer(), static_materials_buf_->buffer,
                    material_data_staging_offset, material_data_gpu_offset, material_data_size);
        copy_buffer(cmd, *staging.get_buffer(), static_vertex_buf_->buffer, vertices_staging_offset,
                    vertices_gpu_offset, vertices_size);
        copy_buffer(cmd, *staging.get_buffer(), static_index_buf_->buffer, indices_staging_offset,
                    indices_gpu_offset, indices_size);

        transfer_q_state_.reset(cmd)
            .queue_transfer_buffer(state_, VK_PIPELINE_STAGE_2_VERTEX_INPUT_BIT,
                                   VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT,
                                   static_vertex_buf_->buffer.buffer(), queues_.transfer_queue_idx,
                                   queues_.graphics_queue_idx)
            .queue_transfer_buffer(state_, VK_PIPELINE_STAGE_2_INDEX_INPUT_BIT,
                                   VK_ACCESS_2_INDEX_READ_BIT, static_vertex_buf_->buffer.buffer(),
                                   queues_.transfer_queue_idx, queues_.graphics_queue_idx)
            .queue_transfer_buffer(state_, VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT,
                                   VK_ACCESS_2_SHADER_READ_BIT,
                                   static_materials_buf_->buffer.buffer(),
                                   queues_.transfer_queue_idx, queues_.graphics_queue_idx)
            .flush_barriers();
        transfers.emplace(staging.get_buffer(), fence);
      });

      static_textures_.reserve(static_textures_.size() + res.textures.size());
      for (auto& t : res.textures) {
        static_textures_.emplace_back(std::move(t));
      }
      resources = &static_scenes_
                       .emplace(path.string(),
                                StaticSceneGPUResources{
                                    .scene_graph_data = std::move(res.scene_graph_data),
                                    .mesh_draw_infos = std::move(res.mesh_draw_infos),
                                    .first_vertex = vertices_gpu_offset / sizeof(gfx::Vertex),
                                    .first_index = indices_gpu_offset / sizeof(u32),
                                    .num_vertices = res.vertices.size(),
                                    .num_indices = res.indices.size(),
                                    .materials_idx_offset =
                                        material_data_gpu_offset / sizeof(gfx::Material)})
                       .first->second;
    }
    static_draw_stats_.total_vertices += resources->num_vertices;
    static_draw_stats_.total_indices += resources->num_indices;

    std::vector<gfx::ObjectData> obj_datas;
    std::vector<InstanceData> instance_datas;
    obj_datas.reserve(resources->scene_graph_data.mesh_node_indices.size());
    instance_datas.reserve(resources->scene_graph_data.mesh_node_indices.size());
    ObjectDraw obj_draw;
    obj_draw.obj_data_slots.reserve(obj_datas.size());
    bool mult_transform = transform != mat4{1};
    for (auto& node : resources->scene_graph_data.node_datas) {
      for (auto& mesh_indices : node.meshes) {
        obj_datas.emplace_back(gfx::ObjectData{
            .model = mult_transform ? transform * node.world_transform : node.world_transform});
        obj_draw.obj_data_slots.emplace_back(static_object_data_buf_->allocator.alloc());
        instance_datas.emplace_back(mesh_indices.material_id + resources->materials_idx_offset,
                                    obj_draw.obj_data_slots.back().idx());
      }
    }
    obj_data_cnt_ += obj_datas.size();

    u64 obj_datas_size = obj_datas.size() * sizeof(gfx::ObjectData);
    u64 instance_datas_size = instance_datas.size() * sizeof(InstanceData);

    std::vector<DrawInfo> cmds;
    cmds.reserve(resources->scene_graph_data.mesh_node_indices.size());

    for (auto& node : resources->scene_graph_data.node_datas) {
      for (auto& mesh_indices : node.meshes) {
        auto& mesh = resources->mesh_draw_infos[mesh_indices.mesh_idx];
        cmds.emplace_back(
            DrawInfo{.index_cnt = mesh.index_count,
                     .first_index = static_cast<u32>(resources->first_index + mesh.first_index),
                     .vertex_offset = static_cast<u32>(resources->first_vertex + mesh.first_vertex),
                     .pad = 0});
      }
    }

    u64 cmds_buf_size = cmds.size() * sizeof(DrawInfo);

    auto staging = LinearStagingBuffer{vk2::StagingBufferPool::get().acquire(
        cmds_buf_size + obj_datas_size + instance_datas_size)};
    u64 cmds_staging_offset = staging.copy(cmds.data(), cmds_buf_size);
    u64 obj_datas_staging_offset = staging.copy(obj_datas.data(), obj_datas_size);
    u64 instance_datas_staging_offset = staging.copy(instance_datas.data(), instance_datas_size);

    transfer_submit([&, this](VkCommandBuffer cmd, VkFence fence,
                              std::queue<InFlightResource<vk2::Buffer*>>& transfers) {
      obj_draw.draw_cmd_slots.reserve(cmds.size());
      for (u64 i = 0; i < cmds.size(); i++) {
        obj_draw.draw_cmd_slots.emplace_back(static_draw_info_buf_->allocator.alloc());
      }
      std::vector<VkBufferCopy2KHR> copies(cmds.size());
      for (u64 cmd_i = 0; cmd_i < cmds.size(); cmd_i++) {
        copies[cmd_i] =
            init::buffer_copy(cmds_staging_offset + (cmd_i * sizeof(DrawInfo)),
                              obj_draw.draw_cmd_slots[cmd_i].offset(), sizeof(DrawInfo));
      }
      vkCmdCopyBuffer2KHR(
          cmd, vk2::addr(VkCopyBufferInfo2KHR{.sType = VK_STRUCTURE_TYPE_COPY_BUFFER_INFO_2,
                                              .srcBuffer = staging.get_buffer()->buffer(),
                                              .dstBuffer = static_draw_info_buf_->buffer.buffer(),
                                              .regionCount = static_cast<u32>(copies.size()),
                                              .pRegions = copies.data()}));
      {
        copies.clear();
        for (u64 obj_data_i = 0; obj_data_i < obj_datas.size(); obj_data_i++) {
          copies.emplace_back(init::buffer_copy(
              obj_datas_staging_offset + (obj_data_i * sizeof(gfx::ObjectData)),
              obj_draw.obj_data_slots[obj_data_i].offset(), sizeof(gfx::ObjectData)));
        }
        vkCmdCopyBuffer2KHR(cmd, vk2::addr(VkCopyBufferInfo2KHR{
                                     .sType = VK_STRUCTURE_TYPE_COPY_BUFFER_INFO_2,
                                     .srcBuffer = staging.get_buffer()->buffer(),
                                     .dstBuffer = static_object_data_buf_->buffer.buffer(),
                                     .regionCount = static_cast<u32>(copies.size()),
                                     .pRegions = copies.data()}));
      }

      // u64 transforms_gpu_offset = static_object_data_buf_->alloc(obj_datas_size);
      // copy_buffer(cmd, *staging.get_buffer(), static_object_data_buf_->buffer,
      //             obj_datas_staging_offset, transforms_gpu_offset, obj_datas_size);

      u64 instance_datas_gpu_offset = static_instance_data_buf_->alloc(instance_datas_size);
      copy_buffer(cmd, *staging.get_buffer(), static_instance_data_buf_->buffer,
                  instance_datas_staging_offset, instance_datas_gpu_offset, instance_datas_size);

      transfer_q_state_.reset(cmd)
          .queue_transfer_buffer(
              state_,
              VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
              VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_2_SHADER_READ_BIT,
              static_draw_info_buf_->buffer.buffer(), queues_.transfer_queue_idx,
              queues_.graphics_queue_idx)
          .queue_transfer_buffer(
              state_,
              VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT,
              VK_ACCESS_2_SHADER_READ_BIT, static_object_data_buf_->buffer.buffer(),
              queues_.transfer_queue_idx, queues_.graphics_queue_idx)
          .queue_transfer_buffer(state_, VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT,
                                 VK_ACCESS_2_SHADER_READ_BIT,
                                 static_instance_data_buf_->buffer.buffer(),
                                 queues_.transfer_queue_idx, queues_.graphics_queue_idx)
          .flush_barriers();
      transfers.emplace(staging.get_buffer(), fence);
    });

    // TODO: only increment draw count when the fence is ready
    draw_cnt_ += cmds.size();
    static_draw_stats_.draw_cmds += cmds.size();
    // });
    return {};
  }
  return {};
  // auto copy_buffer = [](VkCommandBuffer cmd, const Buffer& src_buffer, const Buffer& dst_buffer,
  //                       u64 src_offset, u64 dst_offset, u64 size) {
  //   VkBufferCopy2KHR copy{.sType = VK_STRUCTURE_TYPE_BUFFER_COPY_2_KHR,
  //                         .srcOffset = src_offset,
  //                         .dstOffset = dst_offset,
  //                         .size = size};
  //   VkCopyBufferInfo2KHR copy_info{.sType = VK_STRUCTURE_TYPE_COPY_BUFFER_INFO_2,
  //                                  .srcBuffer = src_buffer.buffer(),
  //                                  .dstBuffer = dst_buffer.buffer(),
  //                                  .regionCount = 1,
  //                                  .pRegions = &copy};
  //   vkCmdCopyBuffer2KHR(cmd, &copy_info);
  // };
  //
  // // TODO: refactor
  // auto ret = gfx::load_gltf(path, default_mat_data_);
  // if (!ret.has_value()) {
  //   return {};
  // }
  // auto res = std::move(ret.value());
  //
  // std::vector<mat4> transforms;
  // std::vector<u32> material_ids;
  // transforms.reserve(res.scene_graph_data.mesh_node_indices.size());
  // material_ids.reserve(res.scene_graph_data.mesh_node_indices.size());
  // u32 material_id_offset = static_draw_stats_.materials;
  // for (auto& node : res.scene_graph_data.node_datas) {
  //   for (auto& mesh_indices : node.meshes) {
  //     transforms.emplace_back(node.world_transform);
  //     material_ids.emplace_back(mesh_indices.material_id + material_id_offset);
  //   }
  // }
  // u64 transforms_size = transforms.size() * sizeof(mat4);
  // u64 material_indices_size = material_ids.size() * sizeof(u32);
  // u64 material_data_size = res.materials.size() * sizeof(gfx::Material);
  // u64 vertices_size = res.vertices.size() * sizeof(gfx::Vertex);
  // u64 indices_size = res.indices.size() * sizeof(u32);
  //
  // std::vector<VkDrawIndexedIndirectCommand> cmds;
  // cmds.reserve(res.scene_graph_data.mesh_node_indices.size());
  //
  // u32 i = 0;
  // for (auto& node : res.scene_graph_data.node_datas) {
  //   for (auto& mesh_indices : node.meshes) {
  //     auto& mesh = res.mesh_draw_infos[mesh_indices.mesh_idx];
  //     cmds.emplace_back(
  //         VkDrawIndexedIndirectCommand{.indexCount = mesh.index_count,
  //                                      .instanceCount = 1,
  //                                      .firstIndex = mesh.first_index,
  //                                      .vertexOffset = static_cast<i32>(mesh.first_vertex),
  //                                      .firstInstance = i++});
  //   }
  // }
  // u64 draw_indirect_buf_size = cmds.size() * sizeof(VkDrawIndexedIndirectCommand);
  // SceneHandle handle{loaded_dynamic_scenes_.size()};
  // loaded_dynamic_scenes_.emplace_back(LoadedScene{
  //     std::move(res.scene_graph_data),
  //     std::make_unique<SceneGPUResources>(
  //         create_storage_buffer(vertices_size),
  //         Buffer{BufferCreateInfo{
  //             .size = indices_size,
  //             .usage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
  //         }},
  //         create_storage_buffer(res.materials.size() * sizeof(gfx::Material)),
  //         Buffer{BufferCreateInfo{
  //             .size = draw_indirect_buf_size,
  //             .usage = VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
  //         }},
  //         create_storage_buffer(material_indices_size), create_storage_buffer(transforms_size),
  //         std::move(res.textures), cmds.size()),
  // });
  //
  // auto& gltf_result = loaded_dynamic_scenes_[handle.get()];
  //
  // auto* resources = gltf_result.resources.get();
  //
  // vk2::Buffer* staging = vk2::StagingBufferPool::get().acquire(
  //     draw_indirect_buf_size + transforms_size + material_data_size + material_indices_size +
  //     vertices_size + indices_size);
  // u64 offset = 0;
  // memcpy(staging->mapped_data(), cmds.data(), draw_indirect_buf_size);
  // offset += draw_indirect_buf_size;
  // memcpy((char*)staging->mapped_data() + offset, transforms.data(), transforms_size);
  // offset += transforms_size;
  // memcpy((char*)staging->mapped_data() + offset, res.materials.data(), material_data_size);
  // offset += material_data_size;
  // memcpy((char*)staging->mapped_data() + offset, material_ids.data(), material_indices_size);
  // offset += material_indices_size;
  // u64 vertices_staging_offset = offset;
  // memcpy((char*)staging->mapped_data() + offset, res.vertices.data(), vertices_size);
  // offset += vertices_size;
  // u64 indices_staging_offset = offset;
  // memcpy((char*)staging->mapped_data() + offset, res.indices.data(), indices_size);
  //
  // {
  //   transfer_submit([&, this](VkCommandBuffer cmd, VkFence fence,
  //                             std::queue<InFlightResource<vk2::Buffer*>>& transfers) {
  //     copy_buffer(cmd, *staging, resources->vertex_buffer, vertices_staging_offset, 0,
  //                 vertices_size);
  //     copy_buffer(cmd, *staging, resources->index_buffer, indices_staging_offset, 0,
  //     indices_size); copy_buffer(cmd, *staging, resources->draw_indirect_buffer, 0, 0,
  //     draw_indirect_buf_size); copy_buffer(cmd, *staging, resources->instance_buffer,
  //     draw_indirect_buf_size, 0,
  //                 transforms_size);
  //     copy_buffer(cmd, *staging, resources->materials_buffer,
  //                 draw_indirect_buf_size + transforms_size, 0, material_data_size);
  //     copy_buffer(cmd, *staging, resources->material_indices,
  //                 draw_indirect_buf_size + transforms_size + material_data_size, 0,
  //                 material_indices_size);
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
  //           .queue_transfer_buffer(state_, VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT,
  //                                  VK_ACCESS_2_SHADER_READ_BIT,
  //                                  resources->materials_buffer.buffer(),
  //                                  queues_.transfer_queue_idx, queues_.graphics_queue_idx)
  //           .queue_transfer_buffer(state_, VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT,
  //                                  VK_ACCESS_2_SHADER_READ_BIT,
  //                                  resources->material_indices.buffer(),
  //                                  queues_.transfer_queue_idx, queues_.graphics_queue_idx)
  //           .flush_barriers();
  //     }
  //     transfers.emplace(staging, fence);
  //   });
  // }
  // return handle;
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
const char* VkRender2::debug_mode_to_string(u32 mode) {
  switch (mode) {
    case DEBUG_MODE_AO_MAP:
      return "AO Map";
    case DEBUG_MODE_NORMALS:
      return "Normals";
    case DEBUG_MODE_CASCADE_LEVELS:
      return "Cascade Levels";
    default:
      return "None";
  }
}

void VkRender2::transfer_submit(
    std::function<void(VkCommandBuffer cmd, VkFence fence,
                       std::queue<InFlightResource<vk2::Buffer*>>&)>&& function) {
  VkCommandBuffer cmd = transfer_queue_manager_->get_cmd_buffer();
  VK_CHECK(vkResetCommandBuffer(cmd, 0));
  auto info = vk2::init::command_buffer_begin_info();
  VK_CHECK(vkBeginCommandBuffer(cmd, &info));
  VkFence transfer_fence = vk2::FencePool::get().allocate(true);
  function(cmd, transfer_fence, pending_buffer_transfers_);
  VK_CHECK(vkEndCommandBuffer(cmd));

  auto cmd_buf_submit_info = vk2::init::command_buffer_submit_info(cmd);

  auto signal_info = vk2::init::semaphore_submit_info(transfer_queue_manager_->submit_semaphore_,
                                                      VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                                                      ++transfer_queue_manager_->semaphore_value_);

  auto submit = init::queue_submit_info(SPAN1(cmd_buf_submit_info), {}, SPAN1(signal_info));
  VK_CHECK(vkQueueSubmit2KHR(queues_.transfer_queue, 1, &submit, transfer_fence));
  transfer_queue_manager_->submit_signaled_ = true;
}

void VkRender2::init_pipelines() {
  img_pipeline_ = PipelineManager::get().load_compute_pipeline(
      {"debug/clear_img.comp", default_pipeline_layout_});
  draw_pipeline_ = PipelineManager::get().load_graphics_pipeline(GraphicsPipelineCreateInfo{
      .vertex_path = "debug/basic.vert",
      .fragment_path = "debug/basic.frag",
      .layout = default_pipeline_layout_,
      .rendering = {.color_formats = {img_->format()},
                    .color_formats_cnt = 1,
                    .depth_format = depth_img_->format()},
      .depth_stencil = GraphicsPipelineCreateInfo::depth_enable(true, CompareOp::GreaterOrEqual),
  });
  assert(draw_pipeline_);
}

void VkRender2::init_indirect_drawing() {
  draw_cnt_buf_ = vk2::Buffer{BufferCreateInfo{
      .size = sizeof(u32),
      .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT |
               VK_BUFFER_USAGE_TRANSFER_DST_BIT,
  }};
  final_draw_cmd_buf_ = vk2::Buffer{BufferCreateInfo{
      .size = max_draws * sizeof(VkDrawIndexedIndirectCommand),
      .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT |
               VK_BUFFER_USAGE_TRANSFER_DST_BIT,
  }};
  cull_objs_pipeline_ = PipelineManager::get().load_compute_pipeline({"cull_objects.comp"});
}
