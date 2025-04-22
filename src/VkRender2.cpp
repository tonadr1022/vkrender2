#include "VkRender2.hpp"

// clang-format off
#include <volk.h>
#include <vulkan/vk_enum_string_helper.h>
#include <vulkan/vulkan_core.h>
#include <imgui_impl_vulkan.h>
// clang-format on

#include <cassert>
#include <filesystem>
#include <tracy/Tracy.hpp>
#include <tracy/TracyVulkan.hpp>

#include "CommandEncoder.hpp"
#include "GLFW/glfw3.h"
#include "Logger.hpp"
#include "RenderGraph.hpp"
#include "SceneLoader.hpp"
#include "StateTracker.hpp"
#include "Timer.hpp"
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
#include "vk2/VkTypes.hpp"

namespace gfx {

namespace {
VkRender2* vkrender2_instance{};

AutoCVarInt ao_map_enabled{"renderer.ao_map", "AO Map", 1, CVarFlags::EditCheckbox};
AutoCVarInt postprocess_pass_enabled{"renderer.postprocess_pass_enabled",
                                     "PostProcess Pass Enabled", 1, CVarFlags::EditCheckbox};
AutoCVarInt tonemap_enabled{"renderer.tonemap_enabled", "Tonemapping Enabled", 1,
                            CVarFlags::EditCheckbox};
AutoCVarInt gammacorrect_enabled{"renderer.gammacorrect_enabled", "Gamma Correction Enabled", 1,
                                 CVarFlags::EditCheckbox};

AutoCVarInt convoluted_skybox{"renderer.convoluted_skybox", "convoluted_skybox", 1,
                              CVarFlags::EditCheckbox};
AutoCVarInt normal_map_enabled{"renderer.normal_map", "Normal Map", 1, CVarFlags::EditCheckbox};

// clang-format off
gfx::Vertex cube_vertices[] = {
    {{ -1.0f, -1.0f, -1.0f, }}, {{ 1.0f, 1.0f, -1.0f, }}, {{ 1.0f, -1.0f, -1.0f, }}, {{ 1.0f, 1.0f, -1.0f, }},
    {{ -1.0f, -1.0f, -1.0f, }}, {{ -1.0f, 1.0f, -1.0f, }}, {{ -1.0f, -1.0f, 1.0f, }}, {{ 1.0f, -1.0f, 1.0f, }}, {{
        1.0f, 1.0f, 1.0f, }}, {{ 1.0f, 1.0f, 1.0f, }}, {{ -1.0f, 1.0f, 1.0f, }}, {{ -1.0f, -1.0f, 1.0f, }}, {{ -1.0f,
        1.0f, 1.0f, }}, {{ -1.0f, 1.0f, -1.0f, }}, {{ -1.0f, -1.0f, -1.0f, }}, {{ -1.0f, -1.0f, -1.0f, }}, {{ -1.0f,
        -1.0f, 1.0f, }}, {{ -1.0f, 1.0f, 1.0f, }}, {{ 1.0f, 1.0f, 1.0f, }}, {{ 1.0f, -1.0f, -1.0f, }}, {{ 1.0f, 1.0f,
        -1.0f, }}, {{ 1.0f, -1.0f, -1.0f, }}, {{ 1.0f, 1.0f, 1.0f, }}, {{ 1.0f, -1.0f, 1.0f, }}, {{ -1.0f, -1.0f, -1.0f, }}, {{ 1.0f, -1.0f, -1.0f, }}, {{
        1.0f, -1.0f, 1.0f, }}, {{ 1.0f, -1.0f, 1.0f, }}, {{ -1.0f, -1.0f, 1.0f, }}, {{ -1.0f, -1.0f, -1.0f, }}, {{ -1.0f, 1.0f, -1.0f, }}, {{
        1.0f, 1.0f, 1.0f, }}, {{ 1.0f, 1.0f, -1.0f, }}, {{ 1.0f, 1.0f, 1.0f, }}, {{ -1.0f, 1.0f, -1.0f, }}, {{ -1.0f,
        1.0f, 1.0f, }},
};

// clang-format on
// constexpr gfx::Vertex cube_vertices[] = {
//     {.pos = {-1.0, -1.0, 1.0}}, {.pos = {1.0, -1.0, 1.0}},   {.pos = {-1.0, 1.0, 1.0}},
//     {.pos = {1.0, 1.0, 1.0}},   {.pos = {-1.0, -1.0, -1.0}}, {.pos = {1.0, -1.0, -1.0}},
//     {.pos = {-1.0, 1.0, -1.0}}, {.pos = {1.0, 1.0, -1.0}},
// };
// constexpr u32 cube_indices[] = {0, 1, 2, 3, 7, 1, 5, 4, 7, 6, 2, 4, 0, 1};

}  // namespace

VkRender2& VkRender2::get() {
  assert(vkrender2_instance);
  return *vkrender2_instance;
}

void VkRender2::init(const InitInfo& info) {
  assert(!vkrender2_instance);
  new VkRender2{info};
}

void VkRender2::shutdown() {
  assert(vkrender2_instance);
  delete vkrender2_instance;
}

using namespace vk2;

VkRender2::VkRender2(const InitInfo& info)
    : BaseRenderer(info, BaseRenderer::BaseInitInfo{.frames_in_flight = 2}) {
  vkrender2_instance = this;
  allocator_ = vk2::get_device().allocator();

  vk2::StagingBufferPool::init();
  vk2::BindlessResourceAllocator::init(device_, vk2::get_device().allocator());
  main_set_ = vk2::BindlessResourceAllocator::get().main_set();
  main_set2_ = vk2::BindlessResourceAllocator::get().main_set2_;

  PrintTimerMS timer;
  // auto& comp1 = rg_.add_pass("comp1", [this](CmdEncoder& cmd) {
  //   // cmd.bind_compute_pipeline(PipelineManager::get().get(img_pipeline_)->pipeline);
  //   // struct {
  //   //   u32 img_idx;
  //   //   float time;
  //   // } pc{
  //   //
  //   // };
  //   // cmd.dispatch((swapchain_.dims.x + 16) / 16, (swapchain_.dims.y + 16) / 16, 1);
  //   LINFO("executing comp1");
  // });
  // comp1.add_color_output("computed1", {.format = Format::R32G32B32A32Sfloat});
  //
  // auto& comp2 = rg_.add_pass("comp2", [this](CmdEncoder& cmd) {
  //   LINFO("executing comp2");
  //   // cmd.bind_compute_pipeline(PipelineManager::get().get(img_pipeline_)->pipeline);
  //   // cmd.dispatch((swapchain_.dims.x + 16) / 16, (swapchain_.dims.y + 16) / 16, 1);
  // });
  // comp2.add_texture_input("computed1");
  // comp2.add_color_output("final_out", {.format = Format::R32G32B32A32Sfloat});

  // auto& pp = rg_.add_pass("postprocess", [](CmdEncoder&) { LINFO("executing postprocess"); });
  // pp.add_color_output("final_out", {.format = gfx::Format::R8G8B8A8Unorm});
  // pp.add_texture_input("forward_output");

  // auto& pp2 = rg_.add_pass("pp2", [](CmdEncoder&) {});
  // pp2.add_texture_input("postprocessout");
  // pp2.add_color_output("final_out", {.format = gfx::Format::R8G8B8A8Unorm});

  // auto& imgui = rg_.add_pass("imgui", [](CmdEncoder&) {});
  // imgui.add_color_output("final_out", {.format = gfx::Format::R8G8B8A8Unorm});

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

  uint32_t white = glm::packUnorm4x8(glm::vec4(1, 1, 1, 1));
  vk2::Buffer* staging = vk2::StagingBufferPool::get().acquire(32);
  memcpy((char*)staging->mapped_data(), (void*)&white, sizeof(u32));
  default_data_.white_img =
      vk2::create_texture_2d(VK_FORMAT_R8G8B8A8_SRGB, {1, 1, 1}, ImageUsage::ReadOnly);

  immediate_submit([this, staging](VkCommandBuffer cmd) {
    // TODO: extract
    state_.reset(cmd);
    state_.transition(default_data_.white_img->image(), VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                      VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    state_.flush_barriers();
    VkBufferImageCopy2 img_copy{.sType = VK_STRUCTURE_TYPE_BUFFER_IMAGE_COPY_2,
                                .bufferOffset = 0,
                                .bufferRowLength = 0,
                                .bufferImageHeight = 0,
                                .imageSubresource =
                                    {
                                        .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                                        .mipLevel = 0,
                                        .layerCount = 1,
                                    },
                                .imageExtent = VkExtent3D{1, 1, 1}};
    VkCopyBufferToImageInfo2 copy{.sType = VK_STRUCTURE_TYPE_COPY_BUFFER_TO_IMAGE_INFO_2_KHR,
                                  .srcBuffer = staging->buffer(),
                                  .dstImage = default_data_.white_img->image(),
                                  .dstImageLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                  .regionCount = 1,
                                  .pRegions = &img_copy};
    vkCmdCopyBufferToImage2KHR(cmd, &copy);
    state_.transition(default_data_.white_img->image(), VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                      VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                      VK_IMAGE_LAYOUT_READ_ONLY_OPTIMAL);
    state_.flush_barriers();
  });
  vk2::StagingBufferPool::get().free(staging);

  linear_sampler_ = SamplerCache::get().get_or_create_sampler({
      .min_filter = VK_FILTER_LINEAR,
      .mag_filter = VK_FILTER_LINEAR,
      .mipmap_mode = VK_SAMPLER_MIPMAP_MODE_LINEAR,
      .address_mode = VK_SAMPLER_ADDRESS_MODE_REPEAT,
  });
  linear_sampler_clamp_to_edge_ = SamplerCache::get().get_or_create_sampler({
      .min_filter = VK_FILTER_LINEAR,
      .mag_filter = VK_FILTER_LINEAR,
      .mipmap_mode = VK_SAMPLER_MIPMAP_MODE_LINEAR,
      .address_mode = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
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
  static_draw_info_buf_ = SlotBuffer<GPUDrawInfo>{BufferCreateInfo{
      .size = max_static_draws * sizeof(GPUDrawInfo),
      .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT |
               VK_BUFFER_USAGE_TRANSFER_DST_BIT,
  }};

  init_pipelines();

  init_indirect_drawing();
  csm_ = CSM(default_pipeline_layout_);
  ibl_ = IBL{};

  // TODO: make a function for this lmao, so cringe
  {
    auto vert_buf_size = sizeof(cube_vertices);
    // auto index_buf_size = sizeof(cube_indices);
    auto* staging = StagingBufferPool::get().acquire(vert_buf_size);
    memcpy(staging->mapped_data(), cube_vertices, vert_buf_size);
    // memcpy(staging->mapped_data(), cube_indices, index_buf_size);
    cube_vertices_gpu_offset_ = static_vertex_buf_->alloc(vert_buf_size);
    // cube_indices_gpu_offset_ = static_index_buf_->alloc(index_buf_size);
    transfer_submit([this, vert_buf_size, staging](
                        VkCommandBuffer cmd, VkFence fence,
                        std::queue<InFlightResource<vk2::Buffer*>>& transfers) {
      transfer_q_state_.reset(cmd)
          .transition_buffer_to_transfer_dst(static_vertex_buf_->buffer.buffer())
          .transition_buffer_to_transfer_dst(static_index_buf_->buffer.buffer())
          .flush_barriers();
      VkBufferCopy2KHR buf_copy = init::buffer_copy(0, cube_vertices_gpu_offset_, vert_buf_size);
      VkCopyBufferInfo2KHR buf_copy_info{.sType = VK_STRUCTURE_TYPE_COPY_BUFFER_INFO_2,
                                         .srcBuffer = staging->buffer(),
                                         .dstBuffer = static_vertex_buf_->buffer.buffer(),
                                         .regionCount = 1,
                                         .pRegions = &buf_copy};
      vkCmdCopyBuffer2KHR(cmd, &buf_copy_info);
      // vkCmdCopyBuffer2KHR(cmd, vk2::addr(VkCopyBufferInfo2KHR{
      //                              .sType = VK_STRUCTURE_TYPE_COPY_BUFFER_INFO_2,
      //                              .srcBuffer = staging->buffer(),
      //                              .dstBuffer = static_index_buf_->buffer.buffer(),
      //                              .regionCount = 1,
      //                              .pRegions = addr(init::buffer_copy(
      //                                  vert_buf_size, cube_indices_gpu_offset_,
      //                                  index_buf_size))}));

      // TODO: this would be solved by a render graph. double barrier happens if load scene at
      // same time before drawing transfer_q_state_
      //     .queue_transfer_buffer(state_, VK_PIPELINE_STAGE_2_VERTEX_INPUT_BIT,
      //                            VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT,
      //                            static_vertex_buf_->buffer.buffer(),
      //                            queues_.transfer_queue_idx, queues_.graphics_queue_idx)
      //     .queue_transfer_buffer(state_, VK_PIPELINE_STAGE_2_INDEX_INPUT_BIT,
      //                            VK_ACCESS_2_INDEX_READ_BIT,
      //                            static_index_buf_->buffer.buffer(),
      //                            queues_.transfer_queue_idx, queues_.graphics_queue_idx)
      //     .flush_barriers();
      transfers.emplace(staging, fence);
    });
  }
  default_env_map_path_ = info.resource_dir / "hdr" / "newport_loft.hdr";
  rg_.set_swapchain_info(
      RenderGraphSwapchainInfo{.width = swapchain_.dims.x, .height = swapchain_.dims.y});
  rg_.set_backbuffer_img("final_out");
  add_basic_forward_pass3(rg_);
  auto res = rg_.bake();
  if (!res) {
    LERROR("bake error {}", res.error());
    exit(1);
  }
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
    data.proj = proj;
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

  auto cmd_begin_info = vk2::init::command_buffer_begin_info();
  VK_CHECK(vkResetCommandBuffer(cmd, 0));
  VK_CHECK(vkBeginCommandBuffer(cmd, &cmd_begin_info));

  CmdEncoder ctx{cmd, default_pipeline_layout_};

  bind_bindless_descriptors(ctx);
  vk2::BindlessResourceAllocator::get().set_frame_num(curr_frame_num());
  vk2::BindlessResourceAllocator::get().flush_deletions();

  auto& swapchain_img = swapchain_.imgs[curr_swapchain_img_idx()];
  rg_.set_swapchain_info(RenderGraphSwapchainInfo{
      .curr_img = swapchain_img, .width = swapchain_.dims.x, .height = swapchain_.dims.y});
  auto& curr_frame_data = curr_frame_2();
  rg_.set_resource("draw_cnt_buf", curr_frame_data.draw_cnt_buf.handle);
  rg_.set_resource("final_draw_cmd_buf", curr_frame_data.final_draw_cmd_buf.handle);
  csm_->prepare_frame(rg_, curr_frame_num(), info.view, info.light_dir, aspect_ratio(),
                      info.fov_degrees, scene_aabb_, info.view_pos);
  rg_.setup_attachments();
  rg_.execute(ctx);
  // debug_mode_ = DEBUG_MODE_NORMALS;

  // state_.flush_transfers(queues_.graphics_queue_idx);

  // {
  //   TracyVkZone(curr_frame().tracy_vk_ctx, cmd, "draw objects");
  //
  //   // {
  //   //   TracyVkZone(curr_frame().tracy_vk_ctx, cmd, "img comp");
  //   //   state_.transition(*img_, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
  //   //   VK_ACCESS_2_MEMORY_WRITE_BIT,
  //   //                     VK_IMAGE_LAYOUT_GENERAL);
  //   //   state_.flush_barriers();
  //   //   struct {
  //   //     uint idx;
  //   //     float t;
  //   //   } pc{img_->view().storage_img_resource().handle, static_cast<f32>(glfwGetTime())};
  //   //   ctx.bind_compute_pipeline(PipelineManager::get().get(img_pipeline_)->pipeline);
  //   //   ctx.push_constants(default_pipeline_layout_, sizeof(pc), &pc);
  //   //   ctx.dispatch((img_->extent().width + 16) / 16, (img_->extent().height + 16) / 16, 1);
  //   // }
  //
  //   {
  //     TracyVkZone(curr_frame().tracy_vk_ctx, cmd, "Cull Objects");
  //     state_
  //         .buffer_barrier(draw_cnt_buf_.value(), VK_PIPELINE_STAGE_2_TRANSFER_BIT,
  //                         VK_ACCESS_2_TRANSFER_WRITE_BIT)
  //         .flush_barriers();
  //     vkCmdFillBuffer(cmd, draw_cnt_buf_->buffer(), 0, sizeof(u32), 0);
  //     // clear final buffer if we can't use drawindirectcount
  //     if (portable) {
  //       state_
  //           .buffer_barrier(final_draw_cmd_buf_.value(), VK_PIPELINE_STAGE_2_TRANSFER_BIT,
  //                           VK_ACCESS_2_TRANSFER_WRITE_BIT)
  //           .flush_barriers();
  //       vkCmdFillBuffer(cmd, final_draw_cmd_buf_->buffer(), 0, final_draw_cmd_buf_->size(), 0);
  //     }
  //
  //     // cull
  //     state_
  //         .buffer_barrier(static_draw_info_buf_->buffer, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
  //                         VK_ACCESS_2_SHADER_READ_BIT)
  //         .buffer_barrier(static_object_data_buf_->buffer,
  //         VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
  //                         VK_ACCESS_2_SHADER_READ_BIT)
  //         .buffer_barrier(final_draw_cmd_buf_.value(), VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
  //                         VK_ACCESS_2_SHADER_WRITE_BIT)
  //         .buffer_barrier(draw_cnt_buf_.value(), VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
  //                         VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT)
  //         .flush_barriers();
  //
  //     PipelineManager::get().bind_compute(cmd, cull_objs_pipeline_);
  //     struct {
  //       u32 num_objs;
  //       u32 in_draw_cmds_buf;
  //       u32 out_draw_cmds_buf;
  //       u32 draw_cnt_buf;
  //       u32 object_bounds_buf;
  //     } pc{static_cast<u32>(draw_cnt_), static_draw_info_buf_->buffer.resource_info_->handle,
  //          final_draw_cmd_buf_->resource_info_->handle, draw_cnt_buf_->resource_info_->handle,
  //          static_object_data_buf_->buffer.resource_info_->handle};
  //     ctx.push_constants(default_pipeline_layout_, sizeof(pc), &pc);
  //     vkCmdDispatch(cmd, (draw_cnt_ + 256) / 256, 1, 1);
  //   }
  //   state_
  //       .buffer_barrier(draw_cnt_buf_.value(), VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
  //                       VK_ACCESS_2_MEMORY_READ_BIT)
  //       .buffer_barrier(final_draw_cmd_buf_.value(), VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT,
  //                       VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT)
  //       .flush_barriers();
  //
  //   {
  //     TracyVkZone(curr_frame().tracy_vk_ctx, cmd, "CascadeShadowPass");
  //     csm_->render(
  //         state_, cmd, curr_frame_num(), info.view, info.light_dir, aspect_ratio(),
  //         info.fov_degrees,
  //         [&, this](const mat4& vp_matrix) {
  //           ShadowDepthPushConstants pc{
  //               vp_matrix,
  //               static_vertex_buf_->buffer.resource_info_->handle,
  //               static_instance_data_buf_->buffer.resource_info_->handle,
  //               static_object_data_buf_->buffer.resource_info_->handle,
  //               scene_buffer_handle,
  //               static_materials_buf_->buffer.resource_info_->handle,
  //               linear_sampler_->resource_info.handle,
  //           };
  //           ctx.push_constants(default_pipeline_layout_, sizeof(pc), &pc);
  //           vkCmdBindIndexBuffer(cmd, static_index_buf_->buffer.buffer(), 0,
  //           VK_INDEX_TYPE_UINT32); if (portable) {
  //             vkCmdDrawIndexedIndirect(cmd, final_draw_cmd_buf_->buffer(), 0, draw_cnt_,
  //                                      sizeof(VkDrawIndexedIndirectCommand));
  //           } else {
  //             vkCmdDrawIndexedIndirectCount(cmd, final_draw_cmd_buf_->buffer(), 0,
  //                                           draw_cnt_buf_->buffer(), 0, max_draws,
  //                                           sizeof(VkDrawIndexedIndirectCommand));
  //           }
  //         },
  //         scene_aabb_, info.view_pos);
  //   }
  //
  //   csm_->debug_shadow_pass(state_, cmd, linear_sampler_.value());
  //
  //   // draw
  //   {
  //     init::begin_debug_utils_label(cmd, "draw");
  //     state_
  //         .transition(
  //             *img_, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
  //             VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
  //             VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_ASPECT_COLOR_BIT)
  //         .buffer_barrier(csm_->get_shadow_data_buffer(curr_frame_num()).buffer(),
  //                         VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT)
  //         .transition(csm_->get_shadow_img(), VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
  //                     VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_MEMORY_READ_BIT,
  //                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_ASPECT_DEPTH_BIT)
  //         .transition(*depth_img_, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
  //                     VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
  //                     VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, VK_IMAGE_ASPECT_DEPTH_BIT)
  //         .flush_barriers();
  //     VkClearValue depth_clear{};
  //     // VkClearValue depth_clear{.depthStencil = {.depth = 1.f}};
  //     VkRenderingAttachmentInfo color_attachment =
  //         init::rendering_attachment_info(img_->view(),
  //         VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
  //     auto depth_att = init::rendering_attachment_info(
  //         depth_img_->view(), VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, &depth_clear);
  //     auto rendering_info = init::rendering_info(img_->extent_2d(), &color_attachment,
  //     &depth_att); vkCmdBeginRenderingKHR(cmd, &rendering_info); set_viewport_and_scissor(cmd,
  //     img_->extent_2d());
  //
  //     vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
  //                       PipelineManager::get().get(draw_pipeline_)->pipeline);
  //
  //     // TODO: jank
  //     auto draw_objects = [&](const Buffer& vertex_buffer, const Buffer& index_buffer,
  //                             const Buffer& instance_buffer, const Buffer& material_data_buffer,
  //                             const Buffer& object_data_buffer, const Buffer&, u64) {
  //       BasicPushConstants pc{
  //           scene_buffer_handle,
  //           vertex_buffer.resource_info_->handle,
  //           instance_buffer.resource_info_->handle,
  //           object_data_buffer.resource_info_->handle,
  //           material_data_buffer.resource_info_->handle,
  //           linear_sampler_->resource_info.handle,
  //           csm_->get_shadow_data_buffer(curr_frame_num()).resource_info_->handle,
  //           csm_->get_shadow_sampler().resource_info.handle,
  //           csm_->get_shadow_img().view().sampled_img_resource().handle,
  //           ibl_->irradiance_cubemap_tex_->view().sampled_img_resource().handle,
  //           ibl_->brdf_lut_->view().sampled_img_resource().handle,
  //           ibl_->prefiltered_env_map_tex_->texture->view().sampled_img_resource().handle,
  //           linear_sampler_clamp_to_edge_->resource_info.handle};
  //       ctx.push_constants(default_pipeline_layout_, sizeof(pc), &pc);
  //
  //       vkCmdBindIndexBuffer(cmd, index_buffer.buffer(), 0, VK_INDEX_TYPE_UINT32);
  //       if (portable) {
  //         vkCmdDrawIndexedIndirect(cmd, final_draw_cmd_buf_->buffer(), 0, draw_cnt_,
  //                                  sizeof(VkDrawIndexedIndirectCommand));
  //       } else {
  //         vkCmdDrawIndexedIndirectCount(cmd, final_draw_cmd_buf_->buffer(), 0,
  //                                       draw_cnt_buf_->buffer(), 0, max_draws,
  //                                       sizeof(VkDrawIndexedIndirectCommand));
  //       }
  //     };
  //
  //     TracyVkZone(curr_frame().tracy_vk_ctx, cmd, "Final Draw Pass");
  //     if (draw_cnt_) {
  //       draw_objects(static_vertex_buf_->buffer, static_index_buf_->buffer,
  //                    static_instance_data_buf_->buffer, static_materials_buf_->buffer,
  //                    static_object_data_buf_->buffer, static_draw_info_buf_->buffer, draw_cnt_);
  //     }
  //
  //     // skybox
  //     {
  //       TracyVkZone(curr_frame().tracy_vk_ctx, cmd, "Draw skybox");
  //       PipelineManager::get().bind_graphics(cmd, skybox_pipeline_);
  //       // TODO:  use an enum this is awful
  //       u32 skybox_handle{};
  //       if (render_prefilter_mip_skybox_) {
  //         assert(ibl_->prefiltered_env_tex_views_.size() > (size_t)prefilter_mip_skybox_level_);
  //         skybox_handle = ibl_->prefiltered_env_tex_views_[prefilter_mip_skybox_level_]
  //                             ->sampled_img_resource()
  //                             .handle;
  //       } else {
  //         skybox_handle = convoluted_skybox.get()
  //                             ?
  //                             ibl_->irradiance_cubemap_tex_->view().sampled_img_resource().handle
  //                             : ibl_->env_cubemap_tex_->view().sampled_img_resource().handle;
  //       }
  //       struct {
  //         u32 scene_buffer, tex_idx, sampler_idx;
  //       } pc{scene_buffer_handle, skybox_handle, linear_sampler_->resource_info.handle};
  //       ctx.push_constants(default_pipeline_layout_, sizeof(pc), &pc);
  //       vkCmdDraw(cmd, 36, 1, 0, 0);
  //     }
  //     vkCmdEndRenderingKHR(cmd);
  //     // Add after vkCmdEndRenderingKHR(cmd):
  //     state_
  //         .transition(*depth_img_, VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
  //                     VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT, VK_IMAGE_LAYOUT_GENERAL,
  //                     VK_IMAGE_ASPECT_DEPTH_BIT)
  //         .flush_barriers();
  //     init::end_debug_utils_label(cmd);
  //   }
  //
  //   vk2::Image* final_img = &img_.value();
  //   if (debug_mode_ == DEBUG_MODE_NONE && postprocess_pass.get()) {
  //     final_img = &post_processed_img_.value();
  //     u32 postprocess_flags = 0;
  //     if (gammacorrect_enabled.get()) {
  //       postprocess_flags |= 0x2;
  //     }
  //     if (tonemap_enabled.get()) {
  //       postprocess_flags |= 0x1;
  //     }
  //     state_
  //         .transition(*img_, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT,
  //                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
  //         .transition(*post_processed_img_, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
  //                     VK_ACCESS_2_SHADER_WRITE_BIT, VK_IMAGE_LAYOUT_GENERAL)
  //         .flush_barriers();
  //     PipelineManager::get().bind_compute(cmd, postprocess_pipeline_);
  //     struct {
  //       u32 in_tex_idx, out_tex_idx, flags;
  //     } pc{img_->view().storage_img_resource().handle,
  //          post_processed_img_->view().storage_img_resource().handle, postprocess_flags};
  //     ctx.push_constants(default_pipeline_layout_, sizeof(pc), &pc);
  //     vkCmdDispatch(cmd, (post_processed_img_->extent_2d().width + 16) / 16,
  //                   (post_processed_img_->extent_2d().height + 16) / 16, 1);
  //   }
  //
  //   state_.transition(*final_img, VK_PIPELINE_STAGE_2_BLIT_BIT, VK_ACCESS_2_MEMORY_READ_BIT,
  //                     VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_ASPECT_COLOR_BIT);
  //   auto& swapchain_img = swapchain_.imgs[curr_swapchain_img_idx()];
  //   state_.transition(swapchain_img, VK_PIPELINE_STAGE_2_BLIT_BIT,
  //   VK_ACCESS_2_TRANSFER_WRITE_BIT,
  //                     VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_ASPECT_COLOR_BIT);
  //
  //   state_.flush_barriers();
  //
  //   VkExtent3D dims{glm::min(final_img->extent().width, swapchain_.dims.x),
  //                   glm::min(final_img->extent().height, swapchain_.dims.y), 1};
  //   blit_img(cmd, final_img->image(), swapchain_img, dims, VK_IMAGE_ASPECT_COLOR_BIT);
  //
  //   if (draw_imgui) {
  //     state_.transition(
  //         swapchain_img, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
  //         VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT,
  //         VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
  //     state_.flush_barriers();
  //     render_imgui(cmd, {swapchain_.dims.x, swapchain_.dims.y},
  //                  swapchain_.img_views[curr_swapchain_img_idx()]);
  //   }
  //
  //   state_.transition(swapchain_img, VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT_KHR,
  //                     VK_ACCESS_2_MEMORY_READ_BIT, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);
  //   state_.flush_barriers();
  //   TracyVkCollect(curr_frame().tracy_vk_ctx, cmd);
  // }

  VK_CHECK(vkEndCommandBuffer(cmd));

  std::array<VkSemaphoreSubmitInfo, 10> wait_semaphores{};
  u32 next_wait_sem_idx{0};
  // TODO: back to VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT if imgui writes to it
  wait_semaphores[next_wait_sem_idx++] = vk2::init::semaphore_submit_info(
      curr_frame().swapchain_semaphore,
      VK_PIPELINE_STAGE_2_TRANSFER_BIT | VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT);
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

    ImGui::Checkbox("Render prefilter env map skybox", &render_prefilter_mip_skybox_);
    ImGui::SliderInt("Prefilter Env Map Layer", &prefilter_mip_skybox_level_, 0,
                     ibl_->prefiltered_env_map_tex_->texture->create_info().mip_levels - 1);

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

void VkRender2::on_resize() {}

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
                                   VK_ACCESS_2_INDEX_READ_BIT, static_index_buf_->buffer.buffer(),
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

    std::vector<GPUDrawInfo> cmds;
    cmds.reserve(resources->scene_graph_data.mesh_node_indices.size());

    vec3 scene_min = vec3{std::numeric_limits<float>::max()};
    vec3 scene_max = vec3{std::numeric_limits<float>::lowest()};
    for (auto& node : resources->scene_graph_data.node_datas) {
      for (auto& mesh_indices : node.meshes) {
        auto& mesh = resources->mesh_draw_infos[mesh_indices.mesh_idx];
        vec3 min = node.world_transform * vec4(mesh.bounds.origin - mesh.bounds.extents, 1.);
        vec3 max = node.world_transform * vec4(mesh.bounds.origin + mesh.bounds.extents, 1.);
        scene_min = glm::min(scene_min, min);
        scene_max = glm::max(scene_max, max);
        cmds.emplace_back(GPUDrawInfo{
            .index_cnt = mesh.index_count,
            .first_index = static_cast<u32>(resources->first_index + mesh.first_index),
            .vertex_offset = static_cast<u32>(resources->first_vertex + mesh.first_vertex),
            .pad = 0});
      }
    }

    scene_aabb_.min = glm::min(scene_aabb_.min, scene_min);
    scene_aabb_.max = glm::max(scene_aabb_.max, scene_max);

    u64 cmds_buf_size = cmds.size() * sizeof(GPUDrawInfo);

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
            init::buffer_copy(cmds_staging_offset + (cmd_i * sizeof(GPUDrawInfo)),
                              obj_draw.draw_cmd_slots[cmd_i].offset(), sizeof(GPUDrawInfo));
      }
      VkCopyBufferInfo2KHR buf_copy_info{.sType = VK_STRUCTURE_TYPE_COPY_BUFFER_INFO_2,
                                         .srcBuffer = staging.get_buffer()->buffer(),
                                         .dstBuffer = static_draw_info_buf_->buffer.buffer(),
                                         .regionCount = static_cast<u32>(copies.size()),
                                         .pRegions = copies.data()};
      vkCmdCopyBuffer2KHR(cmd, &buf_copy_info);
      {
        copies.clear();
        for (u64 obj_data_i = 0; obj_data_i < obj_datas.size(); obj_data_i++) {
          copies.emplace_back(init::buffer_copy(
              obj_datas_staging_offset + (obj_data_i * sizeof(gfx::ObjectData)),
              obj_draw.obj_data_slots[obj_data_i].offset(), sizeof(gfx::ObjectData)));
        }
        VkCopyBufferInfo2KHR buf_copy_info{.sType = VK_STRUCTURE_TYPE_COPY_BUFFER_INFO_2,
                                           .srcBuffer = staging.get_buffer()->buffer(),
                                           .dstBuffer = static_object_data_buf_->buffer.buffer(),
                                           .regionCount = static_cast<u32>(copies.size()),
                                           .pRegions = copies.data()};
        vkCmdCopyBuffer2KHR(cmd, &buf_copy_info);
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
    case DEBUG_MODE_SHADOW:
      return "Shadow";
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
  postprocess_pipeline_ =
      PipelineManager::get().load_compute_pipeline({"postprocess/postprocess.comp"});
  img_pipeline_ = PipelineManager::get().load_compute_pipeline({"debug/clear_img.comp"});
  draw_pipeline_ = PipelineManager::get().load_graphics_pipeline(GraphicsPipelineCreateInfo{
      .vertex_path = "debug/basic.vert",
      .fragment_path = "debug/basic.frag",
      .layout = default_pipeline_layout_,
      .rendering = {.color_formats = {to_vkformat(draw_img_format_)},
                    .depth_format = to_vkformat(depth_img_format_)},
      .rasterization = {.cull_mode = CullMode::None},
      .depth_stencil = GraphicsPipelineCreateInfo::depth_enable(true, CompareOp::GreaterOrEqual),
  });
  basic_draw3_pipeline_ = PipelineManager::get().load_graphics_pipeline(GraphicsPipelineCreateInfo{
      .vertex_path = "debug/basic3.vert",
      .fragment_path = "debug/basic3.frag",
      .layout = default_pipeline_layout_,
      .rendering = {.color_formats = {to_vkformat(draw_img_format_)},
                    .depth_format = to_vkformat(depth_img_format_)},
      .rasterization = {.cull_mode = CullMode::None},
      .depth_stencil = GraphicsPipelineCreateInfo::depth_enable(true, CompareOp::GreaterOrEqual),
  });

  basic_draw_pipeline_ = PipelineManager::get().load_graphics_pipeline(GraphicsPipelineCreateInfo{
      .vertex_path = "debug/basic2.vert",
      .fragment_path = "debug/basic2.frag",
      .layout = default_pipeline_layout_,
      .rendering = {.color_formats = {to_vkformat(draw_img_format_)},
                    .depth_format = to_vkformat(depth_img_format_)},
      .rasterization = {.cull_mode = CullMode::None},
      .depth_stencil = GraphicsPipelineCreateInfo::depth_enable(true, CompareOp::GreaterOrEqual),
  });
  skybox_pipeline_ = PipelineManager::get().load_graphics_pipeline(GraphicsPipelineCreateInfo{
      .vertex_path = "skybox/skybox.vert",
      .fragment_path = "skybox/skybox.frag",
      .layout = default_pipeline_layout_,
      .rendering = {.color_formats = {to_vkformat(draw_img_format_)},
                    .depth_format = to_vkformat(depth_img_format_)},
      .depth_stencil = GraphicsPipelineCreateInfo::depth_enable(true, CompareOp::GreaterOrEqual),
  });
  assert(draw_pipeline_);
}

void VkRender2::init_indirect_drawing() {
  for (auto& f : per_frame_data_2_) {
    f.draw_cnt_buf = vk2::get_device().create_buffer_holder(BufferCreateInfo{
        .size = sizeof(u32),
        .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT |
                 VK_BUFFER_USAGE_TRANSFER_DST_BIT,
    });
    f.final_draw_cmd_buf = get_device().create_buffer_holder(BufferCreateInfo{
        .size = max_draws * sizeof(VkDrawIndexedIndirectCommand),
        .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT |
                 VK_BUFFER_USAGE_TRANSFER_DST_BIT,
    });
  }
  cull_objs_pipeline_ = PipelineManager::get().load_compute_pipeline({"cull_objects.comp"});
}

std::optional<vk2::Image> VkRender2::load_hdr_img(CmdEncoder& ctx,
                                                  const std::filesystem::path& path, bool flip) {
  VkCommandBuffer cmd = ctx.cmd();
  auto cpu_img_data = gfx::loader::load_hdr(path, 4, flip);
  if (!cpu_img_data.has_value()) return std::nullopt;
  auto tex = vk2::Image{ImageCreateInfo{.view_type = VK_IMAGE_VIEW_TYPE_2D,
                                        .format = VK_FORMAT_R32G32B32A32_SFLOAT,
                                        .extent = {cpu_img_data->w, cpu_img_data->h, 1},
                                        .mip_levels = 1,
                                        .array_layers = 1,
                                        .usage = ImageUsage::ReadOnly}};
  if (!tex.image()) {
    return std::nullopt;
  }
  auto cnt = cpu_img_data->w * cpu_img_data->h;
  u64 cpu_img_size = sizeof(float) * cnt * 4;
  auto* staging_buf = StagingBufferPool::get().acquire(cpu_img_size);
  auto* mapped = (float*)staging_buf->mapped_data();
  const float* src = cpu_img_data->data;
  memcpy(mapped, src, cpu_img_size);

  transition_image(cmd, tex, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
  VkBufferImageCopy2 img_copy_info{.sType = VK_STRUCTURE_TYPE_BUFFER_IMAGE_COPY_2,
                                   .bufferOffset = 0,
                                   .imageSubresource =
                                       {
                                           .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                                           .mipLevel = 0,
                                           .baseArrayLayer = 0,
                                           .layerCount = 1,
                                       },
                                   .imageExtent = tex.extent()};
  VkCopyBufferToImageInfo2 copy_to_img_info{
      .sType = VK_STRUCTURE_TYPE_COPY_BUFFER_TO_IMAGE_INFO_2_KHR,
      .srcBuffer = staging_buf->buffer(),
      .dstImage = tex.image(),
      .dstImageLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
      .regionCount = 1,
      .pRegions = &img_copy_info};
  vkCmdCopyBufferToImage2KHR(cmd, &copy_to_img_info);
  transition_image(cmd, tex, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

  return tex;
}

void VkRender2::generate_mipmaps(StateTracker& state, VkCommandBuffer cmd, vk2::Image& tex) {
  // TODO: this is unbelievably hacky: using manual barriers outside of state tracker and then
  // updating state tracker with the result. solve this by updating state tracker to manage
  // subresource ranges or toss it and make a render graph
  u32 array_layers = tex.create_info().array_layers;
  u32 mip_levels = get_mip_levels(tex.extent_2d());
  VkExtent2D curr_img_size = tex.extent_2d();
  for (u32 mip_level = 0; mip_level < mip_levels; mip_level++) {
    VkImageMemoryBarrier2 img_barrier{.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
    img_barrier.image = tex.image();
    img_barrier.srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT;
    img_barrier.srcStageMask = VK_PIPELINE_STAGE_2_BLIT_BIT;
    img_barrier.dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT;
    img_barrier.dstStageMask = VK_PIPELINE_STAGE_2_BLIT_BIT;
    img_barrier.oldLayout = tex.curr_layout;
    tex.curr_layout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    img_barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    img_barrier.subresourceRange = VkImageSubresourceRange{.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                                                           .baseMipLevel = mip_level,
                                                           .levelCount = 1,
                                                           .baseArrayLayer = 0,
                                                           .layerCount = array_layers};
    auto dep_info = init::dependency_info({}, SPAN1(img_barrier));
    vkCmdPipelineBarrier2KHR(cmd, &dep_info);
    if (mip_level < mip_levels - 1) {
      VkExtent2D half_img_size = {curr_img_size.width / 2, curr_img_size.height / 2};
      VkImageBlit2KHR blit{
          .sType = VK_STRUCTURE_TYPE_IMAGE_BLIT_2_KHR,
          .srcSubresource =
              VkImageSubresourceLayers{
                  .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                  .mipLevel = mip_level,
                  .baseArrayLayer = 0,
                  .layerCount = array_layers,
              },
          .srcOffsets = {VkOffset3D{}, VkOffset3D{static_cast<i32>(curr_img_size.width),
                                                  static_cast<i32>(curr_img_size.height), 1u}},
          .dstSubresource = VkImageSubresourceLayers{.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                                                     .mipLevel = mip_level + 1,
                                                     .baseArrayLayer = 0,
                                                     .layerCount = array_layers},
          .dstOffsets = {VkOffset3D{}, VkOffset3D{static_cast<i32>(half_img_size.width),
                                                  static_cast<i32>(half_img_size.height), 1u}}};
      VkBlitImageInfo2KHR info{.sType = VK_STRUCTURE_TYPE_BLIT_IMAGE_INFO_2_KHR,
                               .srcImage = tex.image(),
                               .srcImageLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                               .dstImage = tex.image(),
                               .dstImageLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                               .regionCount = 1,
                               .pRegions = &blit,
                               .filter = VK_FILTER_LINEAR};
      vkCmdBlitImage2KHR(cmd, &info);
      curr_img_size = half_img_size;
    }
  }
  {
    auto* img_state = state.get_img_state(tex.image());
    transition_image(cmd, tex, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL);
    if (img_state) {
      img_state->curr_layout = VK_IMAGE_LAYOUT_GENERAL;
      img_state->curr_access = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;
      img_state->curr_stage = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    }
  }
}

void VkRender2::set_env_map(const std::filesystem::path& path) {
  immediate_submit([this, &path](CmdEncoder& ctx) { ibl_->load_env_map(ctx, path); });
}

void VkRender2::bind_bindless_descriptors(CmdEncoder& ctx) {
  ctx.bind_descriptor_set(VK_PIPELINE_BIND_POINT_GRAPHICS, default_pipeline_layout_, &main_set_, 0);
  ctx.bind_descriptor_set(VK_PIPELINE_BIND_POINT_COMPUTE, default_pipeline_layout_, &main_set_, 0);
  ctx.bind_descriptor_set(VK_PIPELINE_BIND_POINT_GRAPHICS, default_pipeline_layout_, &main_set2_,
                          1);
  ctx.bind_descriptor_set(VK_PIPELINE_BIND_POINT_COMPUTE, default_pipeline_layout_, &main_set2_, 1);
}

void VkRender2::immediate_submit(std::function<void(CmdEncoder& ctx)>&& function) {
  VkFence imm_fence = FencePool::get().allocate(true);
  VK_CHECK(vkResetCommandBuffer(imm_cmd_buf_, 0));
  auto info = vk2::init::command_buffer_begin_info();
  VK_CHECK(vkBeginCommandBuffer(imm_cmd_buf_, &info));
  CmdEncoder ctx{imm_cmd_buf_, default_pipeline_layout_};
  function(ctx);
  VK_CHECK(vkEndCommandBuffer(imm_cmd_buf_));
  VkCommandBufferSubmitInfo cmd_info = init::command_buffer_submit_info(imm_cmd_buf_);
  VkSubmitInfo2 submit = init::queue_submit_info(SPAN1(cmd_info), {}, {});
  VK_CHECK(vkQueueSubmit2KHR(queues_.graphics_queue, 1, &submit, imm_fence));
  VK_CHECK(vkWaitForFences(device_, 1, &imm_fence, true, 99999999999));
  FencePool::get().free(imm_fence);
}

void VkRender2::generate_mipmaps(CmdEncoder& ctx, vk2::Image& tex) {
  VkCommandBuffer cmd = ctx.cmd();
  u32 array_layers = tex.create_info().array_layers;
  u32 mip_levels = get_mip_levels(tex.extent_2d());
  VkExtent2D curr_img_size = tex.extent_2d();
  for (u32 mip_level = 0; mip_level < mip_levels; mip_level++) {
    VkImageMemoryBarrier2 img_barrier{.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
    img_barrier.image = tex.image();
    img_barrier.srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT;
    img_barrier.srcStageMask = VK_PIPELINE_STAGE_2_BLIT_BIT;
    img_barrier.dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT;
    img_barrier.dstStageMask = VK_PIPELINE_STAGE_2_BLIT_BIT;
    img_barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    img_barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    tex.curr_layout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    img_barrier.subresourceRange = VkImageSubresourceRange{.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                                                           .baseMipLevel = mip_level,
                                                           .levelCount = 1,
                                                           .baseArrayLayer = 0,
                                                           .layerCount = array_layers};
    auto dep_info = init::dependency_info({}, SPAN1(img_barrier));
    vkCmdPipelineBarrier2KHR(cmd, &dep_info);
    if (mip_level < mip_levels - 1) {
      VkExtent2D half_img_size = {curr_img_size.width / 2, curr_img_size.height / 2};
      VkImageBlit2KHR blit{
          .sType = VK_STRUCTURE_TYPE_IMAGE_BLIT_2_KHR,
          .srcSubresource =
              VkImageSubresourceLayers{
                  .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                  .mipLevel = mip_level,
                  .baseArrayLayer = 0,
                  .layerCount = array_layers,
              },
          .srcOffsets = {VkOffset3D{}, VkOffset3D{static_cast<i32>(curr_img_size.width),
                                                  static_cast<i32>(curr_img_size.height), 1u}},
          .dstSubresource = VkImageSubresourceLayers{.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                                                     .mipLevel = mip_level + 1,
                                                     .baseArrayLayer = 0,
                                                     .layerCount = array_layers},
          .dstOffsets = {VkOffset3D{}, VkOffset3D{static_cast<i32>(half_img_size.width),
                                                  static_cast<i32>(half_img_size.height), 1u}}};
      VkBlitImageInfo2KHR info{.sType = VK_STRUCTURE_TYPE_BLIT_IMAGE_INFO_2_KHR,
                               .srcImage = tex.image(),
                               .srcImageLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                               .dstImage = tex.image(),
                               .dstImageLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                               .regionCount = 1,
                               .pRegions = &blit,
                               .filter = VK_FILTER_LINEAR};
      vkCmdBlitImage2KHR(cmd, &info);
      curr_img_size = half_img_size;
    }
  }
  transition_image(cmd, tex, VK_IMAGE_LAYOUT_GENERAL);
}
void VkRender2::draw_cube(VkCommandBuffer cmd) const {
  vkCmdDraw(cmd, 36, 1, cube_vertices_gpu_offset_ / sizeof(gfx::Vertex), 0);
}

void VkRender2::add_basic_forward_pass(RenderGraph& rg) {
  ZoneScoped;
  auto& forward = rg.add_pass("forward");
  auto final_out_handle =
      forward.add("final_out", {.format = draw_img_format_}, Access::ColorWrite);
  forward.add("depth", {.format = depth_img_format_}, Access::DepthStencilWrite);
  forward.set_execute_fn([&rg, final_out_handle, this](CmdEncoder& cmd) {
    auto* resource = rg.get_resource(final_out_handle);
    auto* tex = rg.get_texture(final_out_handle);
    assert(resource && tex);
    if (!resource || !tex) return;
    VkClearValue clear_value{.color = {{0.2, 0.2, 0.2, 1.0}}};
    VkRenderingAttachmentInfo color_attachment = vk2::init::rendering_attachment_info(
        tex->view().view(), VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, &clear_value);
    VkExtent2D render_extent{tex->create_info().extent.width, tex->create_info().extent.height};
    VkRenderingInfo render_info =
        vk2::init::rendering_info(render_extent, &color_attachment, nullptr, nullptr);
    vkCmdBeginRenderingKHR(cmd.cmd(), &render_info);
    set_viewport_and_scissor(cmd.cmd(), render_extent);
    // cube
    {
      PipelineManager::get().bind_graphics(cmd.cmd(), basic_draw_pipeline_);
      float t = glfwGetTime();
      mat4 view = glm::lookAt(vec3{glm::sin(t * .1), 2, glm::cos(t * .1)}, {0, 0, 0}, {0, 1, 0});
      mat4 proj = glm::perspective(glm::radians(90.f), aspect_ratio(), 1000.f, .1f);
      struct {
        mat4 vp;
        u32 vertex_buffer;
      } pc{.vp = proj * view, .vertex_buffer = static_vertex_buf_->buffer.resource_info_->handle};
      cmd.push_constants(sizeof(pc), &pc);
      vkCmdDraw(cmd.cmd(), 36, 1, cube_vertices_gpu_offset_ / sizeof(gfx::Vertex), 0);
    }
    vkCmdEndRenderingKHR(cmd.cmd());
  });
}

namespace {
bool portable = false;
}
void VkRender2::add_basic_forward_pass3(RenderGraph& rg) {
  ZoneScoped;
  u32 scene_buffer_handle = curr_frame_2().scene_uniform_buf->resource_info_->handle;
  auto& clear_buff = rg.add_pass("clear_draw_cnt_buf");
  clear_buff.add_proxy("draw_cnt_buf", Access::TransferWrite);
  clear_buff.add_proxy("final_draw_cmd_buf", Access::TransferWrite);
  clear_buff.set_execute_fn([this](CmdEncoder& cmd) {
    auto& curr_frame_data = curr_frame_2();
    auto& draw_cnt_buf = curr_frame_data.draw_cnt_buf;
    auto& final_draw_cmd_buf = curr_frame_data.final_draw_cmd_buf;
    vkCmdFillBuffer(cmd.cmd(), get_device().get_buffer(draw_cnt_buf)->buffer(), 0, sizeof(u32), 0);
    if (portable) {
      vkCmdFillBuffer(cmd.cmd(), get_device().get_buffer(final_draw_cmd_buf)->buffer(), 0,
                      get_device().get_buffer(final_draw_cmd_buf)->size(), 0);
    }
  });

  auto& cull = rg.add_pass("cull");
  cull.add_proxy("draw_cnt_buf", Access::ComputeRW);
  cull.add_proxy("final_draw_cmd_buf", Access::ComputeWrite);
  cull.set_execute_fn([this](CmdEncoder& cmd) {
    PipelineManager::get().bind_compute(cmd.cmd(), cull_objs_pipeline_);
    auto& curr_frame_data = curr_frame_2();
    auto& draw_cnt_buf = curr_frame_data.draw_cnt_buf;
    auto& final_draw_cmd_buf = curr_frame_data.final_draw_cmd_buf;
    struct {
      u32 num_objs;
      u32 in_draw_cmds_buf;
      u32 out_draw_cmds_buf;
      u32 draw_cnt_buf;
      u32 object_bounds_buf;
    } pc{static_cast<u32>(draw_cnt_), static_draw_info_buf_->buffer.resource_info_->handle,
         get_device().get_buffer(final_draw_cmd_buf)->resource_info_->handle,
         get_device().get_buffer(draw_cnt_buf)->resource_info_->handle,
         static_object_data_buf_->buffer.resource_info_->handle};
    cmd.push_constants(default_pipeline_layout_, sizeof(pc), &pc);
    cmd.dispatch((draw_cnt_ + 256) / 256, 1, 1);
  });

  AttachmentInfo shadow_map_att_info{.size_class = SizeClass::Absolute,
                                     .dims = {4096, 4096, 1},
                                     .format = Format::D32Sfloat,
                                     .layers = csm_->get_cascade_count()};
  {
    auto& csm_prepare_pass = rg.add_pass("csm_prepare");
    csm_prepare_pass.add_proxy("shadow_data_buf", Access::TransferWrite);
    csm_prepare_pass.set_execute_fn([this](CmdEncoder& cmd) {
      auto* buf = get_device().get_buffer(csm_->get_shadow_data_buffer(curr_frame_num()));
      if (!buf) return;
      vkCmdUpdateBuffer(cmd.cmd(), buf->buffer(), 0, sizeof(CSM::ShadowData), &csm_->data);
    });

    auto& csm = rg.add_pass("csm");
    auto rg_shadow_map_img =
        csm.add("shadow_map_img", shadow_map_att_info, Access::DepthStencilWrite);
    csm.add_proxy("draw_cnt_buf", Access::IndirectRead);
    csm.add_proxy("final_draw_cmd_buf", Access::IndirectRead);
    csm.set_execute_fn([this, scene_buffer_handle, rg_shadow_map_img](CmdEncoder& cmd) {
      csm_->shadow_map_img = rg_.get_texture_handle(rg_shadow_map_img);
      csm_->render2(cmd.cmd(), [this, scene_buffer_handle, &cmd](const mat4& vp_matrix) {
        ShadowDepthPushConstants pc{
            vp_matrix,
            static_vertex_buf_->buffer.resource_info_->handle,
            static_instance_data_buf_->buffer.resource_info_->handle,
            static_object_data_buf_->buffer.resource_info_->handle,
            scene_buffer_handle,
            static_materials_buf_->buffer.resource_info_->handle,
            linear_sampler_->resource_info.handle,
        };
        cmd.push_constants(default_pipeline_layout_, sizeof(pc), &pc);
        vkCmdBindIndexBuffer(cmd.cmd(), static_index_buf_->buffer.buffer(), 0,
                             VK_INDEX_TYPE_UINT32);
        auto& curr_frame_data = curr_frame_2();
        auto* draw_cnt_buf = get_device().get_buffer(curr_frame_data.draw_cnt_buf);
        auto* final_draw_cmd_buf = get_device().get_buffer(curr_frame_data.final_draw_cmd_buf);
        if (portable) {
          vkCmdDrawIndexedIndirect(cmd.cmd(), final_draw_cmd_buf->buffer(), 0, draw_cnt_,
                                   sizeof(VkDrawIndexedIndirectCommand));
        } else {
          vkCmdDrawIndexedIndirectCount(cmd.cmd(), final_draw_cmd_buf->buffer(), 0,
                                        draw_cnt_buf->buffer(), 0, max_draws,
                                        sizeof(VkDrawIndexedIndirectCommand));
        }
      });
    });
  }

  {
    auto& forward = rg.add_pass("forward");
    auto final_out_handle =
        forward.add("draw_out", {.format = draw_img_format_}, Access::ColorWrite);
    forward.add_proxy("draw_cnt_buf", Access::IndirectRead);
    forward.add_proxy("final_draw_cmd_buf", Access::IndirectRead);
    forward.add_proxy("shadow_data_buf", Access::FragmentRead);
    forward.add("shadow_map_img", shadow_map_att_info, Access::FragmentRead);
    auto depth_out_handle =
        forward.add("depth", {.format = depth_img_format_}, Access::DepthStencilWrite);
    forward.set_execute_fn([&rg, final_out_handle, this, depth_out_handle,
                            scene_buffer_handle](CmdEncoder& cmd) {
      auto& curr_frame_data = curr_frame_2();
      auto& draw_cnt_buf = curr_frame_data.draw_cnt_buf;
      auto& final_draw_cmd_buf = curr_frame_data.final_draw_cmd_buf;
      auto* tex = rg.get_texture(final_out_handle);
      assert(tex);
      if (!tex) return;
      VkClearValue clear_value{.color = {{0.2, 0.2, 0.2, 1.0}}};
      VkRenderingAttachmentInfo color_attachment = vk2::init::rendering_attachment_info(
          tex->view().view(), VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, &clear_value);
      VkClearValue depth_clear{};
      auto depth_att = init::rendering_attachment_info(
          rg.get_texture(depth_out_handle)->view().view(),
          VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL_KHR, &depth_clear);
      VkExtent2D render_extent{tex->create_info().extent.width, tex->create_info().extent.height};
      VkRenderingInfo render_info =
          vk2::init::rendering_info(render_extent, &color_attachment, &depth_att, nullptr);
      vkCmdBeginRenderingKHR(cmd.cmd(), &render_info);
      set_viewport_and_scissor(cmd.cmd(), render_extent);
      bool draw_fancy{true};
      if (draw_fancy) {
        BasicPushConstants pc{
            scene_buffer_handle,
            static_vertex_buf_->buffer.resource_info_->handle,
            static_instance_data_buf_->buffer.resource_info_->handle,
            static_object_data_buf_->buffer.resource_info_->handle,
            static_materials_buf_->buffer.resource_info_->handle,
            linear_sampler_->resource_info.handle,
            get_device()
                .get_buffer(csm_->get_shadow_data_buffer(curr_frame_num()))
                ->resource_info_->handle,
            csm_->get_shadow_sampler().resource_info.handle,
            get_device().get_image(csm_->shadow_map_img)->view().sampled_img_resource().handle,
            ibl_->irradiance_cubemap_tex_->view().sampled_img_resource().handle,
            ibl_->brdf_lut_->view().sampled_img_resource().handle,
            ibl_->prefiltered_env_map_tex_->texture->view().sampled_img_resource().handle,
            linear_sampler_clamp_to_edge_->resource_info.handle};
        cmd.push_constants(default_pipeline_layout_, sizeof(pc), &pc);
        PipelineManager::get().bind_graphics(cmd.cmd(), draw_pipeline_);
      } else {
        struct {
          u32 scene_buffer;
          u32 vertex_buffer_idx;
          u32 instance_buffer;
          u32 object_data_buffer;
        } pc{scene_buffer_handle, static_vertex_buf_->buffer.resource_info_->handle,
             static_instance_data_buf_->buffer.resource_info_->handle,
             static_object_data_buf_->buffer.resource_info_->handle};
        cmd.push_constants(sizeof(pc), &pc);
        PipelineManager::get().bind_graphics(cmd.cmd(), basic_draw3_pipeline_);
      }
      vkCmdBindIndexBuffer(cmd.cmd(), static_index_buf_->buffer.buffer(), 0, VK_INDEX_TYPE_UINT32);
      if (portable) {
        vkCmdDrawIndexedIndirect(cmd.cmd(), get_device().get_buffer(final_draw_cmd_buf)->buffer(),
                                 0, draw_cnt_, sizeof(VkDrawIndexedIndirectCommand));
      } else {
        vkCmdDrawIndexedIndirectCount(cmd.cmd(),
                                      get_device().get_buffer(final_draw_cmd_buf)->buffer(), 0,
                                      get_device().get_buffer(draw_cnt_buf)->buffer(), 0, max_draws,
                                      sizeof(VkDrawIndexedIndirectCommand));
      }

      {
        PipelineManager::get().bind_graphics(cmd.cmd(), skybox_pipeline_);
        u32 skybox_handle{};
        if (render_prefilter_mip_skybox_) {
          assert(ibl_->prefiltered_env_tex_views_.size() > (size_t)prefilter_mip_skybox_level_);
          skybox_handle = ibl_->prefiltered_env_tex_views_[prefilter_mip_skybox_level_]
                              ->sampled_img_resource()
                              .handle;
        } else {
          skybox_handle = convoluted_skybox.get()
                              ? ibl_->irradiance_cubemap_tex_->view().sampled_img_resource().handle
                              : ibl_->env_cubemap_tex_->view().sampled_img_resource().handle;
        }
        struct {
          u32 scene_buffer, tex_idx, sampler_idx;
        } pc{scene_buffer_handle, skybox_handle, linear_sampler_->resource_info.handle};
        cmd.push_constants(default_pipeline_layout_, sizeof(pc), &pc);
        vkCmdDraw(cmd.cmd(), 36, 1, 0, 0);
      }
      vkCmdEndRenderingKHR(cmd.cmd());
    });
  }

  {
    auto& pp = rg.add_pass("post_process");
    auto draw_out_handle = pp.add("draw_out", {.format = draw_img_format_}, Access::ComputeRead);
    auto final_out_handle =
        pp.add("final_out", {.format = Format::R8G8B8A8Unorm}, Access::ComputeWrite);
    pp.set_execute_fn([this, &rg, draw_out_handle, final_out_handle](CmdEncoder& cmd) {
      if (postprocess_pass_enabled.get()) {
        u32 postprocess_flags = 0;
        if (debug_mode_ != DEBUG_MODE_NONE) {
          postprocess_flags |= 0x4;
        }
        if (gammacorrect_enabled.get()) {
          postprocess_flags |= 0x2;
        }
        if (tonemap_enabled.get()) {
          postprocess_flags |= 0x1;
        }
        PipelineManager::get().bind_compute(cmd.cmd(), postprocess_pipeline_);
        auto* post_processed_img = rg.get_texture(final_out_handle);
        struct {
          u32 in_tex_idx, out_tex_idx, flags;
        } pc{rg.get_texture(draw_out_handle)->view().storage_img_resource().handle,
             post_processed_img->view().storage_img_resource().handle, postprocess_flags};
        cmd.push_constants(default_pipeline_layout_, sizeof(pc), &pc);
        vkCmdDispatch(cmd.cmd(), (post_processed_img->extent_2d().width + 16) / 16,
                      (post_processed_img->extent_2d().height + 16) / 16, 1);
      }
    });
  }
}
}  // namespace gfx
