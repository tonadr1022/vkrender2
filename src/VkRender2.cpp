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
#include <utility>

#include "BaseRenderer.hpp"
#include "CommandEncoder.hpp"
#include "FixedVector.hpp"
#include "Logger.hpp"
#include "RenderGraph.hpp"
#include "SceneLoader.hpp"
#include "StateTracker.hpp"
#include "Timer.hpp"
#include "glm/packing.hpp"
#include "imgui.h"
#include "shaders/common.h.glsl"
#include "shaders/cull_objects_common.h.glsl"
#include "shaders/debug/basic_common.h.glsl"
#include "shaders/gbuffer/gbuffer_common.h.glsl"
#include "shaders/gbuffer/shade_common.h.glsl"
#include "shaders/lines/draw_line_common.h.glsl"
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
AutoCVarInt csm_enabled{"renderer.csm_enabled", "CSM Enabled", 1, CVarFlags::EditCheckbox};
AutoCVarInt ibl_enabled{"renderer.ibl_enabled", "IBL Enabled", 1, CVarFlags::EditCheckbox};
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
  swapchain_att_info_ = {.format = vkformat_to_format(swapchain_.format)};

  vk2::StagingBufferPool::init();
  vk2::BindlessResourceAllocator::init(device_, vk2::get_device().allocator());
  main_set_ = vk2::BindlessResourceAllocator::get().main_set();
  main_set2_ = vk2::BindlessResourceAllocator::get().main_set2_;

  PrintTimerMS timer;

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

  nearest_sampler_ = SamplerCache::get().get_or_create_sampler({
      .min_filter = VK_FILTER_NEAREST,
      .mag_filter = VK_FILTER_NEAREST,
      .mipmap_mode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
      .address_mode = VK_SAMPLER_ADDRESS_MODE_REPEAT,
  });

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
      d.scene_uniform_buf =
          vk2::Buffer{BufferCreateInfo{.size = sizeof(SceneUniforms),
                                       .usage = BufferUsage_Storage,
                                       .flags = vk2::BufferCreateFlags_HostVisible}};

      d.line_draw_buf = get_device().create_buffer_holder(
          BufferCreateInfo{.size = sizeof(LineVertex) * 1000,
                           .usage = BufferUsage_Storage,
                           .flags = BufferCreateFlags_HostVisible});
    }
  }

  auto vertices_size = 10'000'000 * sizeof(gfx::Vertex);
  static_vertex_buf_.buffer = get_device().create_buffer_holder(
      {BufferCreateInfo{.size = vertices_size, .usage = BufferUsage_Storage}});
  static_vertex_buf_.allocator.init(vertices_size, sizeof(Vertex));

  auto indices_size = 10'000'000 * sizeof(u32);
  static_index_buf_.buffer = get_device().create_buffer_holder(BufferCreateInfo{
      .size = indices_size,
      .usage = BufferUsage_Index,
  });
  static_index_buf_.allocator.init(indices_size, sizeof(u32));
  u64 max_static_draws = 100'000;

  auto static_materials_size = 1000 * sizeof(gfx::Material);
  static_materials_buf_.buffer = get_device().create_buffer_holder(
      {BufferCreateInfo{.size = static_materials_size, .usage = BufferUsage_Storage}});
  static_materials_buf_.allocator.init(static_materials_size, sizeof(Material), 100);

  static_instance_data_buf_.buffer = get_device().create_buffer_holder(BufferCreateInfo{
      .size = max_static_draws * sizeof(InstanceData),
      .usage = BufferUsage_Storage,
  });
  static_instance_data_buf_.allocator.init(max_static_draws * sizeof(InstanceData),
                                           sizeof(InstanceData), 100);

  static_object_data_buf_.buffer = get_device().create_buffer_holder(BufferCreateInfo{
      .size = max_static_draws * sizeof(ObjectData),
      .usage = BufferUsage_Storage,
  });
  static_object_data_buf_.allocator.init(max_static_draws * sizeof(ObjectData), sizeof(ObjectData),
                                         100);

  init_pipelines();

  csm_ = std::make_unique<CSM>(
      this,
      [this](CmdEncoder& cmd, const mat4& vp, bool opaque_alpha, u32 cascade_i) {
        const StaticMeshDrawManager* mgr{};
        if (opaque_alpha) {
          mgr = &static_opaque_alpha_mask_draw_mgr_.value();
        } else {
          mgr = &static_opaque_draw_mgr_.value();
        }
        if (!mgr->should_draw()) return;
        ShadowDepthPushConstants pc{
            vp,
            static_vertex_buf_.get_buffer()->device_addr(),
            static_instance_data_buf_.get_buffer()->device_addr(),
            static_object_data_buf_.get_buffer()->device_addr(),
            curr_frame_2().scene_uniform_buf->resource_info_->handle,
            static_materials_buf_.get_buffer()->resource_info_->handle,
            linear_sampler_->resource_info.handle,
        };
        cmd.push_constants(sizeof(pc), &pc);
        vkCmdBindIndexBuffer(cmd.cmd(), static_index_buf_.get_buffer()->buffer(), 0,
                             VK_INDEX_TYPE_UINT32);
        execute_draw(cmd, *mgr->get_draw_passes()[1 + cascade_i].get_frame_out_draw_cmd_buf(),
                     mgr->get_num_draw_cmds());
      },
      [this](RenderGraphPass& pass) {
        if (static_opaque_draw_mgr_->should_draw()) {
          for (u32 i = 0; i < csm_->get_num_cascade_levels(); i++) {
            pass.add_proxy(static_opaque_draw_mgr_->get_draw_passes()[i + 1].name,
                           Access::IndirectRead);
          }
        }

        if (static_opaque_alpha_mask_draw_mgr_->should_draw()) {
          for (u32 i = 0; i < csm_->get_num_cascade_levels(); i++) {
            pass.add_proxy(static_opaque_alpha_mask_draw_mgr_->get_draw_passes()[i + 1].name,
                           Access::IndirectRead);
          }
        }
      });

  static_opaque_draw_mgr_.emplace("Opaque", 1000);
  static_opaque_alpha_mask_draw_mgr_.emplace("Opaque Alpha Mask", 1000);
  static_transparent_draw_mgr_.emplace("Transparent", 1000);
  draw_managers_ = {&static_opaque_draw_mgr_.value(), &static_opaque_alpha_mask_draw_mgr_.value(),
                    &static_transparent_draw_mgr_.value()};

  static_opaque_draw_mgr_->add_draw_pass("main_view");
  static_opaque_alpha_mask_draw_mgr_->add_draw_pass("main_view");
  static_transparent_draw_mgr_->add_draw_pass("main_view");
  main_mesh_pass_idx_ = 0;
  for (u32 i = 0; i < csm_->get_num_cascade_levels(); i++) {
    static_opaque_draw_mgr_->add_draw_pass("csm_" + std::to_string(i));
    static_opaque_alpha_mask_draw_mgr_->add_draw_pass("csm_" + std::to_string(i));
  }

  shadow_sampler_ = SamplerCache::get().get_or_create_sampler(
      SamplerCreateInfo{.min_filter = VK_FILTER_LINEAR,
                        .mag_filter = VK_FILTER_LINEAR,
                        .address_mode = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                        .border_color = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE});

  ibl_ = IBL{};

  // TODO: make a function for this lmao, so cringe
  {
    auto vert_buf_size = sizeof(cube_vertices);
    // auto index_buf_size = sizeof(cube_indices);
    auto* staging = StagingBufferPool::get().acquire(vert_buf_size);
    memcpy(staging->mapped_data(), cube_vertices, vert_buf_size);
    // memcpy(staging->mapped_data(), cube_indices, index_buf_size);
    cube_vertices_slot_ = static_vertex_buf_.allocator.allocate(vert_buf_size);
    // cube_indices_gpu_offset_ = static_index_buf_->alloc(index_buf_size);
    transfer_submit(
        [this, vert_buf_size, staging](VkCommandBuffer cmd, VkFence fence,
                                       std::queue<InFlightResource<vk2::Buffer*>>& transfers) {
          transfer_q_state_.reset(cmd)
              .transition_buffer_to_transfer_dst(static_vertex_buf_.get_buffer()->buffer())
              .transition_buffer_to_transfer_dst(static_index_buf_.get_buffer()->buffer())
              .flush_barriers();
          VkBufferCopy2KHR buf_copy =
              init::buffer_copy(0, cube_vertices_slot_.get_offset(), vert_buf_size);
          VkCopyBufferInfo2KHR buf_copy_info{.sType = VK_STRUCTURE_TYPE_COPY_BUFFER_INFO_2,
                                             .srcBuffer = staging->buffer(),
                                             .dstBuffer = static_vertex_buf_.get_buffer()->buffer(),
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
}

void VkRender2::on_draw(const SceneDrawInfo& info) {
  ZoneScoped;
  {
    ZoneScopedN("scene uniform buffer");
    auto& d = curr_frame_2();
    scene_uniform_cpu_data_.proj = glm::perspective(glm::radians(info.fov_degrees), aspect_ratio(),
                                                    near_far_z_.y, near_far_z_.x);
    scene_uniform_cpu_data_.proj[1][1] *= -1;
    scene_uniform_cpu_data_.view_proj = scene_uniform_cpu_data_.proj * info.view;
    scene_uniform_cpu_data_.view = info.view;
    scene_uniform_cpu_data_.debug_flags = uvec4{};
    if (ao_map_enabled.get()) {
      scene_uniform_cpu_data_.debug_flags.x |= AO_ENABLED_BIT;
    }
    if (normal_map_enabled.get()) {
      scene_uniform_cpu_data_.debug_flags.x |= NORMAL_MAPS_ENABLED_BIT;
    }
    if (csm_enabled.get()) {
      scene_uniform_cpu_data_.debug_flags.x |= CSM_ENABLED_BIT;
    }
    if (ibl_enabled.get()) {
      scene_uniform_cpu_data_.debug_flags.x |= IBL_ENABLED_BIT;
    }
    scene_uniform_cpu_data_.light_color = info.light_color;
    scene_uniform_cpu_data_.light_dir = glm::normalize(info.light_dir);
    scene_uniform_cpu_data_.debug_flags.w = debug_mode_;
    scene_uniform_cpu_data_.view_pos = info.view_pos;
    scene_uniform_cpu_data_.ambient_intensity = info.ambient_intensity;
    memcpy(d.scene_uniform_buf->mapped_data(), &scene_uniform_cpu_data_, sizeof(SceneUniforms));
  }
  if (draw_debug_aabbs_) {
    ZoneScopedN("debug aabbs");
    for (const auto& handle : loaded_model_instance_resources_) {
      if (auto* res = static_model_instance_pool_.get(handle); res != nullptr) {
        for (const auto& obj_data : res->object_datas) {
          draw_box(obj_data.model, AABB{obj_data.aabb_min, obj_data.aabb_max});
        }
      }
    }
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

  VkCommandBuffer cmd_buf = curr_frame().main_cmd_buffer;

  auto cmd_begin_info = vk2::init::command_buffer_begin_info();
  VK_CHECK(vkResetCommandBuffer(cmd_buf, 0));
  VK_CHECK(vkBeginCommandBuffer(cmd_buf, &cmd_begin_info));

  CmdEncoder cmd{cmd_buf, default_pipeline_layout_, curr_frame().tracy_vk_ctx};

  bind_bindless_descriptors(cmd);
  vk2::BindlessResourceAllocator::get().set_frame_num(curr_frame_num());
  vk2::BindlessResourceAllocator::get().flush_deletions();

  for (auto& instance : to_delete_static_model_instances_) {
    free(cmd, instance);
  }

  to_delete_static_model_instances_.clear();

  auto& swapchain_img = swapchain_.imgs[curr_swapchain_img_idx()];

  rg_.reset();
  rg_.set_swapchain_info(RenderGraphSwapchainInfo{
      .curr_img = swapchain_img, .width = swapchain_.dims.x, .height = swapchain_.dims.y});
  // auto& curr_frame_data = curr_frame_2();
  rg_.set_backbuffer_img("final_out");

  csm_->prepare_frame(rg_, curr_frame_num(), info.view, info.light_dir, aspect_ratio(),
                      info.fov_degrees, scene_aabb_, info.view_pos);

  if (!frustum_cull_settings_.paused) {
    cull_vp_matrices_.clear();
    cull_vp_matrices_.emplace_back(scene_uniform_cpu_data_.view_proj);
  }
  for (u32 cascade_level = 0; cascade_level < csm_->get_num_cascade_levels(); cascade_level++) {
    cull_vp_matrices_.emplace_back(csm_->get_light_matrices()[cascade_level]);
  }
  add_rendering_passes(rg_);
  auto res = rg_.bake();
  if (!res) {
    LERROR("bake error {}", res.error());
    exit(1);
  }

  auto add_resources = [this](StaticMeshDrawManager& mgr) {
    if (mgr.should_draw()) {
      for (const auto& draw_pass : mgr.get_draw_passes()) {
        rg_.set_resource(draw_pass.name, draw_pass.get_frame_out_draw_cmd_buf_handle());
      }
    }
  };
  add_resources(*static_opaque_draw_mgr_);
  add_resources(*static_opaque_alpha_mask_draw_mgr_);
  add_resources(*static_transparent_draw_mgr_);

  rg_.setup_attachments();
  rg_.execute(cmd);

  TracyVkCollect(curr_frame().tracy_vk_ctx, cmd_buf);
  VK_CHECK(vkEndCommandBuffer(cmd_buf));

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
  auto cmd_buf_submit_info = vk2::init::command_buffer_submit_info(cmd_buf);
  auto submit = vk2::init::queue_submit_info(SPAN1(cmd_buf_submit_info),
                                             std::span(wait_semaphores.data(), next_wait_sem_idx),
                                             SPAN1(signal_info));
  VK_CHECK(vkQueueSubmit2KHR(queues_.graphics_queue, 1, &submit, curr_frame().render_fence));

  line_draw_vertices_.clear();
}

void VkRender2::on_imgui() {
  if (ImGui::Begin("Renderer")) {
    if (ImGui::TreeNodeEx("Device")) {
      vk2::get_device().on_imgui();
      ImGui::TreePop();
    }
    if (ImGui::TreeNode("stats")) {
      if (ImGui::TreeNode("static geometry")) {
        ImGui::Text("Total vertices: %lu", (size_t)static_draw_stats_.total_vertices);
        ImGui::Text("Total indices: %lu", (size_t)static_draw_stats_.total_indices);
        ImGui::Text("Total triangles: %lu", (size_t)static_draw_stats_.total_vertices / 3);
        ImGui::Text("Vertices %u", static_draw_stats_.vertices);
        ImGui::Text("Indices: %u", static_draw_stats_.indices);
        ImGui::Text("Materials: %u", static_draw_stats_.materials);
        ImGui::Text("Textures: %u", static_draw_stats_.textures);
        ImGui::TreePop();
      }
      ImGui::TreePop();
    }
    if (ImGui::TreeNodeEx("Debug")) {
      ImGui::Checkbox("Draw AABBs", &draw_debug_aabbs_);
      if (ImGui::BeginCombo("Debug Mode", debug_mode_to_string(debug_mode_))) {
        for (u32 mode = 0; mode < DEBUG_MODE_COUNT; mode++) {
          if (ImGui::Selectable(debug_mode_to_string(mode), mode == debug_mode_)) {
            debug_mode_ = mode;
          }
        }
        ImGui::EndCombo();
      }
      ImGui::TreePop();
    }

    if (ImGui::TreeNodeEx("Static Geo", ImGuiTreeNodeFlags_DefaultOpen)) {
      auto static_mesh_mgr_gui = [](StaticMeshDrawManager& mgr) {
        if (ImGui::TreeNodeEx(mgr.get_name().c_str(), ImGuiTreeNodeFlags_DefaultOpen)) {
          ImGui::Checkbox("Enabled", &mgr.draw_enabled);
          ImGui::TreePop();
        }
      };
      static_mesh_mgr_gui(*static_opaque_draw_mgr_);
      static_mesh_mgr_gui(*static_opaque_alpha_mask_draw_mgr_);
      static_mesh_mgr_gui(*static_transparent_draw_mgr_);
      ImGui::TreePop();
    }

    if (ImGui::TreeNode("Shadows")) {
      csm_->on_imgui();
      ImGui::TreePop();
    }
    if (ImGui::TreeNodeEx("Culling")) {
      ImGui::Checkbox("Enabled", &frustum_cull_settings_.enabled);
      ImGui::Checkbox("Paused", &frustum_cull_settings_.paused);
      ImGui::TreePop();
    }

    if (ImGui::TreeNode("Postprocessing")) {
      if (ImGui::BeginCombo("Tonemapper", tonemap_type_names_[tonemap_type_])) {
        for (u32 i = 0; i < 2; i++) {
          if (ImGui::Selectable(tonemap_type_names_[i], tonemap_type_ == i)) {
            tonemap_type_ = i;
          }
        }
        ImGui::EndCombo();
      }
      ImGui::TreePop();
    }

    ImGui::Checkbox("Deferred Rendering", &deferred_enabled_);
    ImGui::Checkbox("Render prefilter env map skybox", &render_prefilter_mip_skybox_);
    ImGui::SliderInt("Prefilter Env Map Layer", &prefilter_mip_skybox_level_, 0,
                     ibl_->prefiltered_env_map_tex_->texture->create_info().mip_levels - 1);
    if (ImGui::TreeNode("textures")) {
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
      ImGui::TreePop();
    }
  }

  if (ImGui::TreeNodeEx("Scene", ImGuiTreeNodeFlags_DefaultOpen)) {
    // TODO: make this external to the renderer
    util::fixed_vector<u32, 8> to_delete;
    for (u32 i = 0; i < loaded_model_instance_resources_.size(); i++) {
      auto handle = loaded_model_instance_resources_[i];
      if (auto* res = static_model_instance_pool_.get(handle); res) {
        ImGui::PushID(res);
        ImGui::Text("%s", res->name);
        ImGui::SameLine();
        if (ImGui::Button("X")) {
          if (!to_delete.full()) {
            to_delete_static_model_instances_.emplace_back(std::move(*res));
            static_model_instance_pool_.destroy(handle);
            to_delete.push_back(i);
          }
        }
        ImGui::PopID();
      }
    }

    for (const u32 idx : to_delete) {
      if (loaded_model_instance_resources_.size() > 1) {
        loaded_model_instance_resources_[idx] = loaded_model_instance_resources_.back();
        loaded_model_instance_resources_.pop_back();
      }
    }
    ImGui::TreePop();
  }
  ImGui::End();
}

VkRender2::~VkRender2() {
  vkDeviceWaitIdle(device_);
  static_vertex_buf_.allocator.free(std::move(cube_vertices_slot_));
  for (auto& instance : loaded_model_instance_resources_) {
    if (auto* i = static_model_instance_pool_.get(instance); i) {
      free(*i);
    }
  }
}

void VkRender2::on_resize() {}

SceneHandle VkRender2::load_model(const std::filesystem::path& path, bool dynamic,
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
    auto handle_it = static_model_name_to_handle_.find(path);
    StaticModelGPUResourcesHandle resources_handle{};
    StaticModelGPUResources* resources{};
    if (handle_it != static_model_name_to_handle_.end()) {
      resources_handle = handle_it->second;
      resources = static_models_pool_.get(resources_handle);
    } else {
      auto ret = gfx::load_gltf(path, default_mat_data_);
      if (!ret.has_value()) {
        return {};
      }
      auto res = std::move(ret.value());

      u64 material_data_size = res.materials.size() * sizeof(gfx::Material);
      u64 vertices_size = res.vertices.size() * sizeof(gfx::Vertex);
      u64 indices_size = res.indices.size() * sizeof(u32);
      auto vertices_gpu_slot = static_vertex_buf_.allocator.allocate(vertices_size);
      auto indices_gpu_slot = static_index_buf_.allocator.allocate(indices_size);

      auto staging = LinearStagingBuffer{
          vk2::StagingBufferPool::get().acquire(material_data_size + vertices_size + indices_size)};
      u64 material_data_staging_offset = staging.copy(res.materials.data(), material_data_size);
      u64 vertices_staging_offset = staging.copy(res.vertices.data(), vertices_size);
      u64 indices_staging_offset = staging.copy(res.indices.data(), indices_size);

      static_draw_stats_.vertices += res.vertices.size();
      static_draw_stats_.indices += res.indices.size();
      static_draw_stats_.textures += res.textures.size();
      static_draw_stats_.materials += res.materials.size();
      auto materials_gpu_slot = static_materials_buf_.allocator.allocate(material_data_size);

      transfer_submit([&, this](VkCommandBuffer cmd, VkFence fence,
                                std::queue<InFlightResource<vk2::Buffer*>>& transfers) {
        copy_buffer(
            cmd, *staging.get_buffer(), *get_device().get_buffer(static_materials_buf_.buffer),
            material_data_staging_offset, materials_gpu_slot.get_offset(), material_data_size);
        copy_buffer(cmd, *staging.get_buffer(), *static_vertex_buf_.get_buffer(),
                    vertices_staging_offset, vertices_gpu_slot.get_offset(), vertices_size);
        copy_buffer(cmd, *staging.get_buffer(), *static_index_buf_.get_buffer(),
                    indices_staging_offset, indices_gpu_slot.get_offset(), indices_size);
        transfer_q_state_.reset(cmd)
            .queue_transfer_buffer(state_, VK_PIPELINE_STAGE_2_VERTEX_INPUT_BIT,
                                   VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT,
                                   static_vertex_buf_.get_buffer()->buffer(),
                                   queues_.transfer_queue_idx, queues_.graphics_queue_idx)
            .queue_transfer_buffer(state_, VK_PIPELINE_STAGE_2_INDEX_INPUT_BIT,
                                   VK_ACCESS_2_INDEX_READ_BIT,
                                   static_index_buf_.get_buffer()->buffer(),
                                   queues_.transfer_queue_idx, queues_.graphics_queue_idx)
            .queue_transfer_buffer(state_, VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT,
                                   VK_ACCESS_2_SHADER_READ_BIT,
                                   static_materials_buf_.get_buffer()->buffer(),
                                   queues_.transfer_queue_idx, queues_.graphics_queue_idx)
            .flush_barriers();
        transfers.emplace(staging.get_buffer(), fence);
      });
      resources_handle = static_models_pool_.alloc(
          std::move(res.scene_graph_data), std::move(res.mesh_draw_infos),
          std::move(materials_gpu_slot), std::move(vertices_gpu_slot), std::move(indices_gpu_slot),
          std::move(res.textures), vertices_gpu_slot.get_offset() / sizeof(gfx::Vertex),
          indices_gpu_slot.get_offset() / sizeof(u32), res.vertices.size(), res.indices.size(),
          path.string(), 0u);
      resources = static_models_pool_.get(resources_handle);
      static_model_name_to_handle_.emplace(resources->name, resources_handle);
    }
    static_draw_stats_.total_vertices += resources->num_vertices;
    static_draw_stats_.total_indices += resources->num_indices;

    u32 num_opaque_objs{}, num_opaque_alpha_mask_objs{}, num_transparent_objs{};
    for (auto& node : resources->scene_graph_data.node_datas) {
      for (auto& mesh_indices : node.meshes) {
        if (mesh_indices.pass_flags & PassFlags_Opaque) {
          num_opaque_objs++;
        } else if (mesh_indices.pass_flags & PassFlags_OpaqueAlpha) {
          num_opaque_alpha_mask_objs++;
        } else if (mesh_indices.pass_flags & PassFlags_Transparent) {
          num_transparent_objs++;
        }
      }
    }
    resources->ref_count++;
    u32 num_objs_tot = num_opaque_alpha_mask_objs + num_opaque_objs + num_transparent_objs;
    auto model_instance_resources_handle = static_model_instance_pool_.alloc();
    loaded_model_instance_resources_.emplace_back(model_instance_resources_handle);
    auto* instance_resources = static_model_instance_pool_.get(model_instance_resources_handle);
    instance_resources->model_resources_handle = resources_handle;
    instance_resources->name = resources->name.c_str();

    std::vector<InstanceData> instance_datas;
    instance_resources->object_datas.reserve(num_objs_tot);
    instance_datas.reserve(num_objs_tot);

    vec3 scene_min = vec3{std::numeric_limits<float>::max()};
    vec3 scene_max = vec3{std::numeric_limits<float>::lowest()};
    // TODO: no allocate
    std::vector<GPUDrawInfo> opaque_cmds;
    std::vector<GPUDrawInfo> alpha_mask_cmds;
    std::vector<GPUDrawInfo> transparent_cmds;
    opaque_cmds.reserve(num_opaque_objs);
    alpha_mask_cmds.reserve(num_opaque_alpha_mask_objs);
    transparent_cmds.reserve(num_transparent_objs);
    instance_resources->instance_data_slot =
        static_instance_data_buf_.allocator.allocate(num_objs_tot * sizeof(InstanceData));
    instance_resources->object_data_slot =
        static_object_data_buf_.allocator.allocate(num_objs_tot * sizeof(ObjectData));

    u32 base_instance_id =
        instance_resources->instance_data_slot.get_offset() / sizeof(InstanceData);
    u32 base_object_data_id =
        instance_resources->object_data_slot.get_offset() / sizeof(ObjectData);
    auto base_material_id = resources->materials_slot.get_offset() / sizeof(Material);

    bool is_non_identity_root_node_transform = transform != mat4{1};
    for (auto& node : resources->scene_graph_data.node_datas) {
      for (auto& node_mesh_data : node.meshes) {
        auto& mesh = resources->mesh_draw_infos[node_mesh_data.mesh_idx];
        mat4 model = is_non_identity_root_node_transform ? transform * node.world_transform
                                                         : node.world_transform;
        // https://stackoverflow.com/questions/6053522/how-to-recalculate-axis-aligned-bounding-box-after-translate-rotate/58630206#58630206
        auto transform_aabb = [](const glm::mat4& model, const AABB& aabb) -> AABB {
          AABB result;
          result.min = glm::vec3(model[3]);  // translation part
          result.max = result.min;
          for (int i = 0; i < 3; ++i) {    // for each row (x, y, z in result)
            for (int j = 0; j < 3; ++j) {  // for each column (x, y, z in input aabb)
              float a = model[i][j] * aabb.min[j];
              float b = model[i][j] * aabb.max[j];
              result.min[i] += glm::min(a, b);
              result.max[i] += glm::max(a, b);
            }
          }
          return result;
        };
        AABB world_space_aabb = transform_aabb(model, mesh.aabb);
        scene_min = glm::min(scene_min, world_space_aabb.min);
        scene_max = glm::max(scene_max, world_space_aabb.max);
        u32 instance_id = base_instance_id + instance_datas.size();
        instance_datas.emplace_back(node_mesh_data.material_id + base_material_id,
                                    base_object_data_id + instance_resources->object_datas.size());
        instance_resources->object_datas.emplace_back(gfx::ObjectData{
            .model = model,
            .aabb_min = vec4(mesh.aabb.min, 0.),
            .aabb_max = vec4(mesh.aabb.max, 0.),

        });
        GPUDrawInfo draw{
            .index_cnt = mesh.index_count,
            .first_index = static_cast<u32>(resources->first_index + mesh.first_index),
            .vertex_offset = static_cast<u32>(resources->first_vertex + mesh.first_vertex),
            .instance_id = instance_id};

        if (node_mesh_data.pass_flags & PassFlags_Opaque) {
          opaque_cmds.emplace_back(draw);
        } else if (node_mesh_data.pass_flags & PassFlags_OpaqueAlpha) {
          alpha_mask_cmds.emplace_back(draw);
        } else if (node_mesh_data.pass_flags & PassFlags_Transparent) {
          transparent_cmds.emplace_back(draw);
        }
      }
    }

    u64 obj_datas_size = instance_resources->object_datas.size() * sizeof(gfx::ObjectData);
    u64 instance_datas_size = instance_datas.size() * sizeof(InstanceData);

    scene_aabb_.min = glm::min(scene_aabb_.min, scene_min);
    scene_aabb_.max = glm::max(scene_aabb_.max, scene_max);

    u64 opaque_cmds_size = opaque_cmds.size() * sizeof(GPUDrawInfo);
    u64 opaque_alpha_cmds_size = alpha_mask_cmds.size() * sizeof(GPUDrawInfo);
    u64 transparent_cmds_size = transparent_cmds.size() * sizeof(GPUDrawInfo);
    auto staging = LinearStagingBuffer{vk2::StagingBufferPool::get().acquire(
        transparent_cmds_size + opaque_alpha_cmds_size + opaque_cmds_size + obj_datas_size +
        instance_datas_size)};
    u64 opaque_cmds_staging_offset = staging.copy(opaque_cmds.data(), opaque_cmds_size);
    u64 opaque_alpha_cmds_staging_offset =
        staging.copy(alpha_mask_cmds.data(), opaque_alpha_cmds_size);
    u64 transparent_cmds_staging_offset =
        staging.copy(transparent_cmds.data(), transparent_cmds_size);
    u64 obj_datas_staging_offset =
        staging.copy(instance_resources->object_datas.data(), obj_datas_size);
    u64 instance_datas_staging_offset = staging.copy(instance_datas.data(), instance_datas_size);

    transfer_submit([&, this](VkCommandBuffer cmd, VkFence fence,
                              std::queue<InFlightResource<vk2::Buffer*>>& transfers) {
      assert(obj_datas_size && instance_datas_size);
      // TODO: track the handles
      transfer_q_state_.reset(cmd);
      if (opaque_cmds_size) {
        instance_resources->opaque_draws_handle =
            static_opaque_draw_mgr_->add_draws(transfer_q_state_, cmd, opaque_cmds_size,
                                               opaque_cmds_staging_offset, *staging.get_buffer());
      }
      if (opaque_alpha_cmds_size) {
        instance_resources->opaque_alpha_draws_handle =
            static_opaque_alpha_mask_draw_mgr_->add_draws(
                transfer_q_state_, cmd, opaque_alpha_cmds_size, opaque_alpha_cmds_staging_offset,
                *staging.get_buffer());
      }
      if (transparent_cmds_size) {
        instance_resources->transparent_draws_handle = static_transparent_draw_mgr_->add_draws(
            transfer_q_state_, cmd, transparent_cmds_size, transparent_cmds_staging_offset,
            *staging.get_buffer());
      }
      copy_buffer(cmd, *staging.get_buffer(), *static_object_data_buf_.get_buffer(),
                  obj_datas_staging_offset, instance_resources->object_data_slot.get_offset(),
                  obj_datas_size);
      copy_buffer(cmd, *staging.get_buffer(), *static_instance_data_buf_.get_buffer(),
                  instance_datas_staging_offset,
                  instance_resources->instance_data_slot.get_offset(), instance_datas_size);

      transfer_q_state_
          .queue_transfer_buffer(
              state_,
              VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT,
              VK_ACCESS_2_SHADER_READ_BIT, static_object_data_buf_.get_buffer()->buffer(),
              queues_.transfer_queue_idx, queues_.graphics_queue_idx)
          .queue_transfer_buffer(state_, VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT,
                                 VK_ACCESS_2_SHADER_READ_BIT,
                                 static_instance_data_buf_.get_buffer()->buffer(),
                                 queues_.transfer_queue_idx, queues_.graphics_queue_idx)
          .flush_barriers();
      transfers.emplace(staging.get_buffer(), fence);
    });

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
  LINFO("transfer submit: {}", curr_frame_num());
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
  cull_objs_pipeline_ = PipelineManager::get().load_compute_pipeline({"cull_objects.comp"});
  postprocess_pipeline_ =
      PipelineManager::get().load_compute_pipeline({"postprocess/postprocess.comp"});
  img_pipeline_ = PipelineManager::get().load_compute_pipeline({"debug/clear_img.comp"});
  draw_pipeline_ = PipelineManager::get().load_graphics_pipeline(GraphicsPipelineCreateInfo{
      .vertex_path = "debug/basic.vert",
      .fragment_path = "debug/basic.frag",
      .rendering = {.color_formats = {to_vkformat(draw_img_format_)},
                    .depth_format = to_vkformat(depth_img_format_)},
      .rasterization = {.cull_mode = CullMode::None},
      .depth_stencil = GraphicsPipelineCreateInfo::depth_enable(true, CompareOp::GreaterOrEqual),
  });

  skybox_pipeline_ = PipelineManager::get().load_graphics_pipeline(GraphicsPipelineCreateInfo{
      .vertex_path = "skybox/skybox.vert",
      .fragment_path = "skybox/skybox.frag",
      .rendering = {.color_formats = {to_vkformat(draw_img_format_)},
                    .depth_format = to_vkformat(depth_img_format_)},
      .depth_stencil = GraphicsPipelineCreateInfo::depth_enable(true, CompareOp::GreaterOrEqual),
  });
  gbuffer_pipeline_ = PipelineManager::get().load_graphics_pipeline(GraphicsPipelineCreateInfo{
      .vertex_path = "gbuffer/gbuffer.vert",
      .fragment_path = "gbuffer/gbuffer.frag",
      .rendering = {.color_formats = {to_vkformat(gbuffer_a_format_),
                                      to_vkformat(gbuffer_b_format_),
                                      to_vkformat(gbuffer_c_format_)},
                    .depth_format = to_vkformat(depth_img_format_)},
      .rasterization = {.cull_mode = CullMode::None},
      .depth_stencil = GraphicsPipelineCreateInfo::depth_enable(true, CompareOp::GreaterOrEqual),
  });
  deferred_shade_pipeline_ = PipelineManager::get().load_compute_pipeline({"gbuffer/shade.comp"});
  line_draw_pipeline_ = PipelineManager::get().load_graphics_pipeline(GraphicsPipelineCreateInfo{
      .vertex_path = "lines/draw_line.vert",
      .fragment_path = "lines/draw_line.frag",
      .topology = PrimitiveTopology::LineList,
      .rendering = {.color_formats = {swapchain_.format},
                    .depth_format = to_vkformat(depth_img_format_)},
      .rasterization = {.cull_mode = CullMode::None},
      .depth_stencil = GraphicsPipelineCreateInfo::depth_enable(false, CompareOp::GreaterOrEqual),
  });
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

void VkRender2::add_rendering_passes(RenderGraph& rg) {
  ZoneScoped;
  {
    auto& clear_buff = rg.add_pass("clear_draw_cnt_buf");

    for (auto& mgr : draw_managers_) {
      if (mgr->should_draw()) {
        for (const auto& draw_pass : mgr->get_draw_passes()) {
          clear_buff.add_proxy(draw_pass.name, Access::TransferWrite);
        }
      }
    }

    clear_buff.set_execute_fn([this](CmdEncoder& cmd) {
      TracyVkZone(curr_frame().tracy_vk_ctx, cmd.cmd(), "clear_draw_cnt_buf");
      for (auto& mgr : draw_managers_) {
        if (!mgr->should_draw()) continue;
        for (const auto& draw_pass : mgr->get_draw_passes()) {
          vk2::Buffer* buf = draw_pass.get_frame_out_draw_cmd_buf();
          assert(buf);
          if (!buf) {
            continue;
          }
          if (portable_) {
            // TODO: only fill the unfilled portion after culling?

            // fill whole buffer with 0 since can't use draw indirect count.
            vkCmdFillBuffer(cmd.cmd(), buf->buffer(), 0, buf->size(), 0);
          } else {
            // fill draw cnt with 0
            vkCmdFillBuffer(cmd.cmd(), buf->buffer(), 0, sizeof(u32), 0);
          }
        }
      }
    });
  }

  {
    auto& cull = rg.add_pass("cull");
    for (const auto& mgr : draw_managers_) {
      if (mgr->should_draw()) {
        for (const auto& draw_pass : mgr->get_draw_passes()) {
          cull.add_proxy(draw_pass.name, Access::ComputeRW);
        }
      }
    }
    cull.set_execute_fn([this](CmdEncoder& cmd) {
      TracyVkZone(curr_frame().tracy_vk_ctx, cmd.cmd(), "cull");
      PipelineManager::get().bind_compute(cmd.cmd(), cull_objs_pipeline_);
      for (const auto& mgr : draw_managers_) {
        if (mgr->should_draw()) {
          for (u32 i = 0; i < mgr->get_draw_passes().size(); i++) {
            assert(i < cull_vp_matrices_.size());
            if (i >= cull_vp_matrices_.size()) break;
            const auto& draw_pass = mgr->get_draw_passes()[i];
            // extract frustum planes
            auto extract_planes_from_projmat = [](const mat4& mat, vec4& left, vec4& right,
                                                  vec4& bottom, vec4& top, vec4& near, vec4& far) {
              for (int i = 4; i--;) {
                left[i] = mat[i][3] + mat[i][0];
              }
              for (int i = 4; i--;) {
                right[i] = mat[i][3] - mat[i][0];
              }
              for (int i = 4; i--;) {
                bottom[i] = mat[i][3] + mat[i][1];
              }
              for (int i = 4; i--;) {
                top[i] = mat[i][3] - mat[i][1];
              }
              for (int i = 4; i--;) {
                near[i] = mat[i][3] + mat[i][2];
              }
              for (int i = 4; i--;) {
                far[i] = mat[i][3] - mat[i][2];
              }
              auto normalize_plane = [](vec4& plane) { plane /= glm::length(vec3(plane)); };
              normalize_plane(left);
              normalize_plane(right);
              normalize_plane(bottom);
              normalize_plane(top);
              normalize_plane(near);
              normalize_plane(far);
            };
            vec4 left, right, bottom, top, near, far;
            const auto& vp = cull_vp_matrices_[i];
            extract_planes_from_projmat(vp, left, right, bottom, top, near, far);
            u32 flags{};
            if (frustum_cull_settings_.enabled) {
              flags |= FRUSTUM_CULL_ENABLED_BIT;
            }
            u32 count = static_cast<u32>(mgr->get_draw_info_buf()->size() / sizeof(GPUDrawInfo));
            CullObjectPushConstants pc{
                left,
                right,
                bottom,
                top,
                near,
                far,
                curr_frame_2().scene_uniform_buf->device_addr(),
                count,
                mgr->get_draw_info_buf()->resource_info_->handle,
                draw_pass.get_frame_out_draw_cmd_buf()->resource_info_->handle,
                static_object_data_buf_.get_buffer()->resource_info_->handle,
                flags,
            };
            cmd.push_constants(sizeof(pc), &pc);
            cmd.dispatch((count + 256) / 256, 1, 1);
          }
        }
      }
    });
  }

  if (csm_enabled.get()) {
    csm_->add_pass(rg);
    csm_->debug_shadow_pass(rg, linear_sampler_.value());
  }

  if (!deferred_enabled_) {
    auto& forward = rg.add_pass("forward");
    auto final_out_handle =
        forward.add("draw_out", {.format = draw_img_format_}, Access::ColorWrite);
    for (const auto& mgr : draw_managers_) {
      if (mgr->should_draw()) {
        forward.add_proxy(mgr->get_draw_passes()[main_mesh_pass_idx_].name, Access::IndirectRead);
      }
    }
    if (csm_enabled.get()) {
      forward.add_proxy("shadow_data_buf", Access::FragmentRead);
      forward.add("shadow_map_img", csm_->get_shadow_map_att_info(), Access::FragmentRead);
    }
    auto depth_out_handle =
        forward.add("depth", {.format = depth_img_format_}, Access::DepthStencilWrite);
    forward.set_execute_fn([&rg, final_out_handle, this, depth_out_handle](CmdEncoder& cmd) {
      TracyVkZone(curr_frame().tracy_vk_ctx, cmd.cmd(), "forward");
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

      BasicPushConstants pc{
          curr_frame_2().scene_uniform_buf->resource_info_->handle,
          static_vertex_buf_.get_buffer()->resource_info_->handle,
          static_instance_data_buf_.get_buffer()->resource_info_->handle,
          static_object_data_buf_.get_buffer()->resource_info_->handle,
          static_materials_buf_.get_buffer()->resource_info_->handle,
          linear_sampler_->resource_info.handle,
          get_device()
              .get_buffer(csm_->get_shadow_data_buffer(curr_frame_num()))
              ->resource_info_->handle,
          shadow_sampler_.resource_info.handle,
          get_device().get_image(csm_->get_shadow_map_img())->view().sampled_img_resource().handle,
          ibl_->irradiance_cubemap_tex_->view().sampled_img_resource().handle,
          ibl_->brdf_lut_->view().sampled_img_resource().handle,
          ibl_->prefiltered_env_map_tex_->texture->view().sampled_img_resource().handle,
          linear_sampler_clamp_to_edge_->resource_info.handle};
      cmd.push_constants(default_pipeline_layout_, sizeof(pc), &pc);
      PipelineManager::get().bind_graphics(cmd.cmd(), draw_pipeline_);
      execute_static_geo_draws(cmd);
      draw_skybox(cmd);
      vkCmdEndRenderingKHR(cmd.cmd());
    });
  } else {
    {
      auto& gbuffer = rg.add_pass("gbuffer");
      auto rg_gbuffer_a =
          gbuffer.add("gbuffer_a", {.format = gbuffer_a_format_}, Access::ColorWrite);
      auto rg_gbuffer_b =
          gbuffer.add("gbuffer_b", {.format = gbuffer_b_format_}, Access::ColorWrite);
      auto rg_gbuffer_c =
          gbuffer.add("gbuffer_c", {.format = gbuffer_c_format_}, Access::ColorWrite);

      if (static_opaque_draw_mgr_->should_draw()) {
        gbuffer.add_proxy(static_opaque_draw_mgr_->get_draw_passes()[main_mesh_pass_idx_].name,
                          Access::IndirectRead);
      }
      if (static_opaque_alpha_mask_draw_mgr_->should_draw()) {
        gbuffer.add_proxy(
            static_opaque_alpha_mask_draw_mgr_->get_draw_passes()[main_mesh_pass_idx_].name,
            Access::IndirectRead);
      }
      auto depth_out_handle =
          gbuffer.add("depth", {.format = depth_img_format_}, Access::DepthStencilWrite);
      gbuffer.set_execute_fn([&rg, rg_gbuffer_a, rg_gbuffer_b, rg_gbuffer_c, this,
                              depth_out_handle](CmdEncoder& cmd) {
        TracyVkZone(curr_frame().tracy_vk_ctx, cmd.cmd(), "gbuffer");
        auto* gbuffer_a = rg.get_texture(rg_gbuffer_a);
        auto* gbuffer_b = rg.get_texture(rg_gbuffer_b);
        auto* gbuffer_c = rg.get_texture(rg_gbuffer_c);
        assert(gbuffer_a && gbuffer_b && gbuffer_c);
        if (!gbuffer_a || !gbuffer_b || !gbuffer_c) {
          return;
        }

        PipelineManager::get().bind_graphics(cmd.cmd(), gbuffer_pipeline_);

        VkClearValue clear_value{.color = {{0.0, 0.0, 0.0, 0.0}}};
        VkRenderingAttachmentInfo color_atts[] = {
            vk2::init::rendering_attachment_info(
                gbuffer_a->view().view(), VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, &clear_value),
            vk2::init::rendering_attachment_info(
                gbuffer_b->view().view(), VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, &clear_value),
            vk2::init::rendering_attachment_info(
                gbuffer_c->view().view(), VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, &clear_value)};
        VkClearValue depth_clear{};
        auto depth_att = init::rendering_attachment_info(
            rg.get_texture(depth_out_handle)->view().view(),
            VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL_KHR, &depth_clear);
        VkExtent2D render_extent{gbuffer_a->create_info().extent.width,
                                 gbuffer_a->create_info().extent.height};
        VkRenderingInfo render_info = vk2::init::rendering_info(
            render_extent, color_atts, COUNTOF(color_atts), &depth_att, nullptr);
        vkCmdBeginRenderingKHR(cmd.cmd(), &render_info);
        set_viewport_and_scissor(cmd.cmd(), render_extent);

        GBufferPushConstants pc{
            static_vertex_buf_.get_buffer()->device_addr(),
            curr_frame_2().scene_uniform_buf->device_addr(),
            static_instance_data_buf_.get_buffer()->device_addr(),
            static_object_data_buf_.get_buffer()->device_addr(),
            static_materials_buf_.get_buffer()->device_addr(),
            linear_sampler_->resource_info.handle,
        };
        cmd.push_constants(sizeof(pc), &pc);
        execute_static_geo_draws(cmd);
        vkCmdEndRenderingKHR(cmd.cmd());
      });
    }
    {
      auto& shade = rg.add_pass("shade");
      auto rg_gbuffer_a =
          shade.add("gbuffer_a", {.format = gbuffer_a_format_}, Access::ComputeRead);
      auto rg_gbuffer_b =
          shade.add("gbuffer_b", {.format = gbuffer_b_format_}, Access::ComputeRead);
      auto rg_gbuffer_c =
          shade.add("gbuffer_c", {.format = gbuffer_c_format_}, Access::ComputeRead);
      if (csm_enabled.get()) {
        shade.add("shadow_map_img", Access::FragmentRead);
        shade.add_proxy("shadow_data_buf", Access::ComputeRead);
      }
      auto depth_handle = shade.add("depth", {.format = depth_img_format_}, Access::ComputeSample);
      auto final_out_handle =
          shade.add("draw_out", {.format = draw_img_format_}, Access::ComputeWrite);
      shade.set_execute_fn([&rg, rg_gbuffer_b, rg_gbuffer_c, rg_gbuffer_a, final_out_handle, this,
                            depth_handle](CmdEncoder& cmd) {
        TracyVkZone(curr_frame().tracy_vk_ctx, cmd.cmd(), "shade");
        auto* gbuffer_a = rg.get_texture(rg_gbuffer_a);
        auto* gbuffer_b = rg.get_texture(rg_gbuffer_b);
        auto* gbuffer_c = rg.get_texture(rg_gbuffer_c);
        auto* depth_img = rg.get_texture(depth_handle);
        auto* shade_out_tex = rg.get_texture(final_out_handle);
        assert(gbuffer_a && gbuffer_b && gbuffer_c && shade_out_tex && depth_img);
        if (!gbuffer_a || !gbuffer_b || !gbuffer_c || !shade_out_tex || !depth_img) {
          return;
        }
        DeferredShadePushConstants pc{
            glm::inverse(scene_uniform_cpu_data_.view_proj),
            gbuffer_a->view().storage_img_resource().handle,
            gbuffer_b->view().storage_img_resource().handle,
            gbuffer_c->view().storage_img_resource().handle,
            depth_img->view().sampled_img_resource().handle,
            shade_out_tex->view().storage_img_resource().handle,
            nearest_sampler_->resource_info.handle,
            curr_frame_2().scene_uniform_buf->resource_info_->handle,
            get_device()
                .get_image(csm_->get_shadow_map_img())
                ->view()
                .sampled_img_resource()
                .handle,
            shadow_sampler_.resource_info.handle,
            get_device()
                .get_buffer(csm_->get_shadow_data_buffer(curr_frame_num()))
                ->resource_info_->handle,
            ibl_->irradiance_cubemap_tex_->view().sampled_img_resource().handle,
            ibl_->brdf_lut_->view().sampled_img_resource().handle,
            ibl_->prefiltered_env_map_tex_->texture->view().sampled_img_resource().handle,
            linear_sampler_clamp_to_edge_->resource_info.handle};
        PipelineManager::get().bind_compute(cmd.cmd(), deferred_shade_pipeline_);

        cmd.push_constants(sizeof(pc), &pc);
        cmd.dispatch((gbuffer_a->create_info().extent.width + 16) / 16,
                     (gbuffer_a->create_info().extent.height + 16) / 16, 1);
      });
    }
    {
      auto& skybox = rg.add_pass("skybox");
      auto draw_tex = skybox.add("draw_out", {.format = draw_img_format_}, Access::ColorRW);
      auto depth_handle =
          skybox.add("depth", {.format = depth_img_format_}, Access::DepthStencilRead);

      skybox.set_execute_fn([this, &rg, draw_tex, depth_handle](CmdEncoder& cmd) {
        TracyVkZone(curr_frame().tracy_vk_ctx, cmd.cmd(), "skybox");
        auto* tex = rg.get_texture(draw_tex);
        VkRenderingAttachmentInfo color_attachment = vk2::init::rendering_attachment_info(
            tex->view().view(), VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
        auto depth_att =
            init::rendering_attachment_info(rg.get_texture(depth_handle)->view().view(),
                                            VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL_KHR);
        VkExtent2D render_extent{tex->create_info().extent.width, tex->create_info().extent.height};
        VkRenderingInfo render_info =
            vk2::init::rendering_info(render_extent, &color_attachment, &depth_att, nullptr);
        vkCmdBeginRenderingKHR(cmd.cmd(), &render_info);
        set_viewport_and_scissor(cmd.cmd(), render_extent);

        draw_skybox(cmd);
        vkCmdEndRenderingKHR(cmd.cmd());
      });
    }
  }

  {
    auto& pp = rg.add_pass("post_process");
    auto draw_out_handle = pp.add("draw_out", {.format = draw_img_format_}, Access::ComputeRead);
    auto final_out_handle = pp.add("final_out", swapchain_att_info_, Access::ComputeWrite);
    pp.set_execute_fn([this, &rg, draw_out_handle, final_out_handle](CmdEncoder& cmd) {
      TracyVkZone(curr_frame().tracy_vk_ctx, cmd.cmd(), "post_process");
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
          u32 in_tex_idx, out_tex_idx, flags, tonemap_type;
        } pc{rg.get_texture(draw_out_handle)->view().storage_img_resource().handle,
             post_processed_img->view().storage_img_resource().handle, postprocess_flags,
             tonemap_type_};
        cmd.push_constants(default_pipeline_layout_, sizeof(pc), &pc);
        vkCmdDispatch(cmd.cmd(), (post_processed_img->extent_2d().width + 16) / 16,
                      (post_processed_img->extent_2d().height + 16) / 16, 1);
      }
    });
  }

  if (line_draw_vertices_.size()) {
    vk2::Buffer* line_draw_buf = get_device().get_buffer(curr_frame_2().line_draw_buf);
    assert(line_draw_buf);
    size_t required_size = line_draw_vertices_.size() * sizeof(LineVertex);
    if (required_size > line_draw_buf->size()) {
      curr_frame_2().line_draw_buf = get_device().create_buffer_holder(
          BufferCreateInfo{.size = glm::max(line_draw_buf->size() * 2, required_size),
                           .usage = BufferUsage_Storage,
                           .flags = BufferCreateFlags_HostVisible});
      line_draw_buf = get_device().get_buffer(curr_frame_2().line_draw_buf);
    }
    memcpy(line_draw_buf->mapped_data(), line_draw_vertices_.data(), required_size);
  }

  if (draw_imgui) {
    auto& imgui_p = rg.add_pass("ui");
    RenderResourceHandle csm_debug_img_handle{UINT32_MAX};
    if (csm_enabled.get() && csm_->get_debug_render_enabled()) {
      csm_debug_img_handle = imgui_p.add("shadow_map_debug_img", Access::FragmentRead);
    }
    auto color_handle = imgui_p.add("final_out", swapchain_att_info_, Access::ColorRW);
    auto depth_handle = imgui_p.add("depth", Access::DepthStencilRead);
    imgui_p.set_execute_fn([this, color_handle, &rg, csm_debug_img_handle,
                            depth_handle](CmdEncoder& cmd) {
      TracyVkZone(curr_frame().tracy_vk_ctx, cmd.cmd(), "imgui");
      if (csm_enabled.get() && csm_->get_debug_render_enabled()) {
        assert(csm_debug_img_handle != UINT32_MAX);
        auto* csm_debug_img = rg.get_texture(csm_debug_img_handle);
        assert(csm_debug_img);
        if (csm_debug_img) {
          csm_->imgui_pass(cmd, *linear_sampler_, *csm_debug_img);
        }
      }
      auto* color_tex = rg.get_texture(color_handle);
      auto* depth_tex = rg.get_texture(depth_handle);
      if (line_draw_vertices_.size()) {
        auto color_att = init::rendering_attachment_info(color_tex->view().view(),
                                                         VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
        auto depth_att = init::rendering_attachment_info(depth_tex->view().view(),
                                                         VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL);
        auto rendering_info = init::rendering_info(color_tex->extent_2d(), &color_att, &depth_att);
        vkCmdBeginRenderingKHR(cmd.cmd(), &rendering_info);
        PipelineManager::get().bind_graphics(cmd.cmd(), line_draw_pipeline_);
        vk2::Buffer* line_draw_buf = get_device().get_buffer(curr_frame_2().line_draw_buf);
        LinesRenderPushConstants pc{
            .vtx = line_draw_buf->device_addr(),
            .scene_buffer = curr_frame_2().scene_uniform_buf->device_addr()};
        cmd.push_constants(sizeof(pc), &pc);
        vkCmdDraw(cmd.cmd(), line_draw_vertices_.size(), 1, 0, 0);
        vkCmdEndRenderingKHR(cmd.cmd());
      }
      render_imgui(cmd.cmd(), {color_tex->extent_2d().width, color_tex->extent_2d().height},
                   color_tex->view().view());
    });
  }
}

void VkRender2::execute_draw(CmdEncoder& cmd, const vk2::Buffer& buffer, u32 draw_count) const {
  VkBuffer draw_cmd_buf = buffer.buffer();
  constexpr u32 draw_cmd_offset{sizeof(u32)};
  if (portable_) {
    vkCmdDrawIndexedIndirect(cmd.cmd(), draw_cmd_buf, draw_cmd_offset, draw_count,
                             sizeof(VkDrawIndexedIndirectCommand));
  } else {
    vkCmdDrawIndexedIndirectCount(cmd.cmd(), draw_cmd_buf, draw_cmd_offset, draw_cmd_buf, 0,
                                  draw_count, sizeof(VkDrawIndexedIndirectCommand));
  }
}

void VkRender2::StaticMeshDrawManager::remove_draws(StateTracker& state, VkCommandBuffer cmd,
                                                    Handle handle) {
  if (!handle.is_valid()) {
    return;
  }
  Alloc* a = allocs_.get(handle);
  assert(a);
  if (!a) return;
  num_draw_cmds_ -= a->draw_cmd_slot.get_size() / sizeof(GPUDrawInfo);

  vkCmdFillBuffer(cmd, draw_cmds_buf_.get_buffer()->buffer(), a->draw_cmd_slot.get_offset(),
                  a->draw_cmd_slot.get_size(), 0);
  state.queue_transfer_buffer(
      VkRender2::get().state_,
      VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
      VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_2_SHADER_READ_BIT,
      draw_cmds_buf_.get_buffer()->buffer(), VkRender2::get().queues_.transfer_queue_idx,
      VkRender2::get().queues_.graphics_queue_idx, a->draw_cmd_slot.get_offset(),
      a->draw_cmd_slot.get_size());
  draw_cmds_buf_.allocator.free(std::move(a->draw_cmd_slot));

  allocs_.destroy(handle);
}

VkRender2::StaticMeshDrawManager::Handle VkRender2::StaticMeshDrawManager::add_draws(
    StateTracker& state, VkCommandBuffer cmd, size_t size, size_t staging_offset,
    vk2::Buffer& staging) {
  assert(size > 0);
  Alloc a;
  a.draw_cmd_slot = draw_cmds_buf_.allocator.allocate(size);
  auto* draw_cmds_buf = draw_cmds_buf_.get_buffer();
  assert(draw_cmds_buf);
  if (!draw_cmds_buf) {
    return {};
  }

  // resize draw cmd bufs
  size_t curr_tot_draw_cmd_buf_size = draw_cmds_buf->size();
  auto new_size = glm::max<size_t>(curr_tot_draw_cmd_buf_size * 2,
                                   a.draw_cmd_slot.get_offset() + a.draw_cmd_slot.get_size());
  for (auto& draw_pass : draw_passes_) {
    // resize output draw cmd buffers
    if (auto* buf = get_device().get_buffer(draw_pass.out_draw_cmds_buf[0].handle);
        a.draw_cmd_slot.get_offset() + a.draw_cmd_slot.get_size() + sizeof(u32) >= buf->size()) {
      for (u32 i = 0; i < VkRender2::get().frames_in_flight_; i++) {
        draw_pass.out_draw_cmds_buf[i] = get_device().create_buffer_holder(BufferCreateInfo{
            .size = new_size + sizeof(u32),
            .usage = BufferUsage_Indirect | BufferUsage_Storage,
        });
      }
    }
  }

  if (a.draw_cmd_slot.get_offset() + a.draw_cmd_slot.get_size() >= curr_tot_draw_cmd_buf_size) {
    // draw cmd buf resize and copy
    auto new_buf = get_device().create_buffer_holder(BufferCreateInfo{
        .size = new_size,
        .usage = BufferUsage_Storage,
    });
    VkRender2::get().copy_buffer(cmd, draw_cmds_buf, get_device().get_buffer(new_buf), 0, 0,
                                 curr_tot_draw_cmd_buf_size);
    state.queue_transfer_buffer(
        VkRender2::get().state_,
        VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_2_SHADER_READ_BIT,
        get_device().get_buffer(new_buf)->buffer(), VkRender2::get().queues_.transfer_queue_idx,
        VkRender2::get().queues_.graphics_queue_idx, 0, curr_tot_draw_cmd_buf_size);
    draw_cmds_buf_.buffer = std::move(new_buf);
    draw_cmds_buf = draw_cmds_buf_.get_buffer();
  }

  if (a.draw_cmd_slot.get_offset() >= draw_cmds_buf->size()) {
    LINFO("unimplemented: need to resize Static mesh draw cmd buffer");
    exit(1);
  }
  VkBufferCopy2KHR copy = init::buffer_copy(staging_offset, a.draw_cmd_slot.get_offset(), size);
  VkCopyBufferInfo2KHR buf_copy_info{.sType = VK_STRUCTURE_TYPE_COPY_BUFFER_INFO_2,
                                     .srcBuffer = staging.buffer(),
                                     .dstBuffer = draw_cmds_buf->buffer(),
                                     .regionCount = 1,
                                     .pRegions = &copy};
  vkCmdCopyBuffer2KHR(cmd, &buf_copy_info);
  state.queue_transfer_buffer(
      VkRender2::get().state_,
      VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
      VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_2_SHADER_READ_BIT, draw_cmds_buf->buffer(),
      VkRender2::get().queues_.transfer_queue_idx, VkRender2::get().queues_.graphics_queue_idx,
      a.draw_cmd_slot.get_offset(), size);

  num_draw_cmds_ += size / sizeof(GPUDrawInfo);
  return allocs_.alloc(std::move(a));
}

VkRender2::StaticMeshDrawManager::StaticMeshDrawManager(std::string name,
                                                        size_t initial_max_draw_cnt)
    : name_(std::move(name)) {
  draw_cmds_buf_.buffer = get_device().create_buffer_holder(BufferCreateInfo{
      .size = initial_max_draw_cnt * sizeof(GPUDrawInfo),
      .usage = BufferUsage_Storage,
  });
  draw_cmds_buf_.allocator.init(initial_max_draw_cnt * sizeof(GPUDrawInfo), sizeof(GPUDrawInfo),
                                100);
}

vk2::Buffer* VkRender2::StaticMeshDrawManager::get_draw_info_buf() const {
  return draw_cmds_buf_.get_buffer();
}

BufferHandle VkRender2::StaticMeshDrawManager::get_draw_info_buf_handle() const {
  return draw_cmds_buf_.buffer.handle;
}

void VkRender2::execute_static_geo_draws(CmdEncoder& cmd) {
  vkCmdBindIndexBuffer(cmd.cmd(), static_index_buf_.get_buffer()->buffer(), 0,
                       VK_INDEX_TYPE_UINT32);
  if (static_opaque_draw_mgr_->should_draw()) {
    execute_draw(cmd,
                 *static_opaque_draw_mgr_->get_draw_passes()[main_mesh_pass_idx_]
                      .get_frame_out_draw_cmd_buf(),
                 static_opaque_draw_mgr_->get_num_draw_cmds());
  }
  if (static_opaque_alpha_mask_draw_mgr_->should_draw()) {
    execute_draw(cmd,
                 *static_opaque_alpha_mask_draw_mgr_->get_draw_passes()[main_mesh_pass_idx_]
                      .get_frame_out_draw_cmd_buf(),
                 static_opaque_alpha_mask_draw_mgr_->get_num_draw_cmds());
  }
}

VkRender2::StaticMeshDrawManager::DrawPass::DrawPass(std::string name, u32 count)
    : name(std::move(name)) {
  for (u32 i = 0; i < VkRender2::get().frames_in_flight_; i++) {
    out_draw_cmds_buf[i] = get_device().create_buffer_holder(BufferCreateInfo{
        .size = (count * sizeof(VkDrawIndexedIndirectCommand)) + sizeof(u32),
        .usage = BufferUsage_Indirect | BufferUsage_Storage,
    });
  }
}

void VkRender2::StaticMeshDrawManager::add_draw_pass(const std::string& name) {
  draw_passes_.emplace_back(name_ + "_" + name, num_draw_cmds_);
}

BufferHandle VkRender2::StaticMeshDrawManager::DrawPass::get_frame_out_draw_cmd_buf_handle() const {
  return out_draw_cmds_buf[VkRender2::get().curr_frame_in_flight_num()].handle;
}

vk2::Buffer* VkRender2::StaticMeshDrawManager::DrawPass::get_frame_out_draw_cmd_buf() const {
  return get_device().get_buffer(out_draw_cmds_buf[VkRender2::get().curr_frame_in_flight_num()]);
}

void VkRender2::copy_buffer(VkCommandBuffer cmd, vk2::Buffer* src, vk2::Buffer* dst,
                            size_t src_offset, size_t dst_offset, size_t size) {
  VkBufferCopy2KHR copy = init::buffer_copy(src_offset, dst_offset, size);
  VkCopyBufferInfo2KHR buf_copy_info{.sType = VK_STRUCTURE_TYPE_COPY_BUFFER_INFO_2,
                                     .srcBuffer = src->buffer(),
                                     .dstBuffer = dst->buffer(),
                                     .regionCount = 1,
                                     .pRegions = &copy};
  vkCmdCopyBuffer2KHR(cmd, &buf_copy_info);
}
void VkRender2::copy_buffer(VkCommandBuffer cmd, BufferHandle src, BufferHandle dst,
                            size_t src_offset, size_t dst_offset, size_t size) {
  copy_buffer(cmd, get_device().get_buffer(src), get_device().get_buffer(dst), src_offset,
              dst_offset, size);
}

void VkRender2::draw_line(const vec3& p1, const vec3& p2, const vec4& color) {
  line_draw_vertices_.emplace_back(vec4{p1, 0.}, color);
  line_draw_vertices_.emplace_back(vec4{p2, 0.}, color);
}

void VkRender2::draw_cube(VkCommandBuffer cmd) const {
  vkCmdDraw(cmd, 36, 1, cube_vertices_slot_.get_offset() / sizeof(gfx::Vertex), 0);
}

void VkRender2::draw_skybox(CmdEncoder& cmd) {
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
  } pc{curr_frame_2().scene_uniform_buf->resource_info_->handle, skybox_handle,
       linear_sampler_->resource_info.handle};
  cmd.push_constants(default_pipeline_layout_, sizeof(pc), &pc);
  vkCmdDraw(cmd.cmd(), 36, 1, 0, 0);
}

void VkRender2::draw_plane(const vec3& o, const vec3& v1, const vec3& v2, float s1, float s2,
                           u32 n1, u32 n2, const vec4& color, const vec4& outline_color) {
  // draw outline quad
  vec3 bot_left = o - .5f * s1 * v1 - .5f * s2 * v2;
  vec3 top_left = o - .5f * s1 * v1 + .5f * s2 * v2;
  vec3 top_right = o + .5f * s1 * v1 + .5f * s2 * v2;
  vec3 bot_right = o + .5f * s1 * v1 - .5f * s2 * v2;
  draw_line(bot_left, top_left, outline_color);
  draw_line(top_left, top_right, outline_color);
  draw_line(top_right, bot_right, outline_color);
  draw_line(bot_right, bot_left, outline_color);
  // draw n1 horizontal and n2 vertical lines
  n1++;
  n2++;
  for (u32 i = 1; i < n1; i++) {
    float t = ((float)i - (float)n1 / 2.0f) * s1 / (float)n1;
    const vec3 o1 = o + t * v1;
    draw_line(o1 - s2 / 2.0f * v2, o1 + s2 / 2.0f * v2, color);
  }
  for (u32 i = 1; i < n2; i++) {
    const float t = ((float)i - (float)n2 / 2.0f) * s2 / (float)n2;
    const vec3 o2 = o + t * v2;
    draw_line(o2 - s1 / 2.0f * v1, o2 + s1 / 2.0f * v1, color);
  }
}

void VkRender2::draw_box(const mat4& model, const vec3& size, const vec4& color) {
  std::array<vec3, 8> pts{{
      {-size.x, -size.y, -size.z},
      {-size.x, -size.y, size.z},
      {-size.x, size.y, -size.z},
      {-size.x, size.y, size.z},
      {size.x, -size.y, -size.z},
      {size.x, -size.y, size.z},
      {size.x, size.y, -size.z},
      {size.x, size.y, size.z},
  }};
  for (auto& p : pts) {
    p = model * vec4(p, 1.f);
  }
  draw_line(pts[0], pts[1], color);
  draw_line(pts[0], pts[2], color);
  draw_line(pts[1], pts[3], color);
  draw_line(pts[2], pts[3], color);
  draw_line(pts[0], pts[4], color);
  draw_line(pts[1], pts[5], color);
  draw_line(pts[2], pts[6], color);
  draw_line(pts[4], pts[6], color);
  draw_line(pts[4], pts[5], color);
  draw_line(pts[3], pts[7], color);
  draw_line(pts[5], pts[7], color);
  draw_line(pts[6], pts[7], color);
}

void VkRender2::draw_box(const mat4& model, const AABB& aabb, const vec4& color) {
  draw_box(model * glm::translate(mat4{1.f}, .5f * (aabb.min + aabb.max)),
           .5f * (aabb.max - aabb.min), color);
}

VkRender2::StaticModelGPUResources::~StaticModelGPUResources() {
  VkRender2::get().static_materials_buf_.allocator.free(std::move(materials_slot));
  VkRender2::get().static_vertex_buf_.allocator.free(std::move(vertices_slot));
}

void VkRender2::free(StaticModelInstanceResources& instance) {
  static_instance_data_buf_.allocator.free(std::move(instance.instance_data_slot));
  static_object_data_buf_.allocator.free(std::move(instance.object_data_slot));
  // free the draws (need to clear to 0)
  auto* resources = static_models_pool_.get(instance.model_resources_handle);
  if (resources->ref_count == 0) {
    // lol
    LERROR("uh oh");
    exit(1);
  }
  resources->ref_count--;
  if (resources->ref_count == 0) {
    static_model_name_to_handle_.erase(resources->name);
    static_draw_stats_.vertices -= resources->num_vertices;
    static_draw_stats_.indices -= resources->num_indices;
    static_draw_stats_.materials -= resources->materials_slot.get_size() / sizeof(Material);
    static_draw_stats_.textures -= resources->textures.size();
    static_materials_buf_.allocator.free(std::move(resources->materials_slot));
    static_vertex_buf_.allocator.free(std::move(resources->vertices_slot));
    static_index_buf_.allocator.free(std::move(resources->indices_slot));
    static_models_pool_.destroy(instance.model_resources_handle);
  }
}

void VkRender2::free(CmdEncoder& cmd, StaticModelInstanceResources& instance) {
  static_opaque_draw_mgr_->remove_draws(state_, cmd.cmd(), instance.opaque_draws_handle);
  static_opaque_alpha_mask_draw_mgr_->remove_draws(state_, cmd.cmd(),
                                                   instance.opaque_alpha_draws_handle);
  static_transparent_draw_mgr_->remove_draws(state_, cmd.cmd(), instance.transparent_draws_handle);
  free(instance);
}

}  // namespace gfx
