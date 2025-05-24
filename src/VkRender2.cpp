#include "VkRender2.hpp"

// clang-format off
#include <volk.h>
#include <vulkan/vk_enum_string_helper.h>
#include <vulkan/vulkan_core.h>
// clang-format on

#include <cassert>
#include <filesystem>
#include <memory>
#include <tracy/Tracy.hpp>
#include <tracy/TracyVulkan.hpp>
#include <utility>

#include "CommandEncoder.hpp"
#include "GLFW/glfw3.h"
#include "RenderGraph.hpp"
#include "Scene.hpp"
#include "SceneLoader.hpp"
#include "StateTracker.hpp"
#include "ThreadPool.hpp"
#include "Types.hpp"
#include "core/Logger.hpp"
#include "glm/packing.hpp"
#include "imgui/imgui.h"
#include "shaders/common.h.glsl"
#include "shaders/cull_objects_common.h.glsl"
#include "shaders/debug/basic_common.h.glsl"
#include "shaders/gbuffer/gbuffer_common.h.glsl"
#include "shaders/gbuffer/shade_common.h.glsl"
#include "shaders/lines/draw_line_common.h.glsl"
#include "shaders/shadow_depth_common.h.glsl"
#include "util/CVar.hpp"
#include "util/IndexAllocator.hpp"
#include "vk2/Buffer.hpp"
#include "vk2/Device.hpp"
#include "vk2/Initializers.hpp"
#include "vk2/PipelineManager.hpp"
#include "vk2/ShaderCompiler.hpp"
#include "vk2/Texture.hpp"
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

}  // namespace

VkRender2& VkRender2::get() {
  assert(vkrender2_instance);
  return *vkrender2_instance;
}

void VkRender2::init(const InitInfo& info, bool& success) {
  assert(!vkrender2_instance);
  new VkRender2{info, success};
}

void VkRender2::shutdown() {
  assert(vkrender2_instance);
  delete vkrender2_instance;
}

using namespace vk2;

void VkRender2::new_frame() { device_->new_imgui_frame(); }

VkRender2::VkRender2(const InitInfo& info, bool& success)
    : device_(info.device), window_(info.window), resource_dir_(info.resource_dir) {
  success = true;
  vkrender2_instance = this;

  if (resource_dir_.empty() || !std::filesystem::exists(resource_dir_)) {
    LCRITICAL("cannot initialize renderer, resource dir not provided or does not exist");
    success = false;
    return;
  }
  if (!info.window) {
    LCRITICAL("cannot initialize renderer, window not provided");
    success = false;
    return;
  }
  if (!info.device) {
    LCRITICAL("cannot initialize renderer, window not provided");
    success = false;
    return;
  }

  {
    ZoneScopedN("init per frame");
    // auto& device = get_device();
    per_frame_data_.resize(device_->get_frames_in_flight());
    // for (auto& frame : per_frame_data_) {
    //   // frame.tracy_vk_ctx =
    //   //     TracyVkContext(device.get_physical_device(), device.device(),
    //   //                    device.get_queue(QueueType::Graphics).queue, frame.main_cmd_buffer);
    // }
  }

  device_->init_imgui();

  PipelineManager::init(device_->device(), resource_dir_ / "shaders", true,
                        device_->default_pipeline_layout_);

  uint32_t white = glm::packUnorm4x8(glm::vec4(1, 1, 1, 1));
  auto copy_cmd = device_->graphics_copy_allocator_.allocate(32);
  memcpy((char*)device_->get_buffer(copy_cmd.staging_buffer)->mapped_data(), (void*)&white,
         sizeof(u32));
  default_data_.white_img = Holder<ImageHandle>{device_->create_image(ImageDesc{
      .format = Format::R8G8B8A8Srgb, .dims = {1, 1, 1}, .bind_flags = BindFlag::ShaderResource})};

  {
    state_.reset(copy_cmd.transfer_cmd_buf);
    state_
        .transition(*device_->get_image(default_data_.white_img), VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                    VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)
        .flush_barriers();
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
    VkCopyBufferToImageInfo2 copy{
        .sType = VK_STRUCTURE_TYPE_COPY_BUFFER_TO_IMAGE_INFO_2_KHR,
        .srcBuffer = device_->get_buffer(copy_cmd.staging_buffer)->buffer(),
        .dstImage = device_->get_image(default_data_.white_img)->image(),
        .dstImageLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        .regionCount = 1,
        .pRegions = &img_copy};
    vkCmdCopyBufferToImage2KHR(copy_cmd.transfer_cmd_buf, &copy);
    state_.transition(device_->get_image(default_data_.white_img)->image(),
                      VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                      VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                      VK_IMAGE_LAYOUT_READ_ONLY_OPTIMAL);
    state_.flush_barriers();
    device_->graphics_copy_allocator_.submit(copy_cmd);
  }

  nearest_sampler_ = device_->get_or_create_sampler({
      .min_filter = FilterMode::Nearest,
      .mag_filter = FilterMode::Nearest,
      .mipmap_mode = FilterMode::Nearest,
      .address_mode = AddressMode::Repeat,
  });
  if (device_->get_bindless_idx(nearest_sampler_) != 1) {
    LCRITICAL("invalid sampler idx {}", device_->get_bindless_idx(nearest_sampler_));
    exit(1);
  }

  linear_sampler_ = device_->get_or_create_sampler({
      .min_filter = FilterMode::Linear,
      .mag_filter = FilterMode::Linear,
      .mipmap_mode = FilterMode::Linear,
      .address_mode = AddressMode::Repeat,
  });
  if (device_->get_bindless_idx(linear_sampler_) != 2) {
    LCRITICAL("invalid sampler idx linear");
    exit(1);
  }
  linear_sampler_clamp_to_edge_ = device_->get_or_create_sampler({
      .min_filter = FilterMode::Linear,
      .mag_filter = FilterMode::Linear,
      .mipmap_mode = FilterMode::Linear,
      .address_mode = AddressMode::ClampToEdge,
  });

  default_mat_data_.white_img_handle =
      device_->get_bindless_idx(default_data_.white_img.handle, SubresourceType::Shader);

  {
    // per frame scene uniforms
    per_frame_data_.resize(device_->get_frames_in_flight());
    for (auto& d : per_frame_data_) {
      d.scene_uniform_buf =
          device_->create_buffer_holder(BufferCreateInfo{.size = sizeof(SceneUniforms),
                                                         .usage = BufferUsage_Storage,
                                                         .flags = BufferCreateFlags_HostVisible});

      d.line_draw_buf =
          device_->create_buffer_holder(BufferCreateInfo{.size = sizeof(LineVertex) * 1000,
                                                         .usage = BufferUsage_Storage,
                                                         .flags = BufferCreateFlags_HostVisible});
    }
  }

  auto vertices_size = 10'000'000 * sizeof(gfx::Vertex);
  static_vertex_buf_.buffer = device_->create_buffer_holder(
      {BufferCreateInfo{.size = vertices_size, .usage = BufferUsage_Storage}});
  static_vertex_buf_.allocator.init(vertices_size, sizeof(Vertex));

  auto indices_size = 10'000'000 * sizeof(u32);
  static_index_buf_.buffer = device_->create_buffer_holder(BufferCreateInfo{
      .size = indices_size,
      .usage = BufferUsage_Index,
  });
  static_index_buf_.allocator.init(indices_size, sizeof(u32));
  u64 max_static_draws = 100'000;

  auto static_materials_size = 1000 * sizeof(gfx::Material);
  static_materials_buf_.buffer = device_->create_buffer_holder(
      {BufferCreateInfo{.size = static_materials_size, .usage = BufferUsage_Storage}});
  static_materials_buf_.allocator.init(static_materials_size, sizeof(Material), 100);

  static_instance_data_buf_.buffer = device_->create_buffer_holder(BufferCreateInfo{
      .size = max_static_draws * sizeof(GPUInstanceData),
      .usage = BufferUsage_Storage,
  });
  static_instance_data_buf_.allocator.init(max_static_draws * sizeof(GPUInstanceData),
                                           sizeof(GPUInstanceData), 100);

  static_object_data_buf_.buffer = device_->create_buffer_holder(BufferCreateInfo{
      .size = max_static_draws * sizeof(ObjectData),
      .usage = BufferUsage_Storage,
  });
  static_object_data_buf_.allocator.init(max_static_draws * sizeof(ObjectData), sizeof(ObjectData),
                                         100);

  // TODO: make a function for this lmao, so cringe
  {
    auto vert_buf_size = sizeof(cube_vertices);
    auto copy_cmd = device_->graphics_copy_allocator_.allocate(vert_buf_size);
    memcpy(device_->get_buffer(copy_cmd.staging_buffer)->mapped_data(), cube_vertices,
           vert_buf_size);
    cube_vertex_buf_ = device_->create_buffer_holder(
        BufferCreateInfo{.size = vert_buf_size, .usage = BufferUsage_Storage});

    state_.reset(copy_cmd.transfer_cmd_buf)
        .buffer_barrier(static_vertex_buf_.get_buffer()->buffer(), VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                        VK_ACCESS_2_TRANSFER_WRITE_BIT)
        .buffer_barrier(static_index_buf_.get_buffer()->buffer(), VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                        VK_ACCESS_2_TRANSFER_WRITE_BIT)
        .flush_barriers();
    copy_cmd.copy_buffer(device_, *device_->get_buffer(cube_vertex_buf_), 0, 0, vert_buf_size);
    device_->graphics_copy_allocator_.submit(copy_cmd);
  }

  shadow_sampler_ = device_->get_or_create_sampler(
      SamplerCreateInfo{.min_filter = FilterMode::Nearest,
                        .mag_filter = FilterMode::Nearest,
                        .address_mode = AddressMode::ClampToEdge,
                        .border_color = BorderColor::FLoatOpaqueWhite});

  PipelineLoader loader;
  loader.reserve(20);
  loader.add_compute("cull_objects.comp", &cull_objs_pipeline_)
      .add_compute("postprocess/postprocess.comp", &postprocess_pipeline_)
      .add_compute("debug/clear_img.comp", &img_pipeline_)
      .add_compute("gbuffer/shade.comp", &deferred_shade_pipeline_)
      .add_graphics(
          GraphicsPipelineCreateInfo{
              .shaders = {{"debug/basic.vert", ShaderType::Vertex},
                          {"debug/basic.frag", ShaderType::Fragment}},
              .rendering = {.color_formats = {convert_format(draw_img_format_)},
                            .depth_format = convert_format(depth_img_format_)},
              .depth_stencil =
                  GraphicsPipelineCreateInfo::depth_enable(true, CompareOp::GreaterOrEqual),
              .name = "basic draw",
          },
          &draw_pipeline_)
      .add_graphics(
          GraphicsPipelineCreateInfo{
              .shaders = {{"skybox/skybox.vert", ShaderType::Vertex},
                          {"skybox/skybox.frag", ShaderType::Fragment}},
              .rendering = {.color_formats = {convert_format(draw_img_format_)},
                            .depth_format = convert_format(depth_img_format_)},
              .depth_stencil =
                  GraphicsPipelineCreateInfo::depth_enable(true, CompareOp::GreaterOrEqual),
              .name = "skybox"},
          &skybox_pipeline_)
      .add_graphics(
          GraphicsPipelineCreateInfo{
              .shaders = {{"lines/draw_line.vert", ShaderType::Vertex},
                          {"lines/draw_line.frag", ShaderType::Fragment}},
              .topology = PrimitiveTopology::LineList,
              .rendering = {.color_formats = {convert_format(device_->get_swapchain_info().format)},
                            .depth_format = convert_format(depth_img_format_)},
              .depth_stencil =
                  GraphicsPipelineCreateInfo::depth_enable(false, CompareOp::GreaterOrEqual),
              .name = "lines"},
          &line_draw_pipeline_);

  GraphicsPipelineCreateInfo gbuffer_info{
      .shaders = {{"gbuffer/gbuffer.vert", ShaderType::Vertex},
                  {"gbuffer/gbuffer.frag", ShaderType::Fragment}},
      .rendering = {.color_formats = {convert_format(gbuffer_a_format_),
                                      convert_format(gbuffer_b_format_),
                                      convert_format(gbuffer_c_format_)},
                    .depth_format = convert_format(depth_img_format_)},
      .depth_stencil = GraphicsPipelineCreateInfo::depth_enable(true, CompareOp::GreaterOrEqual),
      .name = "gbuffer"};

  loader.add_graphics(gbuffer_info, &gbuffer_pipeline_);
  auto alpha_mask_gbuffer_info = gbuffer_info;
  alpha_mask_gbuffer_info.shaders[1].defines = {"#define ALPHA_MASK_ENABLED 1\n"};
  alpha_mask_gbuffer_info.name = "gbuffer alpha mask";
  loader.add_graphics(alpha_mask_gbuffer_info, &gbuffer_alpha_mask_pipeline_);

  csm_ = std::make_unique<CSM>(
      &get_device(),
      [this](CmdEncoder& cmd, const mat4& vp, bool opaque_alpha, u32 cascade_i) {
        u32 pass_i = opaque_alpha ? MeshPass_OpaqueAlphaMask : MeshPass_Opaque;
        const StaticMeshDrawManager* mgr = &static_draw_mgrs_[pass_i];
        if (!mgr->should_draw()) return;
        ShadowDepthPushConstants pc{
            vp,
            static_vertex_buf_.get_buffer()->device_addr(),
            static_instance_data_buf_.get_buffer()->device_addr(),
            static_object_data_buf_.get_buffer()->device_addr(),
            device_->get_buffer(curr_frame().scene_uniform_buf)->resource_info_->handle,
            static_materials_buf_.get_buffer()->resource_info_->handle,
            device_->get_bindless_idx(linear_sampler_),
        };
        cmd.push_constants(sizeof(pc), &pc);
        cmd.bind_index_buffer(static_index_buf_.buffer.handle);
        cmd.bind_index_buffer(static_index_buf_.buffer.handle);
        for (int double_sided = 0; double_sided < 2; double_sided++) {
          cmd.set_cull_mode(double_sided ? CullMode::None : CullMode::Back);
          execute_draw(cmd,
                       mgr->get_draw_pass(shadow_mesh_pass_indices_[cascade_i][pass_i])
                           .get_frame_out_draw_cmd_buf_handle(double_sided),
                       mgr->get_num_draw_cmds(double_sided));
        }
      },
      [this](RenderGraphPass& pass) {
        MeshPass opaque_passes[] = {MeshPass_Opaque, MeshPass_OpaqueAlphaMask};
        for (size_t draw_pass_i = 0; draw_pass_i < COUNTOF(opaque_passes); draw_pass_i++) {
          auto& mgr = static_draw_mgrs_[draw_pass_i];
          if (mgr.should_draw()) {
            for (u32 cascade_i = 0; cascade_i < csm_->get_num_cascade_levels(); cascade_i++) {
              for (int double_sided = 0; double_sided < 2; double_sided++) {
                pass.add(mgr.get_draw_pass(shadow_mesh_pass_indices_[cascade_i][draw_pass_i])
                             .get_frame_out_draw_cmd_buf_handle(double_sided),
                         Access::IndirectRead);
              }
            }
          }
        }
      });

  csm_->load_pipelines(loader);
  ibl_ = IBL{device_, cube_vertex_buf_.handle};
  ibl_->load_pipelines(loader);
  loader.flush();
  ibl_->init_post_pipeline_load();

  for (int i = 0; i < MeshPass_Count; i++) {
    static_draw_mgrs_[i].init(static_cast<MeshPass>(i), 1000, device_);
    main_view_mesh_pass_indices_[i] = static_draw_mgrs_[i].add_draw_pass();
  }

  // main draw passes
  for (u32 i = 0; i < csm_->get_num_cascade_levels(); i++) {
    auto& state = shadow_mesh_pass_indices_.emplace_back();
    for (u32 pass_i : opaque_mesh_pass_idxs_) {
      state[pass_i] = static_draw_mgrs_[pass_i].add_draw_pass();
    }
  }
  default_env_map_path_ = info.resource_dir / "hdr" / "newport_loft.hdr";
}

void VkRender2::draw(const SceneDrawInfo& info) {
  ZoneScoped;
  {
    on_imgui();
    ImGui::Render();
    device_->begin_frame();
  }
  {
    ZoneScopedN("scene uniform buffer");
    auto& d = curr_frame();
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
    memcpy(device_->get_buffer(d.scene_uniform_buf)->mapped_data(), &scene_uniform_cpu_data_,
           sizeof(SceneUniforms));
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

  // VkCommandBuffer cmd_buf = curr_frame().main_cmd_buffer;
  // VK_CHECK(vkResetCommandPool(device_->device(), curr_frame().cmd_pool, 0));
  // auto cmd_begin_info = init::command_buffer_begin_info();
  // VK_CHECK(vkBeginCommandBuffer(cmd_buf, &cmd_begin_info));

  handle_load_scene_requests();

  CmdEncoder* cmd = device_->begin_command_list(QueueType::Graphics);
  state_.reset(*cmd);
  device_->bind_bindless_descriptors(*cmd);

  for (auto& instance : to_delete_static_model_instances_) {
    free(*cmd, instance);
  }

  to_delete_static_model_instances_.clear();

  rg_.reset();
  rg_.set_backbuffer_img("final_out");
  csm_->prepare_frame(device_->curr_frame_num(), info.view, info.light_dir, aspect_ratio(),
                      info.fov_degrees, scene_aabb_, info.view_pos);

  if (!frustum_cull_settings_.paused) {
    cull_vp_matrices_.clear();
    cull_vp_matrices_.emplace_back(scene_uniform_cpu_data_.view_proj);
  }
  for (u32 cascade_level = 0; cascade_level < csm_->get_num_cascade_levels(); cascade_level++) {
    cull_vp_matrices_.emplace_back(csm_->get_light_matrices()[cascade_level]);
  }
  add_rendering_passes(rg_);
  auto res = rg_.bake(cmd);
  if (!res) {
    LERROR("bake error {}", res.error());
    exit(1);
  }

  rg_.setup_attachments();
  rg_.execute(*cmd);

  // TracyVkCollect(curr_frame().tracy_vk_ctx, cmd_buf);

  // TODO: refactor queues lmao
  // std::array<VkSemaphoreSubmitInfo, 10> wait_semaphores{};
  // u32 next_wait_sem_idx{0};
  // wait_semaphores[next_wait_sem_idx++] = init::semaphore_submit_info(
  //     device_->curr_swapchain_semaphore(),
  //     VK_PIPELINE_STAGE_2_TRANSFER_BIT | VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT);
  // auto signal_info = init::semaphore_submit_info(device_->curr_frame().render_semaphore,
  //                                                VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT);
  // auto cmd_buf_submit_info = init::command_buffer_submit_info(cmd_buf);
  // auto submit = init::queue_submit_info(SPAN1(cmd_buf_submit_info),
  //                                       std::span(wait_semaphores.data(), next_wait_sem_idx),
  //                                       SPAN1(signal_info));
  // device_->queue_submit(QueueType::Graphics, SPAN1(submit));

  for (auto& wait_for : frame_imm_submits_) {
    device_->cmd_list_wait(cmd, wait_for);
  }
  frame_imm_submits_.clear();

  device_->submit_commands();
  line_draw_vertices_.clear();
}

void VkRender2::on_imgui() {
  if (ImGui::Begin("Renderer")) {
    if (ImGui::TreeNodeEx("Device")) {
      device_->on_imgui();
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

    if (ImGui::TreeNode("Pipelines")) {
      bool debug_shader = PipelineManager::get().get_shader_debug_mode();
      if (ImGui::Checkbox("Debug info in shaders on compilation", &debug_shader)) {
        PipelineManager::get().set_shader_debug_mode(debug_shader);
      }

      if (ImGui::Button("Reload All Shaders")) {
        PipelineManager::get().reload_shaders();
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
      for (auto& mgr : static_draw_mgrs_) {
        if (ImGui::TreeNodeEx(mgr.get_name().c_str(), ImGuiTreeNodeFlags_DefaultOpen)) {
          ImGui::Checkbox("Enabled", &mgr.draw_enabled);
          ImGui::TreePop();
        }
      }

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
    ImGui::SliderInt("Prefilter Env Map Layer", &prefilter_mip_skybox_render_mip_level_, 0,
                     device_->get_image(ibl_->prefiltered_env_map_tex_)->get_desc().mip_levels - 1);
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
  ZoneScoped;
  threads::pool.wait();
  handle_load_scene_requests();
  device_->wait_idle();
  for (auto& frame : per_frame_data_) {
    frame.scene_uniform_buf = {};
    frame.line_draw_buf = {};
  }
}

ModelHandle VkRender2::load_model(const std::filesystem::path& path, bool, const mat4& transform) {
  ZoneScoped;
  if (!std::filesystem::exists(path)) {
    LERROR("load_static_model: path doesn't exist: {}", path.string());
    return {};
  }

  auto handle_it = static_model_name_to_handle_.find(path);
  if (handle_it != static_model_name_to_handle_.end()) {
    load_instance_reqs_.emplace_back(handle_it->second, transform);
  } else {
    threads::pool.submit_task([this, path, transform]() {
      auto ret = gfx::load_gltf(path, default_mat_data_);
      if (ret.has_value()) {
        load_scene_reqs_.emplace_back(path, std::move(ret.value()), transform);
      }
    });
  }
  return {};
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

ImageHandle VkRender2::load_hdr_img(const std::filesystem::path& path, bool flip) {
  auto cpu_img_data = gfx::loader::load_hdr(path, 4, flip);
  if (!cpu_img_data.has_value()) return {};
  auto tex_handle = device_->create_image(ImageDesc{.type = ImageDesc::Type::TwoD,
                                                    .format = Format::R32G32B32A32Sfloat,
                                                    .dims = {cpu_img_data->w, cpu_img_data->h, 1},
                                                    .bind_flags = BindFlag::ShaderResource});
  auto cnt = cpu_img_data->w * cpu_img_data->h;
  u64 cpu_img_size = sizeof(float) * cnt * 4;
  auto copy_cmd = device_->graphics_copy_allocator_.allocate(cpu_img_size);
  auto* mapped = (float*)device_->get_buffer(copy_cmd.staging_buffer)->mapped_data();
  assert(mapped);
  const float* src = cpu_img_data->data;
  memcpy(mapped, src, cpu_img_size);
  VkImageMemoryBarrier2 barrier{.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
  auto* img = device_->get_image(tex_handle);
  barrier.image = img->image();
  barrier.srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
  barrier.srcAccessMask = 0;
  barrier.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
  barrier.dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
  barrier.oldLayout = img->curr_layout;
  barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
  img->curr_layout = barrier.newLayout;
  barrier.subresourceRange = VkImageSubresourceRange{.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                                                     .baseMipLevel = 0,
                                                     .levelCount = VK_REMAINING_MIP_LEVELS,
                                                     .baseArrayLayer = 0,
                                                     .layerCount = VK_REMAINING_ARRAY_LAYERS};
  auto dep_info = vk2::init::dependency_info({}, SPAN1(barrier));
  vkCmdPipelineBarrier2KHR(copy_cmd.transfer_cmd_buf, &dep_info);

  auto dims = device_->get_image(tex_handle)->size();
  VkBufferImageCopy2 img_copy_info{.sType = VK_STRUCTURE_TYPE_BUFFER_IMAGE_COPY_2,
                                   .bufferOffset = 0,
                                   .imageSubresource =
                                       {
                                           .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                                           .mipLevel = 0,
                                           .baseArrayLayer = 0,
                                           .layerCount = 1,
                                       },
                                   .imageExtent = {dims.x, dims.y, 1}};
  VkCopyBufferToImageInfo2 copy_to_img_info{
      .sType = VK_STRUCTURE_TYPE_COPY_BUFFER_TO_IMAGE_INFO_2_KHR,
      .srcBuffer = device_->get_buffer(copy_cmd.staging_buffer)->buffer(),
      .dstImage = device_->get_image(tex_handle)->image(),
      .dstImageLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
      .regionCount = 1,
      .pRegions = &img_copy_info};
  vkCmdCopyBufferToImage2KHR(copy_cmd.transfer_cmd_buf, &copy_to_img_info);
  device_->graphics_copy_allocator_.submit(copy_cmd);

  std::swap(barrier.srcStageMask, barrier.dstStageMask);
  barrier.oldLayout = img->curr_layout;
  img->curr_layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
  barrier.newLayout = img->curr_layout;
  barrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
  barrier.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
  device_->init_transitions_.push_back(barrier);
  return tex_handle;
}

void VkRender2::generate_mipmaps(StateTracker& state, CmdEncoder& cmd, ImageHandle handle) {
  // TODO: this is unbelievably hacky: using manual barriers outside of state tracker and then
  // updating state tracker with the result. solve this by updating state tracker to manage
  // subresource ranges or toss it and make a render graph
  auto* tex = device_->get_image(handle);
  assert(tex);
  u32 array_layers = tex->get_desc().array_layers;
  u32 mip_levels = get_mip_levels(tex->size());
  VkExtent2D curr_img_size = {tex->size().x, tex->size().y};
  for (u32 mip_level = 0; mip_level < mip_levels; mip_level++) {
    VkImageMemoryBarrier2 img_barrier{.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
    img_barrier.image = tex->image();
    img_barrier.srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT;
    img_barrier.srcStageMask = VK_PIPELINE_STAGE_2_BLIT_BIT;
    img_barrier.dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT;
    img_barrier.dstStageMask = VK_PIPELINE_STAGE_2_BLIT_BIT;
    img_barrier.oldLayout = tex->curr_layout;
    tex->curr_layout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    img_barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    img_barrier.subresourceRange = VkImageSubresourceRange{.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                                                           .baseMipLevel = mip_level,
                                                           .levelCount = 1,
                                                           .baseArrayLayer = 0,
                                                           .layerCount = array_layers};
    auto dep_info = init::dependency_info({}, SPAN1(img_barrier));
    vkCmdPipelineBarrier2KHR(cmd.cmd(), &dep_info);
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
                               .srcImage = tex->image(),
                               .srcImageLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                               .dstImage = tex->image(),
                               .dstImageLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                               .regionCount = 1,
                               .pRegions = &blit,
                               .filter = VK_FILTER_LINEAR};
      vkCmdBlitImage2KHR(cmd.cmd(), &info);
      curr_img_size = half_img_size;
    }
  }
  {
    auto* img_state = state.get_img_state(tex->image());
    // TODO: remove
    cmd.transition_image(handle, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL);
    if (img_state) {
      img_state->curr_layout = VK_IMAGE_LAYOUT_GENERAL;
      img_state->curr_access = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;
      img_state->curr_stage = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    }
  }
}

void VkRender2::set_env_map(const std::filesystem::path& path) { ibl_->load_env_map(path); }

void VkRender2::immediate_submit(std::function<void(CmdEncoder& ctx)>&& function) {
  CmdEncoder* cmd = device_->begin_command_list(QueueType::Graphics);
  frame_imm_submits_.emplace_back(cmd);
  state_.reset(*cmd);
  function(*cmd);
}

void VkRender2::generate_mipmaps(CmdEncoder& ctx, ImageHandle handle) {
  VkCommandBuffer cmd = ctx.cmd();
  auto* tex = device_->get_image(handle);
  u32 array_layers = tex->get_desc().array_layers;
  u32 mip_levels = get_mip_levels(tex->size());
  VkExtent2D curr_img_size = {tex->size().x, tex->size().y};
  for (u32 mip_level = 0; mip_level < mip_levels; mip_level++) {
    VkImageMemoryBarrier2 img_barrier{.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
    img_barrier.image = tex->image();
    img_barrier.srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT;
    img_barrier.srcStageMask = VK_PIPELINE_STAGE_2_BLIT_BIT;
    img_barrier.dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT;
    img_barrier.dstStageMask = VK_PIPELINE_STAGE_2_BLIT_BIT;
    img_barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    img_barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    tex->curr_layout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
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
                               .srcImage = tex->image(),
                               .srcImageLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                               .dstImage = tex->image(),
                               .dstImageLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                               .regionCount = 1,
                               .pRegions = &blit,
                               .filter = VK_FILTER_LINEAR};
      vkCmdBlitImage2KHR(cmd, &info);
      curr_img_size = half_img_size;
    }
  }
  ctx.transition_image(handle, VK_IMAGE_LAYOUT_GENERAL);
}

void VkRender2::add_rendering_passes(RenderGraph& rg) {
  ZoneScoped;
  {
    auto& clear_buff = rg.add_pass("clear_draw_cnt_buf");

    for (auto& mgr : static_draw_mgrs_) {
      if (mgr.should_draw()) {
        for (const auto& draw_pass : mgr.get_draw_passes()) {
          clear_buff.add(draw_pass.get_frame_out_draw_cmd_buf_handle(false), Access::TransferWrite);
          clear_buff.add(draw_pass.get_frame_out_draw_cmd_buf_handle(true), Access::TransferWrite);
        }
      }
    }

    clear_buff.set_execute_fn([this](CmdEncoder& cmd) {
      for (auto& mgr : static_draw_mgrs_) {
        if (!mgr.should_draw()) continue;
        for (const auto& draw_pass : mgr.get_draw_passes()) {
          for (int double_sided = 0; double_sided < 2; double_sided++) {
            auto buf = draw_pass.get_frame_out_draw_cmd_buf_handle(double_sided);
            if (!device_->is_supported(DeviceFeature::DrawIndirectCount)) {
              // TODO: only fill the unfilled portion after culling?
              // fill whole buffer with 0 since can't use draw indirect count.
              cmd.fill_buffer(buf, 0, constants::whole_size, 0);
            } else {
              cmd.fill_buffer(buf, 0, sizeof(u32), 0);
            }
          }
        }
      }
    });
  }

  {
    auto& cull = rg.add_pass("cull");
    for (const auto& mgr : static_draw_mgrs_) {
      if (mgr.should_draw()) {
        for (const auto& draw_pass : mgr.get_draw_passes()) {
          cull.add(draw_pass.get_frame_out_draw_cmd_buf_handle(false), Access::ComputeRW);
          cull.add(draw_pass.get_frame_out_draw_cmd_buf_handle(true), Access::ComputeRW);
        }
      }
    }
    cull.set_execute_fn([this](CmdEncoder& cmd) {
      cmd.bind_pipeline(PipelineBindPoint::Compute, cull_objs_pipeline_);
      for (const auto& mgr : static_draw_mgrs_) {
        if (mgr.should_draw()) {
          for (u32 i = 0; i < mgr.get_draw_passes().size(); i++) {
            assert(i < cull_vp_matrices_.size());
            if (i >= cull_vp_matrices_.size()) break;
            const auto& draw_pass = mgr.get_draw_passes()[i];
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
            u32 count = static_cast<u32>(mgr.get_draw_info_buf()->size() / sizeof(GPUDrawInfo));
            CullObjectPushConstants pc{
                left,
                right,
                bottom,
                top,
                near,
                far,
                device_->get_buffer(curr_frame().scene_uniform_buf)->device_addr(),
                count,
                mgr.get_draw_info_buf()->resource_info_->handle,
                draw_pass.get_frame_out_draw_cmd_buf(false)->resource_info_->handle,
                draw_pass.get_frame_out_draw_cmd_buf(true)->resource_info_->handle,
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
    csm_->debug_shadow_pass(rg, linear_sampler_);
  }

  if (!deferred_enabled_) {
    auto& forward = rg.add_pass("forward");
    auto rg_final_out_handle =
        forward.add("draw_out", {.format = draw_img_format_}, Access::ColorWrite);

    for (int double_sided = 0; double_sided < 2; double_sided++) {
      for (int pass_i = 0; pass_i < MeshPass_Count; pass_i++) {
        auto& mgr = static_draw_mgrs_[pass_i];
        if (mgr.should_draw()) {
          forward.add(mgr.get_draw_pass(main_view_mesh_pass_indices_[pass_i])
                          .get_frame_out_draw_cmd_buf_handle(double_sided),
                      Access::IndirectRead);
        }
      }
    }
    // for (const auto& mgr : static_draw_mgrs_) {
    //   if (mgr.should_draw()) {
    //     forward.add(
    //         mgr->get_draw_pass()[main_mesh_pass_idx_].get_frame_out_draw_cmd_buf_handle(false),
    //         Access::IndirectRead);
    //   }
    // }
    if (csm_enabled.get()) {
      forward.add(csm_->get_shadow_data_buffer(device_->curr_frame_num()), Access::FragmentRead);
      forward.add("shadow_map_img", csm_->get_shadow_map_att_info(), Access::FragmentRead);
    }
    auto depth_out_handle =
        forward.add("depth", {.format = depth_img_format_}, Access::DepthStencilWrite);
    forward.set_execute_fn([&rg, rg_final_out_handle, this, depth_out_handle](CmdEncoder& cmd) {
      auto tex_handle = rg.get_texture_handle(rg_final_out_handle);
      auto dims = device_->get_image(tex_handle)->size();
      cmd.begin_rendering({.extent = dims},
                          {{RenderingAttachmentInfo::color_att(tex_handle, LoadOp::Clear)},
                           {RenderingAttachmentInfo::color_att(
                               rg.get_texture_handle(depth_out_handle), LoadOp::Clear)}});
      cmd.set_viewport_and_scissor(dims);
      cmd.set_cull_mode(CullMode::None);

      BasicPushConstants pc{
          device_->get_buffer(curr_frame().scene_uniform_buf)->resource_info_->handle,
          static_vertex_buf_.get_buffer()->resource_info_->handle,
          static_instance_data_buf_.get_buffer()->resource_info_->handle,
          static_object_data_buf_.get_buffer()->resource_info_->handle,
          static_materials_buf_.get_buffer()->resource_info_->handle,
          device_->get_bindless_idx(linear_sampler_),
          device_->get_buffer(csm_->get_shadow_data_buffer(device_->curr_frame_in_flight()))
              ->resource_info_->handle,
          device_->get_bindless_idx(shadow_sampler_),
          device_->get_bindless_idx(csm_->get_shadow_map_img(), SubresourceType::Shader),
          device_->get_bindless_idx(ibl_->irradiance_cubemap_tex_.handle, SubresourceType::Shader),
          device_->get_bindless_idx(ibl_->brdf_lut_.handle, SubresourceType::Shader),
          device_->get_bindless_idx(ibl_->prefiltered_env_map_tex_.handle, SubresourceType::Shader),
          device_->get_bindless_idx(linear_sampler_clamp_to_edge_)};
      cmd.push_constants(sizeof(pc), &pc);
      // TODO: double sided
      cmd.bind_pipeline(PipelineBindPoint::Graphics, draw_pipeline_);
      for (int pass_i = 0; pass_i < MeshPass_Count; pass_i++) {
        execute_static_geo_draws(cmd, false, static_cast<MeshPass>(pass_i));
      }
      draw_skybox(cmd);
      cmd.end_rendering();
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

      for (int double_sided = 0; double_sided < 2; double_sided++) {
        for (u32 pass_i : opaque_mesh_pass_idxs_) {
          auto& mgr = static_draw_mgrs_[pass_i];
          if (mgr.should_draw()) {
            gbuffer.add(mgr.get_draw_pass(main_view_mesh_pass_indices_[pass_i])
                            .get_frame_out_draw_cmd_buf_handle(double_sided),
                        Access::IndirectRead);
          }
        }
      }
      auto rg_depth_handle =
          gbuffer.add("depth", {.format = depth_img_format_}, Access::DepthStencilWrite);
      gbuffer.set_execute_fn(
          [&rg, rg_gbuffer_a, rg_gbuffer_b, rg_gbuffer_c, this, rg_depth_handle](CmdEncoder& cmd) {
            auto gbuffer_a = rg.get_texture_handle(rg_gbuffer_a);
            auto gbuffer_b = rg.get_texture_handle(rg_gbuffer_b);
            auto gbuffer_c = rg.get_texture_handle(rg_gbuffer_c);
            uvec2 extent = device_->get_image(gbuffer_a)->size();
            cmd.begin_rendering({.extent = extent},
                                {RenderingAttachmentInfo::color_att(gbuffer_a, LoadOp::Clear),
                                 RenderingAttachmentInfo::color_att(gbuffer_b, LoadOp::Clear),
                                 RenderingAttachmentInfo::color_att(gbuffer_c, LoadOp::Clear),
                                 RenderingAttachmentInfo::depth_stencil_att(
                                     rg.get_texture_handle(rg_depth_handle), LoadOp::Clear)});

            cmd.set_viewport_and_scissor(extent);

            cmd.bind_pipeline(PipelineBindPoint::Graphics, gbuffer_pipeline_);
            GBufferPushConstants pc{
                static_vertex_buf_.get_buffer()->device_addr(),
                device_->get_buffer(curr_frame().scene_uniform_buf)->device_addr(),
                static_instance_data_buf_.get_buffer()->device_addr(),
                static_object_data_buf_.get_buffer()->device_addr(),
                static_materials_buf_.get_buffer()->device_addr(),
                device_->get_bindless_idx(linear_sampler_),
            };
            cmd.push_constants(sizeof(pc), &pc);
            cmd.set_cull_mode(CullMode::Back);
            execute_static_geo_draws(cmd, false, MeshPass_Opaque);
            cmd.set_cull_mode(CullMode::None);
            execute_static_geo_draws(cmd, true, MeshPass_Opaque);
            if (gbuffer_alpha_mask_pipeline_ == gbuffer_pipeline_) {
              exit(1);
            }
            cmd.bind_pipeline(PipelineBindPoint::Graphics, gbuffer_alpha_mask_pipeline_);
            cmd.set_cull_mode(CullMode::Back);
            execute_static_geo_draws(cmd, false, MeshPass_OpaqueAlphaMask);
            cmd.set_cull_mode(CullMode::None);
            execute_static_geo_draws(cmd, true, MeshPass_OpaqueAlphaMask);
            cmd.end_rendering();
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
        shade.add_image_access("shadow_map_img", Access::FragmentRead);
        shade.add(csm_->get_shadow_data_buffer(device_->curr_frame_in_flight()),
                  Access::ComputeRead);
      }
      auto depth_handle = shade.add("depth", {.format = depth_img_format_}, Access::ComputeSample);
      auto final_out_handle =
          shade.add("draw_out", {.format = draw_img_format_}, Access::ComputeWrite);
      shade.set_execute_fn([&rg, rg_gbuffer_b, rg_gbuffer_c, rg_gbuffer_a, final_out_handle, this,
                            depth_handle](CmdEncoder& cmd) {
        auto gbuffer_a = rg.get_texture_handle(rg_gbuffer_a);
        auto gbuffer_b = rg.get_texture_handle(rg_gbuffer_b);
        auto gbuffer_c = rg.get_texture_handle(rg_gbuffer_c);
        auto depth_img = rg.get_texture_handle(depth_handle);
        auto shade_out_tex = rg.get_texture_handle(final_out_handle);
        DeferredShadePushConstants pc{
            glm::inverse(scene_uniform_cpu_data_.view_proj),
            device_->get_bindless_idx(gbuffer_a, SubresourceType::Storage),
            device_->get_bindless_idx(gbuffer_b, SubresourceType::Storage),
            device_->get_bindless_idx(gbuffer_c, SubresourceType::Storage),
            device_->get_bindless_idx(depth_img, SubresourceType::Shader),
            device_->get_bindless_idx(shade_out_tex, SubresourceType::Storage),
            device_->get_bindless_idx(nearest_sampler_),
            device_->get_buffer(curr_frame().scene_uniform_buf)->resource_info_->handle,
            device_->get_bindless_idx(csm_->get_shadow_map_img(), SubresourceType::Shader),
            device_->get_bindless_idx(shadow_sampler_),
            device_->get_buffer(csm_->get_shadow_data_buffer(device_->curr_frame_in_flight()))
                ->resource_info_->handle,
            device_->get_bindless_idx(ibl_->irradiance_cubemap_tex_.handle,
                                      SubresourceType::Shader),
            device_->get_bindless_idx(ibl_->brdf_lut_.handle, SubresourceType::Shader),
            device_->get_bindless_idx(ibl_->prefiltered_env_map_tex_.handle,
                                      SubresourceType::Shader),
            device_->get_bindless_idx(linear_sampler_clamp_to_edge_)};
        cmd.bind_pipeline(PipelineBindPoint::Compute, deferred_shade_pipeline_);
        cmd.push_constants(sizeof(pc), &pc);
        auto dims = device_->get_image(gbuffer_a)->size();
        cmd.dispatch((dims.x + 16) / 16, (dims.y + 16) / 16, 1);
      });
    }
    {
      auto& skybox = rg.add_pass("skybox");
      auto draw_tex = skybox.add("draw_out", {.format = draw_img_format_}, Access::ColorRW);
      auto depth_handle =
          skybox.add("depth", {.format = depth_img_format_}, Access::DepthStencilRead);

      skybox.set_execute_fn([this, &rg, draw_tex, depth_handle](CmdEncoder& cmd) {
        auto tex = rg.get_texture_handle(draw_tex);

        auto dims = device_->get_image(tex)->size();
        cmd.begin_rendering(
            {.extent = dims},
            {RenderingAttachmentInfo::color_att(tex),
             RenderingAttachmentInfo::depth_stencil_att(rg.get_texture_handle(depth_handle))});
        cmd.set_viewport_and_scissor(dims);

        draw_skybox(cmd);
        cmd.end_rendering();
      });
    }
  }

  {
    auto& pp = rg.add_pass("post_process");
    auto draw_out_handle = pp.add("draw_out", {.format = draw_img_format_}, Access::ComputeRead);
    auto final_out_handle =
        pp.add("final_out", AttachmentInfo{.format = device_->get_swapchain_info().format},
               Access::ComputeWrite);
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
        cmd.bind_pipeline(PipelineBindPoint::Compute, postprocess_pipeline_);
        auto post_processed_img = rg.get_texture_handle(final_out_handle);
        struct {
          u32 in_tex_idx, out_tex_idx, flags, tonemap_type;
        } pc{device_->get_bindless_idx(rg.get_texture_handle(draw_out_handle),
                                       SubresourceType::Storage),
             device_->get_bindless_idx(post_processed_img, SubresourceType::Storage),
             postprocess_flags, tonemap_type_};
        cmd.push_constants(sizeof(pc), &pc);
        auto dims = device_->get_image(post_processed_img)->size();
        cmd.dispatch((dims.x + 16) / 16, (dims.y + 16) / 16, 1);
      }
    });
  }

  if (line_draw_vertices_.size()) {
    Buffer* line_draw_buf = device_->get_buffer(curr_frame().line_draw_buf);
    assert(line_draw_buf);
    size_t required_size = line_draw_vertices_.size() * sizeof(LineVertex);
    if (required_size > line_draw_buf->size()) {
      curr_frame().line_draw_buf = device_->create_buffer_holder(
          BufferCreateInfo{.size = glm::max<u64>(line_draw_buf->size() * 2, required_size),
                           .usage = BufferUsage_Storage,
                           .flags = BufferCreateFlags_HostVisible});
      line_draw_buf = device_->get_buffer(curr_frame().line_draw_buf);
    }
    memcpy(line_draw_buf->mapped_data(), line_draw_vertices_.data(), required_size);
  }

  if (draw_imgui_) {
    auto& imgui_p = rg.add_pass("ui");
    RGResourceHandle csm_debug_img_handle;
    if (csm_enabled.get() && csm_->get_debug_render_enabled()) {
      csm_debug_img_handle = imgui_p.add_image_access("shadow_map_debug_img", Access::FragmentRead);
    }
    auto rg_color_handle = imgui_p.add_image_access("final_out", Access::ColorRW);
    auto rg_depth_handle = imgui_p.add_image_access("depth", Access::DepthStencilRead);
    imgui_p.set_execute_fn(
        [this, rg_color_handle, &rg, csm_debug_img_handle, rg_depth_handle](CmdEncoder& cmd) {
          if (csm_enabled.get() && csm_->get_debug_render_enabled()) {
            assert(csm_debug_img_handle.is_valid());
            csm_->imgui_pass(cmd, linear_sampler_, rg.get_texture_handle(csm_debug_img_handle));
          }
          auto color_handle = rg.get_texture_handle(rg_color_handle);
          auto depth_handle = rg.get_texture_handle(rg_depth_handle);
          if (line_draw_vertices_.size()) {
            uvec2 extent = device_->get_image(color_handle)->size();
            cmd.begin_rendering({.extent = extent},
                                {RenderingAttachmentInfo::color_att(color_handle),
                                 RenderingAttachmentInfo::depth_stencil_att(depth_handle)});
            cmd.bind_pipeline(PipelineBindPoint::Graphics, line_draw_pipeline_);
            Buffer* line_draw_buf = device_->get_buffer(curr_frame().line_draw_buf);
            LinesRenderPushConstants pc{
                .vtx = line_draw_buf->device_addr(),
                .scene_buffer = device_->get_buffer(curr_frame().scene_uniform_buf)->device_addr()};
            cmd.push_constants(sizeof(pc), &pc);
            cmd.draw(line_draw_vertices_.size(), 1);
            cmd.end_rendering();
          }

          cmd.begin_rendering({.extent = device_->get_image(color_handle)->size()},
                              {RenderingAttachmentInfo::color_att(color_handle)});
          device_->render_imgui(cmd);
          cmd.end_rendering();
        });
  }
}

void VkRender2::execute_draw(CmdEncoder& cmd, BufferHandle buffer, u32 draw_count) const {
  if (draw_count == 0) return;
  constexpr u32 draw_cmd_offset{sizeof(u32)};
  if (!device_->is_supported(DeviceFeature::DrawIndirectCount)) {
    cmd.draw_indexed_indirect(buffer, draw_cmd_offset, draw_count,
                              sizeof(VkDrawIndexedIndirectCommand));
  } else {
    cmd.draw_indexed_indirect_count(buffer, draw_cmd_offset, buffer, 0, draw_count,
                                    sizeof(VkDrawIndexedIndirectCommand));
  }
}

void VkRender2::StaticMeshDrawManager::remove_draws(StateTracker&, CmdEncoder& cmd, Handle handle) {
  if (!handle.is_valid()) {
    return;
  }
  Alloc* a = allocs_.get(handle);
  assert(a);
  if (!a) return;
  for (int double_sided = 0; double_sided < 2; double_sided++) {
    u32 num_double_sided_draws = a->num_double_sided_draws;
    u32 num_draws = a->draw_cmd_slot.get_size() / sizeof(GPUDrawInfo);
    if (double_sided) {
      num_draw_cmds_[double_sided] -= num_double_sided_draws;
    } else {
      num_draw_cmds_[double_sided] -= num_draws - num_double_sided_draws;
    }
  }
  cmd.fill_buffer(draw_cmds_buf_.buffer.handle, a->draw_cmd_slot.get_offset(),
                  a->draw_cmd_slot.get_size(), 0);
  draw_cmds_buf_.allocator.free(std::move(a->draw_cmd_slot));

  allocs_.destroy(handle);
}

VkRender2::StaticMeshDrawManager::Handle VkRender2::StaticMeshDrawManager::add_draws(
    StateTracker& state, Device::CopyAllocator::CopyCmd& cmd, size_t size, size_t staging_offset,
    u32 num_double_sided_draws) {
  assert(size > 0);
  Alloc a;
  a.num_double_sided_draws = num_double_sided_draws;
  a.draw_cmd_slot = draw_cmds_buf_.allocator.allocate(size);
  u32 num_single_sided_draws = (size / sizeof(GPUDrawInfo)) - num_double_sided_draws;
  num_draw_cmds_[0] += num_single_sided_draws;
  num_draw_cmds_[1] += a.num_double_sided_draws;

  // resize draw cmd bufs
  size_t curr_tot_draw_cmd_buf_size = device_->get_buffer(draw_cmds_buf_.buffer)->size();
  auto new_size = glm::max<size_t>(curr_tot_draw_cmd_buf_size * 2,
                                   a.draw_cmd_slot.get_offset() + a.draw_cmd_slot.get_size());
  for (auto& draw_pass : draw_passes_) {
    // resize output draw cmd buffers
    for (int double_sided = 0; double_sided < 2; double_sided++) {
      for (auto& handle : draw_pass.out_draw_cmds_bufs[double_sided]) {
        u32 required_size =
            sizeof(u32) + (num_draw_cmds_[double_sided] * sizeof(VkDrawIndexedIndirectCommand));
        auto* buf = device_->get_buffer(handle);
        if (required_size > buf->size()) {
          handle = device_->create_buffer_holder(BufferCreateInfo{
              .size = new_size + sizeof(u32),
              .usage = BufferUsage_Indirect | BufferUsage_Storage,
          });
        }
      }
    }
  }

  if (a.draw_cmd_slot.get_offset() + a.draw_cmd_slot.get_size() >= curr_tot_draw_cmd_buf_size) {
    // draw cmd buf resize and copy
    auto new_buf = device_->create_buffer_holder(BufferCreateInfo{
        .size = new_size,
        .usage = BufferUsage_Storage,
    });

    cmd.copy_buffer(device_, *device_->get_buffer(new_buf), 0, 0, curr_tot_draw_cmd_buf_size);
    state
        .buffer_barrier(device_->get_buffer(new_buf)->buffer(), VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                        VK_ACCESS_2_TRANSFER_WRITE_BIT)
        .flush_barriers();
    draw_cmds_buf_.buffer = std::move(new_buf);
  }

  if (a.draw_cmd_slot.get_offset() >= device_->get_buffer(draw_cmds_buf_.buffer)->size()) {
    LINFO("unimplemented: need to resize Static mesh draw cmd buffer");
    exit(1);
  }
  auto* draw_cmds_buf = device_->get_buffer(draw_cmds_buf_.buffer);
  state
      .buffer_barrier(draw_cmds_buf->buffer(), VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                      VK_ACCESS_2_TRANSFER_WRITE_BIT)
      .flush_barriers();
  cmd.copy_buffer(device_, *draw_cmds_buf, staging_offset, a.draw_cmd_slot.get_offset(), size);
  state
      .buffer_barrier(
          draw_cmds_buf->buffer(),
          VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
          VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_2_SHADER_READ_BIT)
      .flush_barriers();

  return allocs_.alloc(std::move(a));
}

Buffer* VkRender2::StaticMeshDrawManager::get_draw_info_buf() const {
  return draw_cmds_buf_.get_buffer();
}

BufferHandle VkRender2::StaticMeshDrawManager::get_draw_info_buf_handle() const {
  return draw_cmds_buf_.buffer.handle;
}

void VkRender2::execute_static_geo_draws(CmdEncoder& cmd, bool double_sided, MeshPass pass) {
  cmd.bind_index_buffer(static_index_buf_.buffer.handle);
  auto& mgr = static_draw_mgrs_[pass];
  if (mgr.should_draw() && mgr.get_num_draw_cmds(double_sided) > 0) {
    execute_draw(cmd,
                 mgr.get_draw_pass(main_view_mesh_pass_indices_[pass])
                     .get_frame_out_draw_cmd_buf_handle(double_sided),
                 mgr.get_num_draw_cmds(double_sided));
  }
}

VkRender2::StaticMeshDrawManager::DrawPass::DrawPass(u32 num_single_sided_draws,
                                                     u32 num_double_sided_draws,
                                                     u32 frames_in_flight, Device* device)
    : device_(device) {
  for (u32 i = 0; i < frames_in_flight; i++) {
    for (int double_sided = 0; double_sided < 2; double_sided++) {
      u32 count = double_sided ? num_double_sided_draws : num_single_sided_draws;
      out_draw_cmds_bufs[double_sided].emplace_back(device_->create_buffer_holder(BufferCreateInfo{
          .size = (count * sizeof(VkDrawIndexedIndirectCommand)) + sizeof(u32),
          .usage = BufferUsage_Indirect | BufferUsage_Storage,
      }));
    }
  }
}

u32 VkRender2::StaticMeshDrawManager::add_draw_pass() {
  auto idx = draw_passes_.size();
  draw_passes_.emplace_back(num_draw_cmds_[0], num_draw_cmds_[1], device_->get_frames_in_flight(),
                            device_);
  return idx;
}

BufferHandle VkRender2::StaticMeshDrawManager::DrawPass::get_frame_out_draw_cmd_buf_handle(
    bool double_sided) const {
  return out_draw_cmds_bufs[double_sided][VkRender2::get().curr_frame_in_flight_num()].handle;
}

Buffer* VkRender2::StaticMeshDrawManager::DrawPass::get_frame_out_draw_cmd_buf(
    bool double_sided) const {
  return device_->get_buffer(
      out_draw_cmds_bufs[double_sided][VkRender2::get().curr_frame_in_flight_num()]);
}

void VkRender2::draw_line(const vec3& p1, const vec3& p2, const vec4& color) {
  line_draw_vertices_.emplace_back(vec4{p1, 0.}, color);
  line_draw_vertices_.emplace_back(vec4{p2, 0.}, color);
}

void VkRender2::draw_skybox(CmdEncoder& cmd) {
  cmd.bind_pipeline(PipelineBindPoint::Graphics, skybox_pipeline_);
  u32 skybox_handle{};
  if (render_prefilter_mip_skybox_) {
    skybox_handle = device_->get_bindless_idx(
        ibl_->prefiltered_env_map_tex_.handle, SubresourceType::Shader,
        ibl_->prefiltered_env_map_tex_views_[prefilter_mip_skybox_render_mip_level_]);
  } else {
    skybox_handle =
        device_->get_bindless_idx(ibl_->env_cubemap_tex_.handle, SubresourceType::Shader);
  }
  struct {
    u32 scene_buffer, tex_idx;
  } pc{
      device_->get_buffer(curr_frame().scene_uniform_buf)->resource_info_->handle,
      skybox_handle,
  };
  cmd.push_constants(sizeof(pc), &pc);
  cmd.set_cull_mode(CullMode::None);
  cmd.draw(36);
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
  for (auto& mgr : static_draw_mgrs_) {
    mgr.remove_draws(state_, cmd, instance.mesh_pass_draw_handles[mgr.get_mesh_pass()]);
  }
  free(instance);
}

uvec2 VkRender2::window_dims() const {
  int x, y;
  glfwGetFramebufferSize(window_, &x, &y);
  return {x, y};
}
float VkRender2::aspect_ratio() const {
  auto dims = window_dims();
  return (float)dims.x / (float)dims.y;
}

PipelineTask VkRender2::make_pipeline_task(const GraphicsPipelineCreateInfo& info,
                                           PipelineHandle* out_handle) {
  return {threads::pool.submit_task([=]() { *out_handle = PipelineManager::get().load(info); })};
}

PipelineTask VkRender2::make_pipeline_task(const ComputePipelineCreateInfo& info,
                                           PipelineHandle* out_handle) {
  return {threads::pool.submit_task([=]() { *out_handle = PipelineManager::get().load(info); })};
}

void VkRender2::StaticMeshDrawManager::init(MeshPass type, size_t initial_max_draw_cnt,
                                            Device* device) {
  device_ = device;
  mesh_pass_ = type;
  draw_cmds_buf_.buffer = device_->create_buffer_holder(BufferCreateInfo{
      .size = initial_max_draw_cnt * sizeof(GPUDrawInfo),
      .usage = BufferUsage_Storage,
  });
  draw_cmds_buf_.allocator.init(initial_max_draw_cnt * sizeof(GPUDrawInfo), sizeof(GPUDrawInfo),
                                100);
}
u64 VkRender2::curr_frame_in_flight_num() const {
  return device_->curr_frame_num() % get_device().get_frames_in_flight();
}
VkRender2::PerFrameData& VkRender2::curr_frame() {
  return per_frame_data_[device_->curr_frame_num() % device_->get_frames_in_flight()];
}

Buffer* FreeListBuffer::get_buffer() const { return get_device().get_buffer(buffer); }

void VkRender2::handle_load_scene_requests() {
  for (auto& load_scene_req : load_scene_reqs_) {
    auto& res = load_scene_req.result;

    u64 material_data_size = res.materials.size() * sizeof(gfx::Material);
    u64 vertices_size = res.vertices.size() * sizeof(gfx::Vertex);
    u64 indices_size = res.indices.size() * sizeof(u32);
    auto vertices_gpu_slot = static_vertex_buf_.allocator.allocate(vertices_size);
    auto indices_gpu_slot = static_index_buf_.allocator.allocate(indices_size);

    auto copy_cmd = device_->graphics_copy_allocator_.allocate(material_data_size + vertices_size +
                                                               indices_size);
    auto staging = LinearCopyer{device_->get_buffer(copy_cmd.staging_buffer)->mapped_data()};
    u64 material_data_staging_offset = staging.copy(res.materials.data(), material_data_size);
    u64 vertices_staging_offset = staging.copy(res.vertices.data(), vertices_size);
    u64 indices_staging_offset = staging.copy(res.indices.data(), indices_size);

    static_draw_stats_.vertices += res.vertices.size();
    static_draw_stats_.indices += res.indices.size();
    static_draw_stats_.textures += res.textures.size();
    static_draw_stats_.materials += res.materials.size();
    auto materials_gpu_slot = static_materials_buf_.allocator.allocate(material_data_size);
    {
      state_.reset(copy_cmd.transfer_cmd_buf);
      state_
          .buffer_barrier(static_vertex_buf_.get_buffer()->buffer(),
                          VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT)
          .buffer_barrier(static_index_buf_.get_buffer()->buffer(),
                          VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT)
          .buffer_barrier(static_materials_buf_.get_buffer()->buffer(),
                          VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT)
          .flush_barriers();
      copy_cmd.copy_buffer(device_, *device_->get_buffer(static_materials_buf_.buffer),
                           material_data_staging_offset, materials_gpu_slot.get_offset(),
                           material_data_size);
      copy_cmd.copy_buffer(device_, *static_vertex_buf_.get_buffer(), vertices_staging_offset,
                           vertices_gpu_slot.get_offset(), vertices_size);
      copy_cmd.copy_buffer(device_, *static_index_buf_.get_buffer(), indices_staging_offset,
                           indices_gpu_slot.get_offset(), indices_size);
      state_
          .buffer_barrier(static_vertex_buf_.get_buffer()->buffer(),
                          VK_PIPELINE_STAGE_2_VERTEX_INPUT_BIT,
                          VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT)
          .buffer_barrier(static_index_buf_.get_buffer()->buffer(),
                          VK_PIPELINE_STAGE_2_INDEX_INPUT_BIT, VK_ACCESS_2_INDEX_READ_BIT)
          .buffer_barrier(static_materials_buf_.get_buffer()->buffer(),
                          VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT)
          .flush_barriers();
      device_->graphics_copy_allocator_.submit(copy_cmd);
    }
    auto resources_handle = static_models_pool_.alloc(
        std::move(res.scene_graph_data), std::move(res.mesh_draw_infos),
        std::move(materials_gpu_slot), std::move(vertices_gpu_slot), std::move(indices_gpu_slot),
        std::move(res.textures), std::move(res.materials),
        vertices_gpu_slot.get_offset() / sizeof(gfx::Vertex),
        indices_gpu_slot.get_offset() / sizeof(u32), res.vertices.size(), res.indices.size(),
        load_scene_req.path.string(), 0u);
    auto* resources = static_models_pool_.get(resources_handle);
    static_model_name_to_handle_.emplace(resources->name, resources_handle);
    load_instance_reqs_.emplace_back(resources_handle, load_scene_req.transform);
  }
  load_scene_reqs_.clear();

  for (auto& load_instance_req : load_instance_reqs_) {
    auto* resources = static_models_pool_.get(load_instance_req.resource_handle);
    auto& resources_handle = load_instance_req.resource_handle;
    auto& transform = load_instance_req.transform;
    static_draw_stats_.total_vertices += resources->num_vertices;
    static_draw_stats_.total_indices += resources->num_indices;

    u32 pass_double_sided_draw_cnts[MeshPass_Count] = {};
    u32 pass_obj_counts[MeshPass_Count] = {};

    const auto& scene = resources->scene_graph_data;
    for (size_t node_i = 0; node_i < scene.hierarchies.size(); node_i++) {
      auto it = scene.node_to_mesh_data.find(node_i);
      if (it == scene.node_to_mesh_data.end()) continue;
      const auto& mesh_indices = it->second;
      bool double_sided = resources->materials[mesh_indices.material_id].is_double_sided();
      if (mesh_indices.pass_flags & PassFlags_Opaque) {
        if (double_sided) {
          pass_double_sided_draw_cnts[MeshPass_Opaque]++;
        }
        pass_obj_counts[MeshPass_Opaque]++;
      } else if (mesh_indices.pass_flags & PassFlags_OpaqueAlpha) {
        if (double_sided) {
          pass_double_sided_draw_cnts[MeshPass_OpaqueAlphaMask]++;
        }
        pass_obj_counts[MeshPass_OpaqueAlphaMask]++;
      } else if (mesh_indices.pass_flags & PassFlags_Transparent) {
        if (double_sided) {
          pass_double_sided_draw_cnts[MeshPass_Transparent]++;
        }
        pass_obj_counts[MeshPass_Transparent]++;
      }
    }
    resources->ref_count++;

    u32 num_objs_tot{};
    for (auto pass_obj_count : pass_obj_counts) {
      num_objs_tot += pass_obj_count;
    }

    auto model_instance_resources_handle = static_model_instance_pool_.alloc();
    loaded_model_instance_resources_.emplace_back(model_instance_resources_handle);
    auto* instance_resources = static_model_instance_pool_.get(model_instance_resources_handle);
    instance_resources->model_resources_handle = resources_handle;
    instance_resources->name = resources->name.c_str();

    std::vector<GPUInstanceData> instance_datas;
    instance_resources->object_datas.reserve(num_objs_tot);
    instance_datas.reserve(num_objs_tot);

    vec3 scene_min = vec3{std::numeric_limits<float>::max()};
    vec3 scene_max = vec3{std::numeric_limits<float>::lowest()};
    // TODO: no allocate
    std::array<std::vector<GPUDrawInfo>, MeshPass_Count> pass_cmds;
    // for (auto& c : pass_cmds) {
    //
    // }
    // opaque_cmds.reserve(num_opaque_objs);
    // alpha_mask_cmds.reserve(num_opaque_alpha_mask_objs);
    // transparent_cmds.reserve(num_transparent_objs);
    instance_resources->instance_data_slot =
        static_instance_data_buf_.allocator.allocate(num_objs_tot * sizeof(GPUInstanceData));
    instance_resources->object_data_slot =
        static_object_data_buf_.allocator.allocate(num_objs_tot * sizeof(ObjectData));

    u32 base_instance_id =
        instance_resources->instance_data_slot.get_offset() / sizeof(GPUInstanceData);
    u32 base_object_data_id =
        instance_resources->object_data_slot.get_offset() / sizeof(ObjectData);
    auto base_material_id = resources->materials_slot.get_offset() / sizeof(Material);

    bool is_non_identity_root_node_transform = transform != mat4{1};

    for (size_t node_i = 0; node_i < scene.hierarchies.size(); node_i++) {
      auto it = scene.node_to_mesh_data.find(node_i);
      if (it == scene.node_to_mesh_data.end()) continue;
      const auto& node_mesh_data = it->second;
      // auto& node_mesh_data
      auto& mesh = resources->mesh_draw_infos[node_mesh_data.mesh_idx];
      const mat4 model = is_non_identity_root_node_transform
                             ? scene.global_transforms[node_i] * transform
                             : scene.global_transforms[node_i];
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

      u32 draw_flags{};
      if (resources->materials[node_mesh_data.material_id].is_double_sided()) {
        draw_flags |= GPUDrawInfoFlags_DoubleSided;
      }

      GPUDrawInfo draw{
          .index_cnt = mesh.index_count,
          .first_index = static_cast<u32>(resources->first_index + mesh.first_index),
          .vertex_offset = static_cast<u32>(resources->first_vertex + mesh.first_vertex),
          .instance_id = instance_id,
          .flags = draw_flags};

      if (node_mesh_data.pass_flags & PassFlags_Opaque) {
        pass_cmds[MeshPass_Opaque].emplace_back(draw);
      } else if (node_mesh_data.pass_flags & PassFlags_OpaqueAlpha) {
        pass_cmds[MeshPass_OpaqueAlphaMask].emplace_back(draw);
      } else if (node_mesh_data.pass_flags & PassFlags_Transparent) {
        pass_cmds[MeshPass_Transparent].emplace_back(draw);
      }
    }

    u64 obj_datas_size = instance_resources->object_datas.size() * sizeof(gfx::ObjectData);
    u64 instance_datas_size = instance_datas.size() * sizeof(GPUInstanceData);

    scene_aabb_.min = glm::min(scene_aabb_.min, scene_min);
    scene_aabb_.max = glm::max(scene_aabb_.max, scene_max);

    u32 tot_draw_cmds_buf_size = 0;
    for (auto& cmds : pass_cmds) {
      tot_draw_cmds_buf_size += cmds.size() * sizeof(GPUDrawInfo);
    }
    auto copy_cmd = device_->graphics_copy_allocator_.allocate(
        tot_draw_cmds_buf_size + obj_datas_size + instance_datas_size);
    auto staging = LinearCopyer{device_->get_buffer(copy_cmd.staging_buffer)->mapped_data()};
    u64 cmds_staging_offsets[MeshPass_Count] = {};
    for (int i = 0; i < MeshPass_Count; i++) {
      auto& cmds = pass_cmds[i];
      if (cmds.size()) {
        cmds_staging_offsets[i] = staging.copy(cmds.data(), cmds.size() * sizeof(GPUDrawInfo));
      }
    }
    u64 obj_datas_staging_offset =
        staging.copy(instance_resources->object_datas.data(), obj_datas_size);
    u64 instance_datas_staging_offset = staging.copy(instance_datas.data(), instance_datas_size);
    {
      assert(obj_datas_size && instance_datas_size);
      state_.reset(copy_cmd.transfer_cmd_buf);
      for (int i = 0; i < MeshPass_Count; i++) {
        auto& cmds = pass_cmds[i];
        if (cmds.size()) {
          instance_resources->mesh_pass_draw_handles[i] = static_draw_mgrs_[i].add_draws(
              state_, copy_cmd, cmds.size() * sizeof(GPUDrawInfo), cmds_staging_offsets[i],
              pass_double_sided_draw_cnts[i]);
        }
      }
      copy_cmd.copy_buffer(device_, *static_object_data_buf_.get_buffer(), obj_datas_staging_offset,
                           instance_resources->object_data_slot.get_offset(), obj_datas_size);
      copy_cmd.copy_buffer(
          device_, *static_instance_data_buf_.get_buffer(), instance_datas_staging_offset,
          instance_resources->instance_data_slot.get_offset(), instance_datas_size);
      state_
          .buffer_barrier(
              static_object_data_buf_.get_buffer()->buffer(),
              VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT,
              VK_ACCESS_2_SHADER_READ_BIT)
          .buffer_barrier(static_instance_data_buf_.get_buffer()->buffer(),
                          VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT)
          .flush_barriers();
      device_->graphics_copy_allocator_.submit(copy_cmd);
    }
  }
  load_instance_reqs_.clear();
}

}  // namespace gfx
