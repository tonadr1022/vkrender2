#include "VkRender2.hpp"

// clang-format off
#include <volk.h>
#include <vulkan/vk_enum_string_helper.h>
#include <vulkan/vulkan_core.h>
// clang-format on

#include <algorithm>
#include <cassert>
#include <filesystem>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/string_cast.hpp>
#include <glm/mat4x4.hpp>
#include <memory>
#include <print>
#include <tracy/Tracy.hpp>
#include <tracy/TracyVulkan.hpp>
#include <utility>

#include "CommandEncoder.hpp"
#include "GLFW/glfw3.h"
#include "RenderGraph.hpp"
#include "ResourceManager.hpp"
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
#include "shaders/oit/transparent_common.h.glsl"
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

struct TransparentFragmentData {
  u64 rgba;
  float depth;
  u32 next;
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

  // create per frame staging buffers
  for (u32 i = 0; i < device_->get_frames_in_flight(); i++) {
    // 64 MB
    staging_buffers_.emplace_back(device_->create_staging_buffer(1024ul * 1024 * 64));
    staging_buffer_copiers_.emplace_back(
        device_->get_buffer(staging_buffers_.back())->mapped_data());
  }

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
  auto animated_vertices_size = 1'000'000 * sizeof(gfx::AnimatedVertex);
  static_vertex_buf_.buffer = device_->create_buffer_holder(
      {BufferCreateInfo{.size = vertices_size, .usage = BufferUsage_Storage}});
  static_vertex_buf_.allocator.init(vertices_size, sizeof(Vertex));
  animated_vertex_buf_.buffer = device_->create_buffer_holder(
      {BufferCreateInfo{.size = animated_vertices_size, .usage = BufferUsage_Storage}});
  animated_vertex_buf_.allocator.init(animated_vertices_size, sizeof(gfx::AnimatedVertex));
  animated_vertex_output_bufs_.allocator.init(animated_vertices_size, sizeof(Vertex));
  for (size_t i = 0; i < device_->get_frames_in_flight(); i++) {
    animated_vertex_output_bufs_.buffers[i] = device_->create_buffer_holder(
        BufferCreateInfo{.size = animated_vertices_size, .usage = BufferUsage_Storage});
    // TODO: lol
    bone_matrix_bufs_.emplace_back();
  }
  global_skin_mat_allocator_.init(100'000, sizeof(mat4));
  skin_instance_datas_.allocator.init(animated_vertices_size, sizeof(SkinCommand));
  skin_instance_datas_.buffer = device_->create_buffer_holder(
      BufferCreateInfo{.size = animated_vertices_size, .usage = BufferUsage_Storage});

  auto indices_size = 10'000'000 * sizeof(u32);
  static_index_buf_.buffer = device_->create_buffer_holder(BufferCreateInfo{
      .size = indices_size,
      .usage = BufferUsage_Index,
  });
  static_index_buf_.allocator.init(indices_size, sizeof(u32));

  u64 max_static_draws = 100'000;
  u64 max_animated_draws = 10'0000;

  // TODO: refactor
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

  animated_instance_data_buf_.buffer = device_->create_buffer_holder(BufferCreateInfo{
      .size = max_animated_draws * sizeof(GPUInstanceData),
      .usage = BufferUsage_Storage,
  });
  animated_instance_data_buf_.allocator.init(max_animated_draws * sizeof(GPUInstanceData),
                                             sizeof(GPUInstanceData), 100);

  animated_object_data_buf_.buffer = device_->create_buffer_holder(BufferCreateInfo{
      .size = max_animated_draws * sizeof(ObjectData),
      .usage = BufferUsage_Storage,
  });
  animated_object_data_buf_.allocator.init(max_static_draws * sizeof(ObjectData),
                                           sizeof(ObjectData), 100);
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
      .add_graphics(
          GraphicsPipelineCreateInfo{
              .shaders = {{"fullscreen_quad.vert", ShaderType::Vertex},
                          {"postprocess/postprocess.frag", ShaderType::Fragment}},
              .rendering = {.color_formats = {convert_format(
                                device_->get_swapchain_info().format)}},
              .rasterization = {.cull_mode = CullMode::Front},
              .depth_stencil = GraphicsPipelineCreateInfo::depth_disable(),
              .name = "depth debug",
          },
          &postprocess_pipeline_)
      // .add_compute("postprocess/postprocess.comp", &postprocess_pipeline_)
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
          &line_draw_pipeline_)
      .add_graphics(
          GraphicsPipelineCreateInfo{
              .shaders = {{"oit/transparent.vert", ShaderType::Vertex},
                          {"oit/transparent.frag", ShaderType::Fragment}},
              .rendering = {.color_formats = {}, .depth_format = convert_format(depth_img_format_)},
              .depth_stencil =
                  GraphicsPipelineCreateInfo::depth_enable(false, CompareOp::GreaterOrEqual),
              .name = "transparent_oit"},
          &transparent_oit_pipeline_)
      .add_compute("animation/skinning.comp", &skinning_comp_pipeline_)
      .add_compute("oit/oit.comp", &oit_comp_pipeline_);

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
        std::array<MeshPass, 2> mesh_passes;
        if (opaque_alpha) {
          mesh_passes = {MeshPass_OpaqueAlphaMaskDoubleSided, MeshPass_OpaqueAlphaMask};
        } else {
          mesh_passes = {MeshPass_Opaque, MeshPass_OpaqueDoubleSided};
        }
        for (auto pass : mesh_passes) {
          const StaticMeshDrawManager& mgr = static_draw_mgrs_[pass];
          if (!mgr.should_draw()) return;
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
          cmd.set_cull_mode(get_cull_mode(mgr.get_mesh_pass()));
          execute_draw(cmd,
                       mgr.get_draw_pass(shadow_mesh_pass_indices_[cascade_i][pass])
                           .get_frame_out_draw_cmd_buf_handle(),
                       mgr.get_num_draw_cmds());
        }
      },
      [this](RenderGraphPass& pass) {
        MeshPass opaque_passes[] = {MeshPass_Opaque, MeshPass_OpaqueDoubleSided,
                                    MeshPass_OpaqueAlphaMask, MeshPass_OpaqueAlphaMaskDoubleSided};
        for (size_t draw_pass_i = 0; draw_pass_i < COUNTOF(opaque_passes); draw_pass_i++) {
          for (int animated = 0; animated < 2; animated++) {
            auto& mgr = get_mgr((MeshPass)draw_pass_i, animated);
            if (mgr.should_draw()) {
              for (u32 cascade_i = 0; cascade_i < csm_->get_num_cascade_levels(); cascade_i++) {
                pass.add(mgr.get_draw_pass(shadow_mesh_pass_indices_[cascade_i][draw_pass_i])
                             .get_frame_out_draw_cmd_buf_handle(),
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

  for (int animated = 0; animated < 2; animated++) {
    for (int i = 0; i < MeshPass_Count; i++) {
      auto& mgr = get_mgr((MeshPass)i, animated);
      mgr.init(static_cast<MeshPass>(i), 1000, device_,
               to_string((MeshPass)i) + (animated ? " animated" : ""));
      // TODO: split up, for now it works since both are parallel but this is terrible
      main_view_mesh_pass_indices_[i] = mgr.add_draw_pass();
    }
  }

  // main draw passes
  for (u32 i = 0; i < csm_->get_num_cascade_levels(); i++) {
    auto& state = shadow_mesh_pass_indices_.emplace_back();
    for (u32 pass_i : opaque_mesh_pass_idxs_) {
      for (int animated = 0; animated < 2; animated++) {
        state[pass_i] = get_mgr((MeshPass)pass_i, animated).add_draw_pass();
      }
    }
  }
  default_env_map_path_ = info.resource_dir / "hdr" / "newport_loft.hdr";
  oit_atomic_counter_buf_ = device_->create_buffer_holder(
      BufferCreateInfo{.size = sizeof(u32), .usage = BufferUsage_Storage});
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
    scene_uniform_cpu_data_.inverse_view_proj = glm::inverse(scene_uniform_cpu_data_.view_proj);
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

  staging_buffer_copiers_[device_->curr_frame_in_flight()].reset();

  {
    ZoneScopedN("copy dirty transforms");
    if (object_datas_to_copy_.size()) {
      auto copy_size = sizeof(ObjectData) * object_data_buffer_copier_.copies.size();
      immediate_submit([this, copy_size](CmdEncoder& cmd) {
        state_
            .buffer_barrier(static_object_data_buf_.get_buffer()->buffer(),
                            VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT)
            .flush_barriers();
        staging_buffer_copiers_[device_->curr_frame_in_flight()].copy(object_datas_to_copy_.data(),
                                                                      copy_size);
        cmd.copy_buffer(object_data_buffer_copier_,
                        *device_->get_buffer(staging_buffers_[device_->curr_frame_in_flight()]),
                        *static_object_data_buf_.get_buffer());
        state_
            .buffer_barrier(
                static_object_data_buf_.get_buffer()->buffer(),
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT,
                VK_ACCESS_2_SHADER_READ_BIT)
            .flush_barriers();
      });
      object_datas_to_copy_.clear();
      object_data_buffer_copier_.copies.clear();
    }
  }

  ResourceManager::get().update();

  // TODO: per instance data: need to get the right bone matrix offset for each model

  {
    // upload skin matrices
    if (global_skin_matrices_.size()) {
      auto copy_size = global_skin_matrices_.size() * sizeof(mat4);
      Buffer* curr_mat_buf =
          device_->get_buffer(bone_matrix_bufs_[device_->curr_frame_in_flight()]);
      if (!curr_mat_buf || copy_size > curr_mat_buf->size()) {
        bone_matrix_bufs_[device_->curr_frame_in_flight()] = device_->create_buffer_holder(
            BufferCreateInfo{.size = copy_size, .usage = BufferUsage_Storage});
        curr_mat_buf = device_->get_buffer(bone_matrix_bufs_[device_->curr_frame_in_flight()]);
      }
      auto copy_cmd = device_->graphics_copy_allocator_.allocate(copy_size);
      state_.reset(copy_cmd.transfer_cmd_buf);
      memcpy(device_->get_buffer(copy_cmd.staging_buffer)->mapped_data(),
             global_skin_matrices_.data(), copy_size);
      copy_cmd.copy_buffer(device_, *curr_mat_buf, 0, 0, copy_size);
      state_
          .buffer_barrier(curr_mat_buf->buffer(), VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                          VK_ACCESS_2_SHADER_READ_BIT)
          .flush_barriers();
      device_->graphics_copy_allocator_.submit(copy_cmd);
    }
  }

  if (draw_debug_aabbs_) {
    ZoneScopedN("debug aabbs");
    for (const auto& entry : static_model_instance_pool_.get_entries()) {
      if (!entry.live_) continue;
      for (const auto& obj_data : entry.object.object_datas) {
        draw_box(mat4{1}, AABB{obj_data.aabb_min, obj_data.aabb_max});
      }
    }
  }

  CmdEncoder* cmd = device_->begin_command_list(QueueType::Graphics);
  state_.reset(*cmd);
  device_->bind_bindless_descriptors(*cmd);

  for (auto& instance : to_delete_static_model_instances_) {
    free(*cmd, *static_model_instance_pool_.get(instance));
    static_model_instance_pool_.destroy(instance);
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
  get_device().acquire_next_image(cmd);

  {
    auto draw_img_dims = device_->get_swapchain_info().dims;
    // OIT
    u32 max_oit_fragments = draw_img_dims.x * draw_img_dims.y * 4;
    if (max_oit_fragments > max_oit_fragments_) {
      max_oit_fragments_ = max_oit_fragments;
      // create buffer
      oit_fragment_buffer_ = device_->create_buffer_holder(BufferCreateInfo{
          .size = max_oit_fragments_ * sizeof(TransparentFragmentData),
          .usage = BufferUsage_Storage,
      });
    }
    auto* oit_heads_tex = device_->get_image(oit_heads_tex_);
    if (!oit_heads_tex || oit_heads_tex->size().x != draw_img_dims.x ||
        oit_heads_tex->size().y != draw_img_dims.y) {
      oit_heads_tex_ = device_->create_image_holder(ImageDesc{
          .type = ImageDesc::Type::TwoD,
          .format = Format::R32Uint,
          .dims = draw_img_dims,
          .bind_flags = BindFlag::Storage,
      });
    }
  }

  add_rendering_passes(rg_);
  auto res = rg_.bake();
  if (!res) {
    LERROR("bake error {}", res.error());
    exit(1);
  }
  // rg_.print_pass_order();

  rg_.setup_attachments();
  rg_.execute(*cmd);

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
        ImGui::Text("Total vertices: %lu", (size_t)draw_stats_.total_vertices);
        ImGui::Text("Total indices: %lu", (size_t)draw_stats_.total_indices);
        ImGui::Text("Total triangles: %lu", (size_t)draw_stats_.total_vertices / 3);
        ImGui::Text("Total animated vertices: %lu", (size_t)draw_stats_.animated_vertices);
        ImGui::Text("Vertices %u", draw_stats_.vertices);
        ImGui::Text("Indices: %u", draw_stats_.indices);
        ImGui::Text("Materials: %u", draw_stats_.materials);
        ImGui::Text("Textures: %u", draw_stats_.textures);
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
      ImGui::Text("TODO LMAO");
      for (int mesh_pass = 0; mesh_pass < MeshPass_Count; mesh_pass++) {
        for (int animated = 0; animated < 2; animated++) {
          auto& mgr = get_mgr((MeshPass)mesh_pass, animated);
          ImGui::PushID(&mgr);
          if (ImGui::TreeNodeEx(mgr.get_name().c_str(), ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::Text("draws %u", mgr.get_num_draw_cmds());
            ImGui::Checkbox("Enabled", &mgr.draw_enabled);
            ImGui::TreePop();
          }
          ImGui::PopID();
        }
      }

      ImGui::TreePop();
    }

    if (ImGui::TreeNode("Transparency")) {
      ImGui::Checkbox("OIT enabled", &oit_enabled_);
      ImGui::Checkbox("OIT Debug Heatmap", &oit_debug_heatmap_);
      ImGui::SliderFloat("Opacity boost", &oit_opacity_boost_, 0., 1.);
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
  ImGui::End();
}

VkRender2::~VkRender2() {
  ZoneScoped;
  threads::pool.wait();
  device_->wait_idle();
  for (auto& frame : per_frame_data_) {
    frame.scene_uniform_buf = {};
    frame.line_draw_buf = {};
  }
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
  if (draw_stats_.animated_vertices > 0) {
    assert(global_skin_matrices_.size());
    assert(device_->get_buffer(bone_matrix_bufs_[device_->curr_frame_in_flight()]));
    auto& skinning = rg.add_pass("skinning");
    skinning.add(animated_vertex_output_bufs_.buffers[device_->curr_frame_in_flight()].handle,
                 Access::ComputeWrite);
    skinning.set_execute_fn([this](CmdEncoder& cmd) {
      cmd.bind_pipeline(PipelineBindPoint::Compute, skinning_comp_pipeline_);
      struct {
        u64 bone_mat_buf;
        u64 out_vertex_buf;
        u64 skinned_vertex_buf;
        u64 skin_cmd_buf;
        u64 cnt;
        // u64 instance_data_buf;
      } pc{
          device_->get_buffer(bone_matrix_bufs_[device_->curr_frame_in_flight()])->device_addr(),
          device_->get_buffer(animated_vertex_output_bufs_.buffers[device_->curr_frame_in_flight()])
              ->device_addr(),
          device_->get_buffer(animated_vertex_buf_.buffer)->device_addr(),
          device_->get_buffer(skin_instance_datas_.buffer)->device_addr(),
          draw_stats_.animated_vertices,
      };
      cmd.push_constants(sizeof(pc), &pc);
      // num vertices
      cmd.dispatch(draw_stats_.animated_vertices, 1, 1);
    });
  }
  {
    auto& clear_buff = rg.add_pass("clear_draw_cnt_buf");
    for (int animated = 0; animated < 2; animated++) {
      for (int i = 0; i < MeshPass_Count; i++) {
        auto& mgr = get_mgr((MeshPass)i, animated);
        if (mgr.should_draw()) {
          for (const auto& draw_pass : mgr.get_draw_passes()) {
            clear_buff.add(draw_pass.get_frame_out_draw_cmd_buf_handle(), Access::TransferWrite);
          }
        }
      }
    }

    clear_buff.set_execute_fn([this](CmdEncoder& cmd) {
      for (int animated = 0; animated < 2; animated++) {
        for (int i = 0; i < MeshPass_Count; i++) {
          auto& mgr = get_mgr((MeshPass)i, animated);
          if (!mgr.should_draw()) continue;
          for (const auto& draw_pass : mgr.get_draw_passes()) {
            auto buf = draw_pass.get_frame_out_draw_cmd_buf_handle();
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
    for (int animated = 0; animated < 2; animated++) {
      for (int i = 0; i < MeshPass_Count; i++) {
        auto& mgr = get_mgr((MeshPass)i, animated);
        if (mgr.should_draw()) {
          for (const auto& draw_pass : mgr.get_draw_passes()) {
            cull.add(draw_pass.get_frame_out_draw_cmd_buf_handle(), Access::ComputeRW);
          }
        }
      }
    }
    cull.set_execute_fn([this](CmdEncoder& cmd) {
      cmd.bind_pipeline(PipelineBindPoint::Compute, cull_objs_pipeline_);
      for (int animated = 0; animated < 2; animated++) {
        for (int i = 0; i < MeshPass_Count; i++) {
          auto& mgr = get_mgr((MeshPass)i, animated);
          if (mgr.should_draw()) {
            for (u32 i = 0; i < mgr.get_draw_passes().size(); i++) {
              assert(i < cull_vp_matrices_.size());
              if (i >= cull_vp_matrices_.size()) break;
              // extract frustum planes
              auto extract_planes_from_projmat = [](const mat4& mat, vec4& left, vec4& right,
                                                    vec4& bottom, vec4& top, vec4& near,
                                                    vec4& far) {
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
              // TODO: diff count here, actual draw cnt
              u32 count = static_cast<u32>(mgr.get_draw_info_buf()->size() / sizeof(GPUDrawInfo));
              const auto& draw_pass = mgr.get_draw_passes()[i];
              auto& obj_data_buf = static_object_data_buf_;
              // auto& obj_data_buf = animated ? animated_object_data_buf_ :
              // static_object_data_buf_;
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
                  draw_pass.get_frame_out_draw_cmd_buf()->resource_info_->handle,
                  obj_data_buf.get_buffer()->resource_info_->handle,
                  flags,
              };
              cmd.push_constants(sizeof(pc), &pc);
              cmd.dispatch((count + 256) / 256, 1, 1);
            }
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

    for (int pass_i = 0; pass_i < MeshPass_Count; pass_i++) {
      for (int animated = 0; animated < 2; animated++) {
        auto& mgr = get_mgr((MeshPass)pass_i, animated);
        if (mgr.should_draw()) {
          forward.add(mgr.get_draw_pass(main_view_mesh_pass_indices_[pass_i])
                          .get_frame_out_draw_cmd_buf_handle(),
                      Access::IndirectRead);
        }
      }
    }
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
        execute_static_geo_draws(cmd, static_cast<MeshPass>(pass_i));
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
      if (draw_stats_.animated_vertices > 0) {
        gbuffer.add(animated_vertex_output_bufs_.buffers[device_->curr_frame_in_flight()].handle,
                    (Access)(Access::VertexRead | Access::ComputeRead));
      }

      for (int animated = 0; animated < 2; animated++) {
        for (u32 pass_i : opaque_mesh_pass_idxs_) {
          auto& mgr = get_mgr((MeshPass)pass_i, animated);
          if (mgr.should_draw()) {
            gbuffer.add(mgr.get_draw_pass(main_view_mesh_pass_indices_[pass_i])
                            .get_frame_out_draw_cmd_buf_handle(),
                        Access::IndirectRead);
          }
        }
      }
      auto rg_depth_handle =
          gbuffer.add("depth", {.format = depth_img_format_}, Access::DepthStencilWrite);
      gbuffer.set_execute_fn([&rg, rg_gbuffer_a, rg_gbuffer_b, rg_gbuffer_c, this,
                              rg_depth_handle](CmdEncoder& cmd) {
        cmd.begin_region("gbuffer");
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

        // TODO: pipeline binding should be on the outside
        for (int animated = 0; animated < 2; animated++) {
          if (animated && draw_stats_.animated_vertices == 0) {
            continue;
          }
          GBufferPushConstants pc{
              animated
                  ? device_
                        ->get_buffer(
                            animated_vertex_output_bufs_.buffers[device_->curr_frame_in_flight()])
                        ->device_addr()
                  : static_vertex_buf_.get_buffer()->device_addr(),
              device_->get_buffer(curr_frame().scene_uniform_buf)->device_addr(),
              static_instance_data_buf_.get_buffer()->device_addr(),
              static_object_data_buf_.get_buffer()->device_addr(),
              static_materials_buf_.get_buffer()->device_addr(),
              device_->get_bindless_idx(linear_sampler_),
          };
          cmd.push_constants(sizeof(pc), &pc);
          cmd.bind_pipeline(PipelineBindPoint::Graphics, gbuffer_pipeline_);
          execute_static_geo_draws(cmd, MeshPass_Opaque, animated);
          execute_static_geo_draws(cmd, MeshPass_OpaqueDoubleSided, animated);
          if (gbuffer_alpha_mask_pipeline_ == gbuffer_pipeline_) {
            exit(1);
          }
          cmd.bind_pipeline(PipelineBindPoint::Graphics, gbuffer_alpha_mask_pipeline_);
          execute_static_geo_draws(cmd, MeshPass_OpaqueAlphaMask, animated);
          execute_static_geo_draws(cmd, MeshPass_OpaqueAlphaMaskDoubleSided, animated);
        }
        cmd.end_rendering();
        cmd.end_region();
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
      auto depth_handle = skybox.add_image_access("depth", (Access)(Access::DepthStencilRead));

      skybox.set_execute_fn([this, &rg, draw_tex, depth_handle](CmdEncoder& cmd) {
        cmd.begin_region("skybox");
        auto tex = rg.get_texture_handle(draw_tex);

        auto dims = device_->get_image(tex)->size();
        cmd.begin_rendering(
            {.extent = dims},
            {RenderingAttachmentInfo::color_att(tex),
             RenderingAttachmentInfo::depth_stencil_att(rg.get_texture_handle(depth_handle))});
        cmd.set_viewport_and_scissor(dims);

        draw_skybox(cmd);
        cmd.end_rendering();
        cmd.end_region();
      });
    }
  }

  bool oit_pass_needed = oit_enabled_;
  for (int animated = 0; animated < 2; animated++) {
    oit_pass_needed = oit_pass_needed ||
                      get_mgr(MeshPass_TransparentDoubleSided, animated).should_draw() ||
                      get_mgr(MeshPass_Transparent, animated).should_draw();
  }

  const char* post_process_input_img_name = "draw_out";
  if (oit_pass_needed) {
    post_process_input_img_name = "oit_out";
  }

  if (oit_pass_needed) {
    {
      auto& prepare_transparent_draw_pass = rg.add_pass("prepare_transparent_draw");
      prepare_transparent_draw_pass.add(oit_heads_tex_.handle, Access::TransferWrite);
      prepare_transparent_draw_pass.set_execute_fn([this](CmdEncoder& cmd) {
        VkClearColorValue clear_color{.uint32 = {0xFFFFFFFF}};
        VkImageSubresourceRange range{
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .levelCount = 1, .layerCount = 1};
        vkCmdClearColorImage(cmd.get_cmd_buf(), device_->get_image(oit_heads_tex_)->image(),
                             VK_IMAGE_LAYOUT_GENERAL, &clear_color, 1, &range);
      });
    }
    {
      auto& transparent_draw_pass = rg.add_pass("transparent_draw");
      // draw cmd inputs
      MeshPass passes[2] = {MeshPass_Transparent, MeshPass_TransparentDoubleSided};
      for (auto pass : passes) {
        for (int animated = 0; animated < 2; animated++) {
          auto& mgr = get_mgr(pass, animated);
          if (mgr.should_draw()) {
            transparent_draw_pass.add(mgr.get_draw_pass(main_view_mesh_pass_indices_[pass])
                                          .get_frame_out_draw_cmd_buf_handle(),
                                      Access::IndirectRead);
          }
        }
      }

      // reads depth buffer after opaque
      auto rg_depth_handle =
          transparent_draw_pass.add("depth", {.format = depth_img_format_},
                                    (Access)(Access::DepthStencilRead | Access::FragmentRead));
      transparent_draw_pass.add(oit_heads_tex_.handle,
                                (Access)(Access::ComputeRead | Access::ComputeWrite));
      transparent_draw_pass.add(oit_fragment_buffer_.handle, Access::ComputeWrite);
      transparent_draw_pass.set_execute_fn([&rg, rg_depth_handle, this](CmdEncoder& cmd) {
        cmd.begin_region("oit draw");
        auto dims = rg.get_texture(rg_depth_handle)->size();
        state_
            .buffer_barrier(*device_->get_buffer(oit_atomic_counter_buf_),
                            VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT)
            .flush_barriers();
        cmd.fill_buffer(oit_atomic_counter_buf_.handle, 0, constants::whole_size, 0);
        state_
            .buffer_barrier(
                *device_->get_buffer(oit_atomic_counter_buf_),
                VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_WRITE_BIT | VK_ACCESS_2_SHADER_READ_BIT)
            .flush_barriers();

        cmd.begin_rendering({.extent = dims},
                            {RenderingAttachmentInfo::depth_stencil_att(
                                rg.get_texture_handle(rg_depth_handle), LoadOp::Load)});
        cmd.set_viewport_and_scissor(dims);
        cmd.bind_pipeline(PipelineBindPoint::Graphics, transparent_oit_pipeline_);

        // TODO: clear atomic count buffer
        TransparentPushConstants pc{
            .instance_buffer = static_instance_data_buf_.get_buffer()->device_addr(),
            .vertex_buffer = static_vertex_buf_.get_buffer()->device_addr(),
            .object_data_buffer = static_object_data_buf_.get_buffer()->device_addr(),
            .materials_buffer = static_materials_buf_.get_buffer()->device_addr(),
            .scene_buffer = device_->get_bindless_idx(curr_frame().scene_uniform_buf),
            .irradiance_img_idx =
                device_->get_bindless_idx(ibl_->irradiance_cubemap_tex_, SubresourceType::Shader),
            .brdf_lut_idx =
                device_->get_bindless_idx(ibl_->brdf_lut_.handle, SubresourceType::Shader),
            .prefiltered_env_map_idx = device_->get_bindless_idx(
                ibl_->prefiltered_env_map_tex_.handle, SubresourceType::Shader),
            .linear_clamp_to_edge_sampler_idx =
                device_->get_bindless_idx(linear_sampler_clamp_to_edge_),
            .atomic_counter_buffer = device_->get_bindless_idx(oit_atomic_counter_buf_),
            .max_oit_fragments = max_oit_fragments_,
            .oit_tex_heads = device_->get_bindless_idx(oit_heads_tex_, SubresourceType::Storage),
            .oit_lists_buf = device_->get_bindless_idx(oit_fragment_buffer_),
            .sampler_idx = device_->get_bindless_idx(linear_sampler_),
            .img_size = {dims.x, dims.y},
            .depth_img = device_->get_bindless_idx(rg.get_texture_handle(rg_depth_handle),
                                                   SubresourceType::Shader)};
        cmd.push_constants(sizeof(pc), &pc);
        cmd.set_cull_mode(CullMode::Back);
        execute_static_geo_draws(cmd, MeshPass_Transparent);
        cmd.set_cull_mode(CullMode::None);
        execute_static_geo_draws(cmd, MeshPass_TransparentDoubleSided);

        cmd.end_rendering();
        cmd.end_region();
      });
    }

    {
      auto& oit_pass = rg.add_pass("oit");
      auto draw_img_handle = oit_pass.add("draw_out", {.format = draw_img_format_},
                                          (Access)(Access::ColorRead | Access::ComputeRead));
      auto oit_out_handle = oit_pass.add(post_process_input_img_name, {.format = draw_img_format_},
                                         Access::ComputeWrite);
      oit_pass.add(oit_heads_tex_.handle, Access::ComputeRead);
      oit_pass.add(oit_fragment_buffer_.handle, Access::ComputeRead);
      oit_pass.set_execute_fn([this, &rg, draw_img_handle, oit_out_handle](CmdEncoder& cmd) {
        cmd.bind_pipeline(PipelineBindPoint::Compute, oit_comp_pipeline_);
        struct {
          u64 oit_lists_buf;
          u32 oit_heads_tex;
          float opacity_boost;
          u32 color_tex;
          u32 out_tex;
          float time;
          u32 show_heatmap;
        } pc{
            device_->get_buffer(oit_fragment_buffer_)->device_addr(),
            device_->get_bindless_idx(oit_heads_tex_, SubresourceType::Storage),
            oit_opacity_boost_,
            device_->get_bindless_idx(rg.get_texture_handle(draw_img_handle),
                                      SubresourceType::Storage),
            device_->get_bindless_idx(rg.get_texture_handle(oit_out_handle),
                                      SubresourceType::Storage),
            static_cast<float>(glfwGetTime()),
            (u32)oit_debug_heatmap_,
        };
        auto dims = device_->get_image(oit_heads_tex_)->size();
        cmd.push_constants(sizeof(pc), &pc);
        cmd.dispatch((dims.x + 16) / 16, (dims.y + 16) / 16, 1);
      });
    }
  }

  {
    auto& pp = rg.add_pass("post_process");
    auto draw_out_handle =
        pp.add(post_process_input_img_name, {.format = draw_img_format_}, Access::FragmentRead);
    auto final_out_handle =
        pp.add("final_out", AttachmentInfo{.format = device_->get_swapchain_info().format},
               Access::ColorWrite);
    pp.set_execute_fn([this, &rg, draw_out_handle, final_out_handle](CmdEncoder& cmd) {
      cmd.begin_region("post_process");
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

        auto tex = rg.get_texture_handle(final_out_handle);
        auto dims = device_->get_image(tex)->size();
        cmd.begin_rendering({.extent = dims}, {RenderingAttachmentInfo::color_att(tex)});
        cmd.set_viewport_and_scissor(dims);
        cmd.bind_pipeline(PipelineBindPoint::Graphics, postprocess_pipeline_);
        auto post_processed_img = rg.get_texture_handle(final_out_handle);
        struct {
          u32 in_tex_idx, out_tex_idx, flags, tonemap_type;
        } pc{device_->get_bindless_idx(rg.get_texture_handle(draw_out_handle),
                                       SubresourceType::Shader),
             device_->get_bindless_idx(post_processed_img, SubresourceType::Storage),
             postprocess_flags, tonemap_type_};
        cmd.push_constants(sizeof(pc), &pc);
        cmd.draw(3);
        cmd.end_rendering();
        cmd.end_region();
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
          cmd.begin_region("ui");
          if (csm_enabled.get() && csm_->get_debug_render_enabled()) {
            assert(csm_debug_img_handle.is_valid());
            csm_->imgui_pass(cmd, linear_sampler_, rg.get_texture_handle(csm_debug_img_handle));
          }
          auto color_handle = rg.get_texture_handle(rg_color_handle);
          auto depth_handle = rg.get_texture_handle(rg_depth_handle);
          uvec2 extent = device_->get_image(color_handle)->size();
          if (line_draw_vertices_.size()) {
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
          cmd.set_viewport_and_scissor(extent);
          device_->render_imgui(cmd);
          cmd.end_rendering();
          cmd.end_region();
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

void VkRender2::StaticMeshDrawManager::remove_draws(StateTracker&, CmdEncoder& cmd, u32 handle) {
  if (handle == null_handle) {
    return;
  }
  assert(handle < allocs_.size());
  Alloc& a = allocs_[handle];
  u32 num_draws = a.draw_cmd_slot.get_size() / sizeof(GPUDrawInfo);
  num_draw_cmds_ -= num_draws;
  cmd.fill_buffer(draw_cmds_buf_.buffer.handle, a.draw_cmd_slot.get_offset(),
                  a.draw_cmd_slot.get_size(), 0);
  draw_cmds_buf_.allocator.free(a.draw_cmd_slot);
  free_alloc_indices_.emplace_back(handle);
}

u32 VkRender2::StaticMeshDrawManager::add_draws(StateTracker& state,
                                                Device::CopyAllocator::CopyCmd& cmd, size_t size,
                                                size_t staging_offset) {
  assert(size > 0);
  Alloc a{};
  a.draw_cmd_slot = draw_cmds_buf_.allocator.allocate(size);
  u32 num_draws = (size / sizeof(GPUDrawInfo));
  num_draw_cmds_ += num_draws;

  // resize draw cmd bufs
  size_t curr_tot_draw_cmd_buf_size = device_->get_buffer(draw_cmds_buf_.buffer)->size();
  auto new_size = glm::max<size_t>(curr_tot_draw_cmd_buf_size * 2,
                                   a.draw_cmd_slot.get_offset() + a.draw_cmd_slot.get_size());
  for (auto& draw_pass : draw_passes_) {
    // resize output draw cmd buffers
    for (auto& handle : draw_pass.out_draw_cmds_bufs) {
      u32 required_size = sizeof(u32) + (num_draw_cmds_ * sizeof(VkDrawIndexedIndirectCommand));
      auto* buf = device_->get_buffer(handle);
      if (required_size > buf->size()) {
        handle = device_->create_buffer_holder(BufferCreateInfo{
            .size = new_size + sizeof(u32),
            .usage = BufferUsage_Indirect | BufferUsage_Storage,
        });
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

  if (free_alloc_indices_.size()) {
    auto h = free_alloc_indices_.back();
    free_alloc_indices_.pop_back();
    allocs_[h] = a;
    return h;
  }

  auto h = allocs_.size();
  allocs_.emplace_back(a);
  return h;
}

Buffer* VkRender2::StaticMeshDrawManager::get_draw_info_buf() const {
  return draw_cmds_buf_.get_buffer();
}

BufferHandle VkRender2::StaticMeshDrawManager::get_draw_info_buf_handle() const {
  return draw_cmds_buf_.buffer.handle;
}

void VkRender2::execute_static_geo_draws(CmdEncoder& cmd, MeshPass pass, bool is_animated) {
  cmd.bind_index_buffer(static_index_buf_.buffer.handle);
  cmd.set_cull_mode(get_cull_mode(pass));
  auto& mgr = is_animated ? animated_draw_mgrs_[pass] : static_draw_mgrs_[pass];
  if (mgr.should_draw() && mgr.get_num_draw_cmds() > 0) {
    execute_draw(
        cmd,
        mgr.get_draw_pass(main_view_mesh_pass_indices_[pass]).get_frame_out_draw_cmd_buf_handle(),
        mgr.get_num_draw_cmds());
  }
}

VkRender2::StaticMeshDrawManager::DrawPass::DrawPass(u32 num_draws, u32 frames_in_flight,
                                                     Device* device)
    : device_(device) {
  for (u32 i = 0; i < frames_in_flight; i++) {
    out_draw_cmds_bufs.emplace_back(device_->create_buffer_holder(BufferCreateInfo{
        .size = (num_draws * sizeof(VkDrawIndexedIndirectCommand)) + sizeof(u32),
        .usage = BufferUsage_Indirect | BufferUsage_Storage,
    }));
  }
}

u32 VkRender2::StaticMeshDrawManager::add_draw_pass() {
  auto idx = draw_passes_.size();
  draw_passes_.emplace_back(num_draw_cmds_, device_->get_frames_in_flight(), device_);
  return idx;
}

BufferHandle VkRender2::StaticMeshDrawManager::DrawPass::get_frame_out_draw_cmd_buf_handle() const {
  return out_draw_cmds_bufs[VkRender2::get().curr_frame_in_flight_num()].handle;
}

Buffer* VkRender2::StaticMeshDrawManager::DrawPass::get_frame_out_draw_cmd_buf() const {
  return device_->get_buffer(out_draw_cmds_bufs[VkRender2::get().curr_frame_in_flight_num()]);
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
      {-1, -1, -1},
      {-1, -1, 1},
      {-1, 1, -1},
      {-1, 1, 1},
      {1, -1, -1},
      {1, -1, 1},
      {1, 1, -1},
      {1, 1, 1},
  }};
  for (auto& p : pts) {
    p = model * vec4(p * size, 1.f);  // scale *after*
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
  draw_box(model * glm::translate(mat4{1}, .5f * (aabb.min + aabb.max)),
           .5f * (aabb.max - aabb.min), color);
}

void VkRender2::free(StaticModelInstanceResources& instance) {
  auto& instance_buf =
      instance.is_animated ? animated_instance_data_buf_ : static_instance_data_buf_;
  auto& obj_data_buf = instance.is_animated ? animated_object_data_buf_ : static_object_data_buf_;
  instance_buf.allocator.free(instance.instance_data_slot);
  obj_data_buf.allocator.free(instance.object_data_slot);
  if (instance.is_animated) {
    animated_vertex_output_bufs_.allocator.free(instance.animated_vertex_buf_slot);
  }
  // free the draws (need to clear to 0)
  auto* pmodel = ResourceManager::get().get_model(instance.model_handle);
  assert(pmodel);

  // ref counting should be tracked in resource manager

  auto* resources = model_gpu_resources_pool_.get(pmodel->gpu_resource_handle);
  if (resources->ref_count == 0) {
    LERROR("uh oh");
    exit(1);
  }
  resources->ref_count--;
  if (resources->ref_count == 0) {
    draw_stats_.vertices -= resources->num_vertices;
    draw_stats_.indices -= resources->num_indices;
    draw_stats_.materials -= resources->materials_slot.get_size() / sizeof(Material);
    draw_stats_.textures -= resources->textures.size();
    static_materials_buf_.allocator.free(resources->materials_slot);
    static_vertex_buf_.allocator.free(resources->vertices_slot);
    static_index_buf_.allocator.free(resources->indices_slot);
    model_gpu_resources_pool_.destroy(pmodel->gpu_resource_handle);
  }
}

void VkRender2::free(CmdEncoder& cmd, StaticModelInstanceResources& instance) {
  auto& mgrs = instance.is_animated ? animated_draw_mgrs_ : static_draw_mgrs_;
  for (auto& mgr : mgrs) {
    mgr.remove_draws(state_, cmd, instance.mesh_pass_draw_handles[mgr.get_mesh_pass()]);
  }
  free(instance);
}

uvec2 VkRender2::window_dims() const {
  int x, y;
  glfwGetWindowSize(window_, &x, &y);
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
                                            Device* device, const std::string& name) {
  name_ = name;
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

bool VkRender2::load_model2(const std::filesystem::path& path, LoadedModelData& result) {
  auto load_result = gfx::load_gltf(path, gfx::DefaultMaterialData{});
  if (load_result.has_value()) {
    auto& res = *load_result;
    result.animations = std::move(res.animations);
    u64 material_data_size = res.materials.size() * sizeof(gfx::Material);
    u64 vertices_size = res.vertices.size() * sizeof(gfx::Vertex);
    u64 indices_size = res.indices.size() * sizeof(u32);
    u64 animated_vertices_size = res.animated_vertices.size() * sizeof(gfx::AnimatedVertex);
    util::FreeListAllocator::Slot animated_vertices_gpu_slot{};
    if (animated_vertices_size) {
      animated_vertices_gpu_slot = animated_vertex_buf_.allocator.allocate(animated_vertices_size);
    }
    auto vertices_gpu_slot = static_vertex_buf_.allocator.allocate(vertices_size);
    auto indices_gpu_slot = static_index_buf_.allocator.allocate(indices_size);

    auto copy_cmd = device_->graphics_copy_allocator_.allocate(
        material_data_size + vertices_size + animated_vertices_size + indices_size);
    auto staging = LinearCopyer{device_->get_buffer(copy_cmd.staging_buffer)->mapped_data()};
    u64 material_data_staging_offset = staging.copy(res.materials.data(), material_data_size);
    u64 vertices_staging_offset = staging.copy(res.vertices.data(), vertices_size);
    u64 animated_vertices_staging_offset{};
    if (animated_vertices_size) {
      animated_vertices_staging_offset =
          staging.copy(res.animated_vertices.data(), animated_vertices_size);
    }
    u64 indices_staging_offset = staging.copy(res.indices.data(), indices_size);

    draw_stats_.vertices += res.vertices.size();
    draw_stats_.indices += res.indices.size();
    draw_stats_.textures += res.textures.size();
    draw_stats_.materials += res.materials.size();
    auto materials_gpu_slot = static_materials_buf_.allocator.allocate(material_data_size);
    {
      state_.reset(copy_cmd.transfer_cmd_buf);
      state_
          .buffer_barrier(static_vertex_buf_.get_buffer()->buffer(),
                          VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT)
          .buffer_barrier(static_index_buf_.get_buffer()->buffer(),
                          VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT)
          .buffer_barrier(static_materials_buf_.get_buffer()->buffer(),
                          VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
      if (animated_vertices_size) {
        state_.buffer_barrier(animated_vertex_buf_.get_buffer()->buffer(),
                              VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
      }
      state_.flush_barriers();

      if (animated_vertices_size) {
        copy_cmd.copy_buffer(device_, *device_->get_buffer(animated_vertex_buf_.buffer),
                             animated_vertices_staging_offset,
                             animated_vertices_gpu_slot.get_offset(), animated_vertices_size);
      }
      copy_cmd.copy_buffer(device_, *device_->get_buffer(static_materials_buf_.buffer),
                           material_data_staging_offset, materials_gpu_slot.get_offset(),
                           material_data_size);
      copy_cmd.copy_buffer(device_, *static_vertex_buf_.get_buffer(), vertices_staging_offset,
                           vertices_gpu_slot.get_offset(), vertices_size);
      copy_cmd.copy_buffer(device_, *static_index_buf_.get_buffer(), indices_staging_offset,
                           indices_gpu_slot.get_offset(), indices_size);

      state_
          .buffer_barrier(
              static_vertex_buf_.get_buffer()->buffer(),
              VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_VERTEX_INPUT_BIT,
              VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT | VK_ACCESS_2_SHADER_READ_BIT)
          .buffer_barrier(static_index_buf_.get_buffer()->buffer(),
                          VK_PIPELINE_STAGE_2_INDEX_INPUT_BIT, VK_ACCESS_2_INDEX_READ_BIT)
          .buffer_barrier(static_materials_buf_.get_buffer()->buffer(),
                          VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT);
      if (animated_vertices_size) {
        state_.buffer_barrier(
            animated_vertex_buf_.get_buffer()->buffer(),
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_VERTEX_INPUT_BIT,
            VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT | VK_ACCESS_2_SHADER_READ_BIT);
      }
      state_.flush_barriers();

      device_->graphics_copy_allocator_.submit(copy_cmd);
    }

    result.scene_graph_data = std::move(load_result.value().scene_graph_data);
    // result.materials = std::move(load_result.value().materials);
    result.gpu_resource_handle = model_gpu_resources_pool_.alloc();
    auto* resources = model_gpu_resources_pool_.get(result.gpu_resource_handle);
    resources->animated_vertices_gpu_slot = animated_vertices_gpu_slot;
    resources->animated_vertices = std::move(res.vertices);
    resources->textures = std::move(res.textures);
    resources->mesh_draw_infos = res.mesh_draw_infos;
    resources->materials_slot = materials_gpu_slot;
    resources->vertices_slot = vertices_gpu_slot;
    resources->indices_slot = indices_gpu_slot;
    resources->materials = std::move(res.materials);
    resources->num_vertices = res.vertices.size();
    resources->num_indices = res.indices.size();
    resources->name = path;
    resources->first_vertex = vertices_gpu_slot.get_offset() / sizeof(gfx::Vertex);
    resources->first_index = indices_gpu_slot.get_offset() / sizeof(u32);
    resources->ref_count = 0;
    return true;
  }
  return false;
}

// TODO: thread safe
StaticModelInstanceResourcesHandle VkRender2::add_instance(ModelHandle model_handle) {
  auto* pmodel = ResourceManager::get().get_model(model_handle);
  assert(pmodel);
  if (!pmodel) {
    return {};
  }
  auto& model = *pmodel;
  auto* resources = model_gpu_resources_pool_.get(model.gpu_resource_handle);
  assert(resources);
  draw_stats_.total_vertices += resources->num_vertices;
  draw_stats_.total_indices += resources->num_indices;

  u32 pass_obj_counts[MeshPass_Count] = {};

  const auto& scene = model.scene_graph_data;

  assert(scene.hierarchies.size() == scene.node_mesh_indices.size());
  for (size_t node_i = 0; node_i < scene.hierarchies.size(); node_i++) {
    auto mesh_data_i = scene.node_mesh_indices[node_i];
    if (mesh_data_i == -1) continue;
    const auto& mesh_indices = scene.mesh_datas[mesh_data_i];
    bool double_sided = resources->materials[mesh_indices.material_id].is_double_sided();
    MeshPass pass{MeshPass_Count};
    if (mesh_indices.pass_flags & PassFlags_Opaque) {
      pass = MeshPass_Opaque;
    } else if (mesh_indices.pass_flags & PassFlags_OpaqueAlpha) {
      pass = MeshPass_OpaqueAlphaMask;
    } else if (mesh_indices.pass_flags & PassFlags_Transparent) {
      pass = MeshPass_Transparent;
    }
    pass_obj_counts[double_sided ? get_double_sided_pass(pass) : pass]++;
  }
  resources->ref_count++;

  u32 num_objs_tot{};
  for (auto pass_obj_count : pass_obj_counts) {
    num_objs_tot += pass_obj_count;
  }

  auto model_instance_resources_handle = static_model_instance_pool_.alloc();
  auto* instance_resources = static_model_instance_pool_.get(model_instance_resources_handle);
  std::ranges::fill(instance_resources->mesh_pass_draw_handles, StaticMeshDrawManager::null_handle);
  instance_resources->model_handle = model_handle;
  instance_resources->name = resources->name.c_str();
  instance_resources->object_datas.reserve(num_objs_tot);
  instance_resources->instance_datas.reserve(num_objs_tot);

  if (model.scene_graph_data.skins.size()) {
    instance_resources->is_animated = true;
  }
  // FreeListBuffer& instance_data_buf =
  //     instance_resources->is_animated ? animated_instance_data_buf_ : static_instance_data_buf_;
  // FreeListBuffer& object_data_buf =
  //     instance_resources->is_animated ? animated_object_data_buf_ : static_object_data_buf_;
  FreeListBuffer& instance_data_buf = static_instance_data_buf_;
  FreeListBuffer& object_data_buf = static_object_data_buf_;

  vec3 scene_min = vec3{std::numeric_limits<float>::max()};
  vec3 scene_max = vec3{std::numeric_limits<float>::lowest()};
  std::array<std::vector<GPUDrawInfo>, MeshPass_Count> pass_cmds;
  instance_resources->instance_data_slot =
      instance_data_buf.allocator.allocate(num_objs_tot * sizeof(GPUInstanceData));
  instance_resources->object_data_slot =
      object_data_buf.allocator.allocate(num_objs_tot * sizeof(ObjectData));
  u32 base_instance_id =
      instance_resources->instance_data_slot.get_offset() / sizeof(GPUInstanceData);
  u32 base_object_data_id = instance_resources->object_data_slot.get_offset() / sizeof(ObjectData);
  auto base_material_id = resources->materials_slot.get_offset() / sizeof(Material);

  // if animated:
  // copy the model's vertices output buffers since each instance needs its own
  u32 first_vertex{};
  std::vector<SkinCommand> skin_cmds;
  if (instance_resources->is_animated) {
    instance_resources->animated_vertex_buf_slot = animated_vertex_output_bufs_.allocator.allocate(
        resources->animated_vertices.size() * sizeof(Vertex));
    first_vertex = instance_resources->animated_vertex_buf_slot.get_offset() / sizeof(Vertex);

    draw_stats_.animated_vertices += resources->animated_vertices.size();

    // update global bone matrices
    u32 num_new_skin_mats{};
    for (const auto& skin : model.scene_graph_data.skins) {
      num_new_skin_mats += skin.inverse_bind_matrices.size();
    }
    if (num_new_skin_mats > 0) {
      instance_resources->global_bone_mat_slot =
          global_skin_mat_allocator_.allocate(sizeof(mat4) * num_new_skin_mats);
      global_skin_matrices_.resize(global_skin_mat_allocator_.max_seen_size() / sizeof(mat4));
    }

    {
      skin_cmds.reserve(resources->animated_vertices.size());
      u32 skin_vtx_i = resources->animated_vertices_gpu_slot.get_offset() / sizeof(AnimatedVertex);
      u32 output_vertex_i = first_vertex;
      u32 bone_mat_i = instance_resources->global_bone_mat_slot.get_offset() / sizeof(mat4);
      for (size_t i = 0; i < resources->animated_vertices.size(); i++) {
        skin_cmds.emplace_back(SkinCommand{
            .skin_vtx_i = skin_vtx_i++,
            .out_vtx_i = output_vertex_i++,
            .bone_mat_start_i = bone_mat_i,
        });
      }
      instance_resources->skin_commands_slot =
          skin_instance_datas_.allocator.allocate(skin_cmds.size() * sizeof(SkinCommand));
    }

    // update skin instance datas so skinned vertices can know what to do

  } else {
    first_vertex = resources->first_vertex;
  }

  for (size_t node_i = 0; node_i < scene.hierarchies.size(); node_i++) {
    // TODO: model wide?
    instance_resources->node_to_instance_and_obj.emplace_back(-1);
    auto mesh_data_i = scene.node_mesh_indices[node_i];
    if (mesh_data_i == -1) continue;
    const auto& node_mesh_data = scene.mesh_datas[mesh_data_i];
    // auto& node_mesh_data
    auto& mesh = resources->mesh_draw_infos[node_mesh_data.mesh_idx];
    // TODO: get rid
    const mat4 model = scene.global_transforms[node_i];
    AABB world_space_aabb = transform_aabb(model, mesh.aabb);
    scene_min = glm::min(scene_min, world_space_aabb.min);
    scene_max = glm::max(scene_max, world_space_aabb.max);
    u32 instance_id = base_instance_id + instance_resources->instance_datas.size();
    instance_resources->node_to_instance_and_obj.back() = instance_resources->instance_datas.size();
    instance_resources->instance_datas.emplace_back(
        node_mesh_data.material_id + base_material_id,
        base_object_data_id + instance_resources->object_datas.size());
    instance_resources->object_datas.emplace_back(gfx::ObjectData{
        .model = model,
        .aabb_min = vec4(world_space_aabb.min, 0.),
        .aabb_max = vec4(world_space_aabb.max, 0.),
    });

    u32 draw_flags{};
    bool double_sided = resources->materials[node_mesh_data.material_id].is_double_sided();
    if (double_sided) {
      draw_flags |= GPUDrawInfoFlags_DoubleSided;
    }

    GPUDrawInfo draw{.index_cnt = mesh.index_count,
                     .first_index = static_cast<u32>(resources->first_index + mesh.first_index),
                     .vertex_offset = first_vertex + mesh.first_vertex,
                     .instance_id = instance_id,
                     .flags = draw_flags};
    if (node_mesh_data.pass_flags & PassFlags_Opaque) {
      if (double_sided) {
        pass_cmds[MeshPass_OpaqueDoubleSided].emplace_back(draw);
      } else {
        pass_cmds[MeshPass_Opaque].emplace_back(draw);
      }
    } else if (node_mesh_data.pass_flags & PassFlags_OpaqueAlpha) {
      if (double_sided) {
        pass_cmds[MeshPass_OpaqueAlphaMaskDoubleSided].emplace_back(draw);
      } else {
        pass_cmds[MeshPass_OpaqueAlphaMask].emplace_back(draw);
      }
    } else if (node_mesh_data.pass_flags & PassFlags_Transparent) {
      if (double_sided) {
        pass_cmds[MeshPass_TransparentDoubleSided].emplace_back(draw);
      } else {
        pass_cmds[MeshPass_Transparent].emplace_back(draw);
      }
    }
  }

  u64 obj_datas_size = instance_resources->object_datas.size() * sizeof(gfx::ObjectData);
  u64 instance_datas_size = instance_resources->instance_datas.size() * sizeof(GPUInstanceData);

  scene_aabb_.min = glm::min(scene_aabb_.min, scene_min);
  scene_aabb_.max = glm::max(scene_aabb_.max, scene_max);

  u32 tot_draw_cmds_buf_size = 0;
  for (auto& cmds : pass_cmds) {
    tot_draw_cmds_buf_size += cmds.size() * sizeof(GPUDrawInfo);
  }
  u64 skin_cmd_copy_size = skin_cmds.size() * sizeof(SkinCommand);
  auto copy_cmd = device_->graphics_copy_allocator_.allocate(
      tot_draw_cmds_buf_size + obj_datas_size + instance_datas_size +
      instance_resources->animated_vertex_buf_slot.get_size() + skin_cmd_copy_size);
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
  u64 instance_datas_staging_offset =
      staging.copy(instance_resources->instance_datas.data(), instance_datas_size);
  u64 skin_instance_datas_staging_offset{};
  if (skin_cmds.size()) {
    skin_instance_datas_staging_offset = staging.copy(skin_cmds.data(), skin_cmd_copy_size);
  }

  u64 vertices_staging_offset{};
  if (instance_resources->is_animated) {
    vertices_staging_offset = staging.copy(resources->animated_vertices.data(),
                                           instance_resources->animated_vertex_buf_slot.get_size());
  }

  {
    assert(obj_datas_size && instance_datas_size);
    state_.reset(copy_cmd.transfer_cmd_buf);
    for (int i = 0; i < MeshPass_Count; i++) {
      auto& cmds = pass_cmds[i];
      if (cmds.size()) {
        auto& mgr = get_mgr((MeshPass)i, instance_resources->is_animated);
        instance_resources->mesh_pass_draw_handles[i] = mgr.add_draws(
            state_, copy_cmd, cmds.size() * sizeof(GPUDrawInfo), cmds_staging_offsets[i]);
      }
    }
    copy_cmd.copy_buffer(device_, *object_data_buf.get_buffer(), obj_datas_staging_offset,
                         instance_resources->object_data_slot.get_offset(), obj_datas_size);
    copy_cmd.copy_buffer(device_, *instance_data_buf.get_buffer(), instance_datas_staging_offset,
                         instance_resources->instance_data_slot.get_offset(), instance_datas_size);
    state_
        .buffer_barrier(
            object_data_buf.get_buffer()->buffer(),
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT,
            VK_ACCESS_2_SHADER_READ_BIT)
        .buffer_barrier(instance_data_buf.get_buffer()->buffer(),
                        VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT)
        .flush_barriers();
    if (instance_resources->is_animated) {
      for (size_t i = 0; i < device_->get_frames_in_flight(); i++) {
        state_.buffer_barrier(
            device_->get_buffer(animated_vertex_output_bufs_.buffers[i])->buffer(),
            VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
      }
      state_.flush_barriers();
      // copy to output vertices
      for (size_t i = 0; i < device_->get_frames_in_flight(); i++) {
        copy_cmd.copy_buffer(
            device_, *device_->get_buffer(animated_vertex_output_bufs_.buffers[i].handle),
            vertices_staging_offset, instance_resources->animated_vertex_buf_slot.get_offset(),
            instance_resources->animated_vertex_buf_slot.get_size());
      }
      // TODO: get rid of these barriers
      for (size_t i = 0; i < device_->get_frames_in_flight(); i++) {
        state_.buffer_barrier(
            device_->get_buffer(animated_vertex_output_bufs_.buffers[i])->buffer(),
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_VERTEX_INPUT_BIT,
            VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT | VK_ACCESS_2_SHADER_READ_BIT);
      }
      state_.flush_barriers();

      if (skin_cmds.size()) {
        copy_cmd.copy_buffer(device_, *device_->get_buffer(skin_instance_datas_.buffer),
                             skin_instance_datas_staging_offset,
                             instance_resources->skin_commands_slot.get_offset(),
                             skin_cmd_copy_size);
        state_
            .buffer_barrier(device_->get_buffer(skin_instance_datas_.buffer)->buffer(),
                            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT)
            .flush_barriers();
      }
    }
    device_->graphics_copy_allocator_.submit(copy_cmd);
  }
  return model_instance_resources_handle;
}

std::string to_string(MeshPass p) {
  switch (p) {
    case MeshPass_Opaque:
      return "MeshPass_Opaque";
    case MeshPass_OpaqueDoubleSided:
      return "MeshPass_OpaqueDoubleSided";
    case MeshPass_OpaqueAlphaMask:
      return "MeshPass_OpaqueAlphaMask";
    case MeshPass_OpaqueAlphaMaskDoubleSided:
      return "MeshPass_OpaqueAlphaMaskDoubleSided";
    case MeshPass_Transparent:
      return "MeshPass_Transparent";
    case MeshPass_TransparentDoubleSided:
      return "MeshPass_TransparentDoubleSided";
    case MeshPass_Count:
      return "MeshPass_Count";
  }
}

void VkRender2::remove_instance(StaticModelInstanceResourcesHandle handle) {
  to_delete_static_model_instances_.emplace_back(handle);
}

void VkRender2::update_transforms(LoadedInstanceData& instance, std::vector<i32>& changed_nodes) {
  ZoneScoped;
  auto* instance_resources = static_model_instance_pool_.get(instance.instance_resources_handle);
  assert(instance_resources);
  auto* model = ResourceManager::get().get_model(instance.model_handle);
  assert(model);
  auto* model_resources = model_gpu_resources_pool_.get(model->gpu_resource_handle);
  auto& scene = instance.scene_graph_data;
  for (auto node_i : changed_nodes) {
    auto mesh_data_i = scene.node_mesh_indices[node_i];
    if (mesh_data_i == -1) continue;
    const auto& mesh_info =
        model_resources->mesh_draw_infos[scene.mesh_datas[mesh_data_i].mesh_idx];
    auto instance_i = instance_resources->node_to_instance_and_obj[node_i];
    object_data_buffer_copier_.add_copy(
        object_datas_to_copy_.size() * sizeof(ObjectData),
        (instance_i * sizeof(ObjectData)) + instance_resources->object_data_slot.get_offset(),
        sizeof(ObjectData));

    AABB world_aabb = transform_aabb(scene.global_transforms[node_i], mesh_info.aabb);
    object_datas_to_copy_.emplace_back(scene.global_transforms[node_i], vec4{world_aabb.min, 0.},
                                       vec4{world_aabb.max, 0.});
    // object_datas_to_copy_.emplace_back(scene.global_transforms[node_i],
    //                                    vec4{mesh_info.aabb.min, 0.}, vec4{mesh_info.aabb.max,
    //                                    0.});
    instance_resources->object_datas[instance_i] = object_datas_to_copy_.back();
  }
}

void VkRender2::mark_dirty(InstanceHandle handle) { dirty_instances_.emplace_back(handle); }

namespace {

float get_interpolation_value(float start_anim_t, float end_anim_t, float curr_t) {
  return (curr_t - start_anim_t) / (end_anim_t - start_anim_t);
}

}  // namespace

void VkRender2::update_animation(LoadedInstanceData& instance, float dt) {
  ZoneScoped;
  LoadedModelData* model = ResourceManager::get().get_model(instance.model_handle);
  assert(model);
  size_t anim_i = 0;
  for (auto& animation : model->animations) {
    assert(animation.duration > 0.f);
    AnimationState& anim_state = instance.animation_states[anim_i];
    if (!anim_state.active) {
      anim_i++;
      continue;
    }
    anim_state.curr_t += animation.ticks_per_second * dt;
    if (anim_state.play_once && anim_state.curr_t > animation.duration) {
      // animation is over
      anim_state.active = false;
      anim_state.curr_t = animation.duration;
    } else {
      anim_state.curr_t = std::fmodf(anim_state.curr_t, animation.duration);
    }
    assert(anim_state.anim_id != UINT32_MAX);
    anim_i++;
  }

  // traverse from the root node? or iterate animations?
  anim_i = 0;
  for (Animation& animation : model->animations) {
    const AnimationState& anim_state = instance.animation_states[anim_i];
    if (!anim_state.active) {
      anim_i++;
      continue;
    }
    for (size_t channel_i = 0; channel_i < animation.channels.nodes.size(); channel_i++) {
      int channel_node_i = animation.get_channel_node(channel_i);
      assert(channel_node_i >= 0);
      u32 channel_sampler_i = animation.get_channel_sampler_i(channel_i);
      assert(channel_sampler_i < animation.samplers.size());
      AnimationPath channel_anim_path = animation.get_channel_anim_path(channel_i);
      const AnimSampler& sampler = animation.samplers[channel_sampler_i];
      // binary search
      auto it = std::ranges::lower_bound(sampler.inputs, anim_state.curr_t);
      size_t time_i = 0;
      if (it != sampler.inputs.begin()) {
        time_i = std::distance(sampler.inputs.begin(), it) - 1;
      }
      size_t next_time_i = sampler.inputs.size() == 1 ? 0 : (time_i + 1) % sampler.inputs.size();
      assert(next_time_i < sampler.inputs.size());
      assert(time_i < sampler.inputs.size());
      float interpolation_val =
          sampler.inputs.size() == 1
              ? 0.f
              : get_interpolation_value(sampler.inputs[time_i], sampler.inputs[next_time_i],
                                        anim_state.curr_t);

      auto& nt = instance.scene_graph_data.node_transforms[channel_node_i];
      if (channel_anim_path == AnimationPath::Translation) {
        assert(sampler.outputs_raw.size() == sampler.inputs.size() * 3);
        const vec3* translations = reinterpret_cast<const vec3*>(sampler.outputs_raw.data());
        nt.translation =
            sampler.inputs.size() == 1
                ? translations[0]
                : glm::mix(translations[time_i], translations[next_time_i], interpolation_val);
      } else if (channel_anim_path == AnimationPath::Rotation) {
        assert(sampler.outputs_raw.size() == sampler.inputs.size() * 4);
        assert(time_i < sampler.inputs.size());
        const quat* rotations = reinterpret_cast<const quat*>(sampler.outputs_raw.data());
        if (sampler.inputs.size() == 1) {
          nt.rotation = rotations[0];
        } else {
          quat q0 = rotations[time_i];
          quat q1 = rotations[next_time_i];
          if (glm::dot(q0, q1) < 0.0f) q1 = -q1;  // ensure shortest path
          nt.rotation = glm::normalize(glm::slerp(q0, q1, interpolation_val));
        }
      } else if (channel_anim_path == AnimationPath::Scale) {
        assert(sampler.outputs_raw.size() == sampler.inputs.size() * 3);
        const vec3* scales = reinterpret_cast<const vec3*>(sampler.outputs_raw.data());
        nt.scale = sampler.inputs.size() == 1
                       ? scales[0]
                       : glm::mix(scales[time_i], scales[next_time_i], interpolation_val);
      }
    }

    for (size_t channel_i = 0; channel_i < animation.channels.nodes.size(); channel_i++) {
      auto node_i = animation.get_channel_node(channel_i);
      const auto& nt = instance.scene_graph_data.node_transforms[node_i];
      mat4 t = glm::translate(mat4(1), nt.translation);
      mat4 r = glm::mat4_cast(nt.rotation);
      mat4 s = glm::scale(mat4(1), nt.scale);
      instance.scene_graph_data.local_transforms[node_i] = t * r * s;
      mark_changed(instance.scene_graph_data, node_i);
    }

    anim_i++;
  }
}

// https://stackoverflow.com/questions/6053522/how-to-recalculate-axis-aligned-bounding-box-after-translate-rotate/58630206#58630206
AABB transform_aabb(const glm::mat4& model, const AABB& aabb) {
  AABB result;
  result.min = glm::vec3(model[3]);  // translation part
  result.max = result.min;
  for (int i = 0; i < 3; ++i) {    // for each row (x, y, z in result)
    for (int j = 0; j < 3; ++j) {  // for each column (x, y, z in input aabb)
      float a = model[j][i] * aabb.min[j];
      float b = model[j][i] * aabb.max[j];
      result.min[i] += glm::min(a, b);
      result.max[i] += glm::max(a, b);
    }
  }
  return result;
};

void VkRender2::draw_sphere(const glm::vec3& center, float radius, const glm::vec4& color,
                            int segments) {
  float step = glm::two_pi<float>() / segments;

  // XY plane
  for (int i = 0; i < segments; ++i) {
    float a0 = i * step;
    float a1 = (i + 1) * step;
    glm::vec3 p0 = center + radius * glm::vec3(cos(a0), sin(a0), 0);
    glm::vec3 p1 = center + radius * glm::vec3(cos(a1), sin(a1), 0);
    draw_line(p0, p1, color);
  }

  // XZ plane
  for (int i = 0; i < segments; ++i) {
    float a0 = i * step;
    float a1 = (i + 1) * step;
    glm::vec3 p0 = center + radius * glm::vec3(cos(a0), 0, sin(a0));
    glm::vec3 p1 = center + radius * glm::vec3(cos(a1), 0, sin(a1));
    draw_line(p0, p1, color);
  }

  // YZ plane
  for (int i = 0; i < segments; ++i) {
    float a0 = i * step;
    float a1 = (i + 1) * step;
    glm::vec3 p0 = center + radius * glm::vec3(0, cos(a0), sin(a0));
    glm::vec3 p1 = center + radius * glm::vec3(0, cos(a1), sin(a1));
    draw_line(p0, p1, color);
  }
}

CullMode get_cull_mode(MeshPass pass) {
  switch (pass) {
    case MeshPass_OpaqueDoubleSided:
    case MeshPass_OpaqueAlphaMaskDoubleSided:
    case MeshPass_TransparentDoubleSided:
      return CullMode::None;
    default:
      return CullMode::Back;
  }
  return CullMode::Back;
}

MeshPass get_double_sided_pass(MeshPass pass) {
  switch (pass) {
    case MeshPass_Opaque:
    case MeshPass_OpaqueDoubleSided:
      return MeshPass_OpaqueDoubleSided;
    case MeshPass_OpaqueAlphaMask:
    case MeshPass_OpaqueAlphaMaskDoubleSided:
      return MeshPass_OpaqueAlphaMaskDoubleSided;
    case MeshPass_Transparent:
    case MeshPass_TransparentDoubleSided:
      return MeshPass_TransparentDoubleSided;
    default:
      return MeshPass_Count;
  }
  assert(0);
  return MeshPass_Count;
}

bool VkRender2::update_skins(LoadedInstanceData& instance) {
  ZoneScoped;
  const auto& skins = instance.scene_graph_data.skins;
  auto* instance_resources = static_model_instance_pool_.get(instance.instance_resources_handle);
  assert(instance_resources);
  u32 instance_bone_mat_start_i =
      instance_resources->global_bone_mat_slot.get_offset() / sizeof(mat4);
  for (size_t skin_i = 0; skin_i < skins.size(); skin_i++) {
    const auto& skin = skins[skin_i];
    for (size_t joint_i = 0; joint_i < skin.joint_node_indices.size(); joint_i++) {
      int joint_node = skin.joint_node_indices[joint_i];
      const mat4& inv_bind_mat = skin.inverse_bind_matrices[joint_i];
      const mat4& global_transform_mat = instance.scene_graph_data.global_transforms[joint_node];
      mat4 inv_skel = glm::inverse(instance.scene_graph_data.global_transforms[skin.skeleton_i]);
      assert(instance_bone_mat_start_i + skin.model_bone_mat_start_i + skin_i <
             global_skin_matrices_.size());
      global_skin_matrices_[instance_bone_mat_start_i + skin.model_bone_mat_start_i + joint_i] =
          inv_skel * global_transform_mat * inv_bind_mat;
    }
  }
  return skins.size() > 0;
}

}  // namespace gfx
