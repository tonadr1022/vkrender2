#include "CSM.hpp"

#include <volk.h>
#include <vulkan/vulkan_core.h>

#include <tracy/Tracy.hpp>
#include <tracy/TracyVulkan.hpp>
#include <utility>

#include "AABB.hpp"
#include "BaseRenderer.hpp"
#include "CommandEncoder.hpp"
#include "RenderGraph.hpp"
#include "StateTracker.hpp"
#include "Types.hpp"
#include "glm/ext/matrix_transform.hpp"
#include "imgui.h"
#include "imgui_impl_vulkan.h"
#include "util/CVar.hpp"
#include "vk2/Device.hpp"
#include "vk2/Initializers.hpp"
#include "vk2/ShaderCompiler.hpp"
#include "vk2/VkTypes.hpp"

using namespace gfx::vk2;

namespace {

// TODO: move
AutoCVarInt stable_light_view{"renderer.stable_light_view", "stable_light_view", 0,
                              CVarFlags::EditCheckbox};

void calc_frustum_corners_world_space(std::span<vec4> corners, const mat4& vp_matrix) {
  const auto inv_vp = glm::inverse(vp_matrix);
  for (u32 z = 0, i = 0; z < 2; z++) {
    for (u32 y = 0; y < 2; y++) {
      for (u32 x = 0; x < 2; x++, i++) {
        vec4 pt = inv_vp * vec4((2.f * x) - 1.f, (2.f * y) - 1.f, (float)z, 1.f);
        corners[i] = pt / pt.w;
      }
    }
  }
}

AutoCVarFloat z_pad{"z_pad", "z_padding", 1.5, CVarFlags::EditFloatDrag};

// https://github.com/walbourn/directx-sdk-samples/blob/main/CascadedShadowMaps11/CascadedShadowMaps11.cpp
mat4 calc_light_space_matrix(const mat4& cam_view, const mat4& cam_proj, vec3 light_dir, float,
                             mat4& light_proj) {
  vec3 center{};
  std::array<vec4, 8> corners;
  calc_frustum_corners_world_space(corners, cam_proj * cam_view);
  for (auto v : corners) {
    center += vec3(v);
  }
  center /= 8;
  mat4 light_view = glm::lookAt(center + light_dir, center, {0, 1, 0});
  vec3 min{std::numeric_limits<float>::max()};
  vec3 max{std::numeric_limits<float>::lowest()};
  for (auto corner : corners) {
    vec3 c = light_view * corner;
    min.x = glm::min(min.x, c.x);
    max.x = glm::max(max.x, c.x);
    min.y = glm::min(min.y, c.y);
    max.y = glm::max(max.y, c.y);
    min.z = glm::min(min.z, c.z);
    max.z = glm::max(max.z, c.z);
  }

  float z_padding = (max.z - min.z) * z_pad.get();
  min.z -= z_padding;
  max.z += z_padding;
  // if (min.z < 0) {
  //   min.z *= z_mult;
  // } else {
  //   min.z /= z_mult;
  // }
  // if (max.z < 0) {
  //   max.z /= z_mult;
  // } else {
  //   max.z *= z_mult;
  // }
  light_proj = glm::orthoRH_ZO(min.x, max.x, max.y, min.y, min.z, max.z);
  return light_proj * light_view;
}

// TODO: calculate near and far based on scene AABB:
// https://github.com/walbourn/directx-sdk-samples/blob/main/CascadedShadowMaps11/CascadedShadowsManager.cpp
mat4 calc_light_space_matrix(const mat4& cam_view, const mat4& cam_proj, vec3 light_dir,
                             float z_mult, u32 shadow_map_size, mat4& proj_mat) {
  if (!stable_light_view.get()) {
    return calc_light_space_matrix(cam_view, cam_proj, light_dir, z_mult, proj_mat);
  }

  // get center of frustum corners of cascade
  vec3 center{};
  std::array<vec4, 8> corners;
  calc_frustum_corners_world_space(corners, cam_proj * cam_view);
  for (auto v : corners) {
    center += vec3(v);
  }
  center /= 8;

  // bounding sphere around frustum corners
  float radius = .0f;
  for (u32 i = 0; i < 8; i++) {
    radius = glm::max(radius, glm::length(vec3{corners[i]} - center));
  }
  radius = std::ceil(radius * 16.f) / 16.f;

  // min/max extents == the bounding sphere of the frustum corners
  vec3 max = vec3{radius};
  vec3 min = -max;

  // flip-y in ortho projection in vulkan
  vec3 shadow_cam_pos = center + light_dir;
  mat4 light_space_view = glm::lookAt(shadow_cam_pos, center, {0, 1, 0});

  proj_mat = glm::orthoRH_ZO(min.x, max.x, max.y, min.y, min.z, max.z);
  mat4 shadow_cam_vp = proj_mat * light_space_view;
  // scale origin by shadow map size
  // round it (nearest texel)
  // get the offset
  // scale it back down, only use x,y and apply it to vp matrix
  vec3 shadow_origin = shadow_cam_vp * vec4(vec3(0.), 1.);
  shadow_origin = shadow_origin * (float)shadow_map_size / 2.f;
  vec3 rounded_origin = glm::round(shadow_origin);
  vec3 round_offset = rounded_origin - shadow_origin;
  round_offset = round_offset * 2.f / (float)shadow_map_size;
  round_offset.z = 0;
  shadow_cam_vp[3] += vec4(round_offset, 0.);
  return shadow_cam_vp;
}

void calc_csm_light_space_matrices(std::span<mat4> matrices, std::span<mat4> proj_matrices,
                                   std::span<float> levels, const mat4& cam_view, vec3 light_dir,
                                   float z_mult, float fov_deg, float aspect, float cam_near,
                                   float cam_far, u32 shadow_map_res) {
  assert(matrices.size() && levels.size() && matrices.size() - 1 == levels.size());
  auto get_proj = [&](float near, float far) {
    auto mat = glm::perspective(glm::radians(fov_deg), aspect, near, far);
    mat[1][1] *= -1;
    return mat;
  };

  vec3 dir = -glm::normalize(light_dir);
  matrices[0] = calc_light_space_matrix(cam_view, get_proj(cam_near, levels[0]), dir, z_mult,
                                        shadow_map_res, proj_matrices[0]);
  for (u32 i = 1; i < matrices.size() - 1; i++) {
    matrices[i] = calc_light_space_matrix(cam_view, get_proj(levels[i - 1], levels[i]), dir, z_mult,
                                          shadow_map_res, proj_matrices[i]);
  }
  matrices[matrices.size() - 1] =
      calc_light_space_matrix(cam_view, get_proj(levels[levels.size() - 1], cam_far), dir, z_mult,
                              shadow_map_res, proj_matrices[matrices.size() - 1]);
}

}  // namespace

namespace gfx {

CSM::CSM(BaseRenderer* renderer, DrawFunc draw_fn, AddRenderDependenciesFunc add_deps_fn)
    : draw_fn_(std::move(draw_fn)),
      add_deps_fn_(std::move(add_deps_fn)),
      shadow_map_res_(uvec2{4096}),
      renderer_(renderer) {
  for (auto& b : shadow_data_bufs_) {
    b = get_device().create_buffer_holder(BufferCreateInfo{
        .size = sizeof(ShadowData),
        .usage = BufferUsage_Storage,
    });
  }
  ZoneScoped;
  GraphicsPipelineCreateInfo shadow_depth_info{
      .shaders = {{"shadow_depth.vert", ShaderType::Vertex},
                  {"shadow_depth.frag", ShaderType::Fragment}},
      .rendering = {.depth_format = to_vkformat(Format::D32Sfloat)},
      .rasterization = {.depth_clamp = true, .depth_bias = true},
      .depth_stencil = GraphicsPipelineCreateInfo::depth_enable(true, CompareOp::Less),
      .dynamic_state = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR,
                        VK_DYNAMIC_STATE_DEPTH_BIAS, VK_DYNAMIC_STATE_CULL_MODE},
  };
  shadow_depth_alpha_mask_pipline_ = PipelineManager::get().load(shadow_depth_info);
  shadow_depth_info.shaders.pop_back();
  shadow_depth_pipline_ = PipelineManager::get().load(shadow_depth_info);

  depth_debug_pipeline_ = PipelineManager::get().load(GraphicsPipelineCreateInfo{
      .shaders = {{"fullscreen_quad.vert", ShaderType::Vertex},
                  {"debug/depth_debug.frag", ShaderType::Fragment}},
      .rendering = {.color_formats = {to_vkformat(debug_shadow_img_format_)}},
      .rasterization = {.cull_mode = CullMode::Front},
      .depth_stencil = GraphicsPipelineCreateInfo::depth_disable(),
  });

  shadow_map_img_att_info_ = {.size_class = SizeClass::Absolute,
                              .dims = {shadow_map_res_, 1},
                              .format = Format::D32Sfloat,
                              .layers = cascade_count_};
}

void CSM::imgui_pass(CmdEncoder&, const vk2::Sampler& sampler, const vk2::Image& tex) {
  if (tex.image() != curr_debug_img_) {
    curr_debug_img_ = tex.image();
    if (imgui_set_) {
      ImGui_ImplVulkan_RemoveTexture(imgui_set_);
    }
    imgui_set_ = ImGui_ImplVulkan_AddTexture(sampler.sampler, tex.view().view(),
                                             VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    curr_debug_img_size_ = {tex.extent_2d().width, tex.extent_2d().height};
  }
}

void CSM::on_imgui() {
  ImGui::Checkbox("shadow map debug", &debug_render_enabled_);
  ImGui::SliderFloat("Z mult", &z_mult_, 0.0, 50.f);
  ImGui::DragFloat("Shadow z far", &shadow_z_far_, 1., 0.f, 10000.f);
  ImGui::Checkbox("AABB Based Shadow Z Far", &aabb_based_z_far_);
  ImGui::DragFloat("Min Bias", &min_bias_, .001, 0.00001, max_bias_);
  ImGui::DragFloat("Max Bias", &max_bias_, .001, min_bias_, 0.01);
  ImGui::DragFloat("Cascade Split Linear Factor", &cascade_linear_factor_, .001f, 0.f, 1.f);
  ImGui::Checkbox("Depth Bias", &depth_bias_enabled_);
  if (depth_bias_enabled_) {
    ImGui::DragFloat("Depth Bias Constant", &depth_bias_constant_factor_, .001, 0.f, 2.f);
    ImGui::DragFloat("Depth Bias Slope", &depth_bias_slope_factor_, .001, .0f, 5.f);
  }
  ImGui::DragFloat("PCF Scale", &pcf_scale_, .001, .0f, 5.f);
  ImGui::Checkbox("PCF", &pcf_);
  ImGui::Checkbox("Alpha Cutout", &alpha_cutout_enabled_);
  if (ImGui::TreeNode("shadow map")) {
    ImGui::SliderInt("view level", &debug_cascade_idx_, 0, cascade_count_ - 1);
    if (debug_render_enabled_) {
      ImVec2 window_size = ImGui::GetContentRegionAvail();
      float scale_width = window_size.x / curr_debug_img_size_.x;
      float scaled_height = curr_debug_img_size_.y * scale_width;
      ImGui::Image(reinterpret_cast<ImTextureID>(imgui_set_),
                   ImVec2(window_size.x * .8f, scaled_height * .8f));
    }
    ImGui::TreePop();
  }
}

void CSM::prepare_frame(RenderGraph& rg, u32 frame_num, const mat4& cam_view, vec3 light_dir,
                        float aspect_ratio, float fov_deg, const AABB& aabb, vec3 view_pos) {
  float shadow_z_far = shadow_z_far_;
  if (aabb_based_z_far_) {
    std::array<vec3, 8> aabb_corners;
    aabb.get_corners(aabb_corners);
    float max_dist = std::numeric_limits<float>::lowest();
    for (auto c : aabb_corners) {
      max_dist = std::max(glm::distance(view_pos, c), max_dist);
    }
    shadow_z_far = max_dist;
    shadow_z_far_ = max_dist;
  }
  shadow_z_far = std::max(shadow_z_far, 50.f);

  std::array<float, max_cascade_levels - 1> levels;
  for (u32 i = 0; i < cascade_count_ - 1; i++) {
    float p = (i + 1) / static_cast<float>(cascade_count_);
    float log_split = shadow_z_near_ * std::pow(shadow_z_far / shadow_z_near_, p);
    float linear_split = shadow_z_near_ + ((shadow_z_far - shadow_z_near_) * p);
    float lambda = cascade_linear_factor_;
    levels[i] = (lambda * log_split) + ((1.0f - lambda) * linear_split);
  }
  // TODO: separate camera z near/far
  calc_csm_light_space_matrices(std::span(light_matrices_.data(), cascade_count_),
                                std::span(light_proj_matrices_.data(), cascade_count_),
                                std::span(levels.data(), cascade_count_ - 1), cam_view, light_dir,
                                z_mult_, fov_deg, aspect_ratio, shadow_z_near_, shadow_z_far,
                                shadow_map_res_.x);
  data_ = {.biases = {0., 0., 0., shadow_z_far}};
  for (u32 i = 0; i < cascade_count_; i++) {
    data_.light_space_matrices[i] = light_matrices_[i];
  }
  for (u32 i = 0; i < cascade_count_ - 1; i++) {
    data_.cascade_levels[i] = levels[i];
  }
  data_.settings = {};
  if (pcf_) {
    data_.settings.x |= 0x1;
  }
  data_.biases.x = min_bias_;
  data_.biases.y = max_bias_;
  data_.biases.z = pcf_scale_;
  data_.settings.w = cascade_count_;

  auto& buf = shadow_data_bufs_[frame_num % shadow_data_bufs_.size()];
  rg.set_resource("shadow_data_buf", buf.handle);
}

void CSM::add_pass(RenderGraph& rg) {
  auto& csm_prepare_pass = rg.add_pass("csm_prepare");
  csm_prepare_pass.add_proxy("shadow_data_buf", Access::TransferWrite);
  csm_prepare_pass.set_execute_fn([this](CmdEncoder& cmd) {
    TracyVkZone(cmd.get_tracy_ctx(), cmd.cmd(), "csm_prepare");
    auto* buf = get_device().get_buffer(
        shadow_data_bufs_[renderer_->curr_frame_num() % shadow_data_bufs_.size()]);
    if (!buf) return;
    vkCmdUpdateBuffer(cmd.cmd(), buf->buffer(), 0, sizeof(CSM::ShadowData), &data_);
  });

  auto& csm = rg.add_pass("csm");
  auto rg_shadow_map_img =
      csm.add("shadow_map_img", shadow_map_img_att_info_, Access::DepthStencilWrite);
  add_deps_fn_(csm);
  csm.set_execute_fn([this, rg_shadow_map_img, &rg](CmdEncoder& cmd) {
    TracyVkZone(cmd.get_tracy_ctx(), cmd.cmd(), "csm");
    shadow_map_img_ = rg.get_texture_handle(rg_shadow_map_img);
    init::begin_debug_utils_label(cmd.cmd(), "csm render");
    if (curr_shadow_map_img_ != shadow_map_img_) {
      curr_shadow_map_img_ = shadow_map_img_;
      for (u32 i = 0; i < cascade_count_; i++) {
        shadow_map_img_views_[i] =
            get_device().create_image_view_holder(*get_device().get_image(curr_shadow_map_img_),
                                                  vk2::ImageViewCreateInfo{
                                                      to_vkformat(Format::D32Sfloat),
                                                      VkImageSubresourceRange{
                                                          .aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT,
                                                          .baseMipLevel = 0,
                                                          .levelCount = 1,
                                                          .baseArrayLayer = i,
                                                          .layerCount = 1,
                                                      },
                                                  });
      }
    }

    for (u32 i = 0; i < cascade_count_; i++) {
      VkClearValue depth_clear{.depthStencil = {.depth = 1.f}};
      // TODO: make image views into a handle. when image is destroyed, image views will be too, so
      // then can make new ones
      VkRenderingAttachmentInfo depth_att = init::rendering_attachment_info(
          get_device().get_image_view(shadow_map_img_views_[i])->view(),
          VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, &depth_clear);
      auto* sm_img = get_device().get_image(shadow_map_img_);
      auto rendering_info = init::rendering_info(sm_img->extent_2d(), nullptr, &depth_att);
      vkCmdBeginRenderingKHR(cmd.cmd(), &rendering_info);
      cmd.set_cull_mode(CullMode::None);
      cmd.set_viewport_and_scissor(sm_img->extent_2d().width, sm_img->extent_2d().height);
      if (depth_bias_enabled_) {
        vkCmdSetDepthBias(cmd.cmd(), depth_bias_constant_factor_, 0.0f, depth_bias_slope_factor_);
      } else {
        vkCmdSetDepthBias(cmd.cmd(), 0, 0.0f, 0);
      }

      {
        TracyVkZone(cmd.get_tracy_ctx(), cmd.cmd(), "opaque");
        vkCmdBindPipeline(cmd.cmd(), VK_PIPELINE_BIND_POINT_GRAPHICS,
                          PipelineManager::get().get(shadow_depth_pipline_)->pipeline);
        draw_fn_(cmd, light_matrices_[i], false, i);
      }
      {
        TracyVkZone(cmd.get_tracy_ctx(), cmd.cmd(), "opaque_alpha_mask");
        if (alpha_cutout_enabled_) {
          vkCmdBindPipeline(cmd.cmd(), VK_PIPELINE_BIND_POINT_GRAPHICS,
                            PipelineManager::get().get(shadow_depth_alpha_mask_pipline_)->pipeline);
        }
        draw_fn_(cmd, light_matrices_[i], true, i);
      }
      vkCmdEndRenderingKHR(cmd.cmd());
    }
    init::end_debug_utils_label(cmd.cmd());
  });
}

void CSM::debug_shadow_pass(RenderGraph& rg, const vk2::Sampler& linear_sampler) {
  if (debug_render_enabled_) {
    auto& pass = rg.add_pass("debug_csm");
    auto shadow_map_debug_img_handle =
        pass.add("shadow_map_debug_img",
                 AttachmentInfo{.size_class = SizeClass::Absolute,
                                .dims = {shadow_map_res_, 1},
                                .format = Format::R16G16B16A16Sfloat},
                 Access::ColorWrite);
    pass.add("shadow_map_img", Access::FragmentRead);
    pass.set_execute_fn([this, &linear_sampler, &rg, shadow_map_debug_img_handle](CmdEncoder& cmd) {
      auto* tex = rg.get_texture(shadow_map_debug_img_handle);
      VkRenderingAttachmentInfo color_attachment = init::rendering_attachment_info(
          tex->view().view(), VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, nullptr);
      VkRenderingInfo render_info =
          init::rendering_info(tex->extent_2d(), &color_attachment, nullptr, nullptr);

      vkCmdBeginRenderingKHR(cmd.cmd(), &render_info);
      cmd.set_viewport_and_scissor(tex->extent_2d().width, tex->extent_2d().height);

      assert(debug_cascade_idx_ >= 0 && (u32)debug_cascade_idx_ < cascade_count_);

      PipelineManager::get().bind_graphics(cmd.cmd(), depth_debug_pipeline_);

      struct {
        u32 tex_idx;
        u32 sampler_idx;
        u32 array_idx;
      } pc{get_device().get_image(shadow_map_img_)->view().sampled_img_resource().handle,
           linear_sampler.resource_info.handle, static_cast<u32>(debug_cascade_idx_)};
      cmd.push_constants(sizeof(pc), &pc);
      vkCmdDraw(cmd.cmd(), 3, 1, 0, 0);
      vkCmdEndRenderingKHR(cmd.cmd());
    });
  }
}
}  // namespace gfx
