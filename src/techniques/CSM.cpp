#include "CSM.hpp"

#include <volk.h>
#include <vulkan/vulkan_core.h>

#include <tracy/Tracy.hpp>

#include "AABB.hpp"
#include "StateTracker.hpp"
#include "glm/ext/matrix_transform.hpp"
#include "imgui.h"
#include "imgui_impl_vulkan.h"
#include "util/CVar.hpp"
#include "vk2/Initializers.hpp"
#include "vk2/Rendering.hpp"

using namespace vk2;

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

AutoCVarFloat z_pad{"z_pad", "z_padding", .5, CVarFlags::EditFloatDrag};

// https://github.com/walbourn/directx-sdk-samples/blob/main/CascadedShadowMaps11/CascadedShadowMaps11.cpp
mat4 calc_light_space_matrix(const mat4& cam_view, const mat4& proj, vec3 light_dir, float) {
  vec3 center{};
  std::array<vec4, 8> corners;
  calc_frustum_corners_world_space(corners, proj * cam_view);
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
  mat4 light_proj = glm::orthoRH_ZO(min.x, max.x, max.y, min.y, min.z, max.z);
  return light_proj * light_view;
}

// TODO: calculate near and far based on scene AABB:
// https://github.com/walbourn/directx-sdk-samples/blob/main/CascadedShadowMaps11/CascadedShadowsManager.cpp
mat4 calc_light_space_matrix(const mat4& cam_view, const mat4& proj, vec3 light_dir, float z_mult,
                             u32 shadow_map_size) {
  if (!stable_light_view.get()) {
    return calc_light_space_matrix(cam_view, proj, light_dir, z_mult);
  }

  // get center of frustum corners of cascade
  vec3 center{};
  std::array<vec4, 8> corners;
  calc_frustum_corners_world_space(corners, proj * cam_view);
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

  mat4 shadow_cam_vp = glm::orthoRH_ZO(min.x, max.x, max.y, min.y, min.z, max.z) * light_space_view;
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

void calc_csm_light_space_matrices(std::span<mat4> matrices, std::span<float> levels,
                                   const mat4& cam_view, vec3 light_dir, float z_mult,
                                   float fov_deg, float aspect, float cam_near, float cam_far,
                                   u32 shadow_map_res) {
  assert(matrices.size() && levels.size() && matrices.size() - 1 == levels.size());
  auto get_proj = [&](float near, float far) {
    auto mat = glm::perspective(glm::radians(fov_deg), aspect, near, far);
    mat[1][1] *= -1;
    return mat;
  };

  vec3 dir = -glm::normalize(light_dir);
  matrices[0] =
      calc_light_space_matrix(cam_view, get_proj(cam_near, levels[0]), dir, z_mult, shadow_map_res);
  for (u32 i = 1; i < matrices.size() - 1; i++) {
    matrices[i] = calc_light_space_matrix(cam_view, get_proj(levels[i - 1], levels[i]), dir, z_mult,
                                          shadow_map_res);
  }
  matrices[matrices.size() - 1] = calc_light_space_matrix(
      cam_view, get_proj(levels[levels.size() - 1], cam_far), dir, z_mult, shadow_map_res);
}

}  // namespace

CSM::CSM(VkPipelineLayout pipeline_layout)
    : shadow_map_res_(uvec2{4096}),
      shadow_data_bufs_(
          {create_storage_buffer(sizeof(ShadowData)), create_storage_buffer(sizeof(ShadowData))}),
      shadow_map_img_(
          vk2::Texture{TextureCreateInfo{.view_type = VK_IMAGE_VIEW_TYPE_2D_ARRAY,
                                         .format = VK_FORMAT_D32_SFLOAT,
                                         .extent = {shadow_map_res_.x, shadow_map_res_.y, 1},
                                         .mip_levels = 1,
                                         .array_layers = cascade_count_,
                                         .usage = TextureUsage::General}}),
      shadow_map_debug_img_(
          vk2::Texture{TextureCreateInfo{.view_type = VK_IMAGE_VIEW_TYPE_2D,
                                         .format = VK_FORMAT_R16G16B16A16_SFLOAT,
                                         .extent = {shadow_map_res_.x, shadow_map_res_.y, 1},
                                         .mip_levels = 1,
                                         .array_layers = 1,
                                         .usage = TextureUsage::General}}),
      pipeline_layout_(pipeline_layout) {
  ZoneScoped;
  for (u32 i = 0; i < cascade_count_; i++) {
    shadow_map_img_views_[i] =
        vk2::TextureView{shadow_map_img_, vk2::TextureViewCreateInfo{
                                              shadow_map_img_.format(),
                                              VkImageSubresourceRange{
                                                  .aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT,
                                                  .baseMipLevel = 0,
                                                  .levelCount = 1,
                                                  .baseArrayLayer = i,
                                                  .layerCount = 1,
                                              },
                                          }};
  }
  shadow_sampler_ = SamplerCache::get().get_or_create_sampler(VkSamplerCreateInfo{
      .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
      .magFilter = VK_FILTER_LINEAR,
      .minFilter = VK_FILTER_LINEAR,
      .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
      .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
      .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
      .borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE

  });
  shadow_depth_pipline_ = PipelineManager::get().load_graphics_pipeline(GraphicsPipelineCreateInfo{
      .vertex_path = "shadow_depth.vert",
      .fragment_path = "shadow_depth.frag",
      .layout = pipeline_layout_,
      .rendering = {.depth_format = shadow_map_img_.format()},
      .rasterization = {.depth_clamp = true, .depth_bias = true},
      .depth_stencil = GraphicsPipelineCreateInfo::depth_enable(true, CompareOp::Less),
      .dynamic_state = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR,
                        VK_DYNAMIC_STATE_DEPTH_BIAS},
  });
  depth_debug_pipeline_ = PipelineManager::get().load_graphics_pipeline(GraphicsPipelineCreateInfo{
      .vertex_path = "fullscreen_quad.vert",
      .fragment_path = "debug/depth_debug.frag",
      .layout = pipeline_layout_,
      .rendering = {.color_formats = {shadow_map_debug_img_.format()}, .color_formats_cnt = 1},
      .rasterization = {.cull_mode = CullMode::Front},
      .depth_stencil = GraphicsPipelineCreateInfo::depth_disable(),
  });
}

void CSM::debug_shadow_pass(StateTracker& state, VkCommandBuffer cmd,
                            const vk2::Sampler& linear_sampler) {
  if (!debug_render_enabled_) return;
  state.transition(shadow_map_debug_img_.image(), VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                   VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT,
                   VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
  state.flush_barriers();

  VkRenderingAttachmentInfo color_attachment = init::rendering_attachment_info(
      shadow_map_debug_img_.view().view(), VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, nullptr);

  VkExtent2D extent = shadow_map_debug_img_.extent_2d();
  VkRenderingInfo render_info = init::rendering_info(extent, &color_attachment, nullptr, nullptr);
  vkCmdBeginRenderingKHR(cmd, &render_info);
  set_viewport_and_scissor(cmd, extent);
  assert(debug_cascade_idx_ >= 0 && (u32)debug_cascade_idx_ < cascade_count_);

  PipelineManager::get().bind_graphics(cmd, depth_debug_pipeline_);

  struct {
    u32 tex_idx;
    u32 sampler_idx;
    u32 array_idx;
  } pc{shadow_map_img_.view().sampled_img_resource().handle, linear_sampler.resource_info.handle,
       static_cast<u32>(debug_cascade_idx_)};
  vkCmdPushConstants(cmd, pipeline_layout_, VK_SHADER_STAGE_ALL, 0, sizeof(pc), &pc);
  vkCmdDraw(cmd, 3, 1, 0, 0);
  vkCmdEndRenderingKHR(cmd);
  state.transition(shadow_map_debug_img_.image(), VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                   VK_ACCESS_2_SHADER_READ_BIT, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
  state.flush_barriers();
}

void CSM::render(StateTracker& state, VkCommandBuffer cmd, u32 frame_num, const mat4& cam_view,
                 vec3 light_dir, float aspect_ratio, float fov_deg, const DrawFunc& draw,
                 const AABB& aabb, vec3 view_pos) {
  // draw shadows
  float shadow_z_far = shadow_z_far_;
  {
    std::array<vec3, 8> aabb_corners;
    aabb.get_corners(aabb_corners);
    float max_dist = std::numeric_limits<float>::lowest();
    for (auto c : aabb_corners) {
      max_dist = std::max(glm::distance(view_pos, c), max_dist);
    }
    shadow_z_far = max_dist;
    shadow_z_far_ = max_dist;
  }

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
                                std::span(levels.data(), cascade_count_ - 1), cam_view, light_dir,
                                z_mult_, fov_deg, aspect_ratio, shadow_z_near_, shadow_z_far,
                                shadow_map_res_.x);
  ShadowData data{.biases = {0., 0., 0., shadow_z_far}};
  for (u32 i = 0; i < cascade_count_; i++) {
    data.light_space_matrices[i] = light_matrices_[i];
  }
  for (u32 i = 0; i < cascade_count_ - 1; i++) {
    data.cascade_levels[i] = levels[i];
  }
  data.settings = {};
  if (pcf_) {
    data.settings.x |= 0x1;
  }
  data.biases.x = min_bias_;
  data.biases.y = max_bias_;
  data.biases.z = pcf_scale_;
  data.settings.w = cascade_count_;

  auto& buf = shadow_data_bufs_[frame_num % shadow_data_bufs_.size()];
  state.buffer_barrier(buf.buffer(), VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                       VK_ACCESS_2_TRANSFER_WRITE_BIT);
  state.flush_barriers();
  vkCmdUpdateBuffer(cmd, buf.buffer(), 0, sizeof(ShadowData), &data);
  state.transition(
      shadow_map_img_.image(),
      VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
      VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
      VK_IMAGE_ASPECT_DEPTH_BIT);
  state.flush_barriers();

  for (u32 i = 0; i < cascade_count_; i++) {
    VkClearValue depth_clear{.depthStencil = {.depth = 1.f}};
    VkRenderingAttachmentInfo depth_att = init::rendering_attachment_info(
        shadow_map_img_views_[i]->view(), VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, &depth_clear);
    auto rendering_info = init::rendering_info(shadow_map_img_.extent_2d(), nullptr, &depth_att);
    vkCmdBeginRenderingKHR(cmd, &rendering_info);
    set_viewport_and_scissor(cmd, shadow_map_img_.extent_2d());
    if (depth_bias_enabled_) {
      vkCmdSetDepthBias(cmd, depth_bias_constant_factor_, 0.0f, depth_bias_slope_factor_);
    }
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                      PipelineManager::get().get(shadow_depth_pipline_)->pipeline);
    draw(light_matrices_[i]);
    vkCmdEndRenderingKHR(cmd);
  }
}
void CSM::on_imgui(VkSampler sampler) {
  if (!imgui_set_) {
    imgui_set_ = ImGui_ImplVulkan_AddTexture(sampler, shadow_map_debug_img_.view().view(),
                                             VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
  }
  ImGui::Checkbox("shadow map debug", &debug_render_enabled_);
  ImGui::SliderFloat("Z mult", &z_mult_, 0.0, 50.f);
  ImGui::DragFloat("Shadow z far", &shadow_z_far_, 1., 0.f, 10000.f);
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
  if (ImGui::TreeNode("shadow map")) {
    ImGui::SliderInt("view level", &debug_cascade_idx_, 0, cascade_count_ - 1);
    if (debug_render_enabled_) {
      ImVec2 window_size = ImGui::GetContentRegionAvail();
      float scale_width = window_size.x / shadow_map_debug_img_.extent_2d().width;
      float scaled_height = shadow_map_debug_img_.extent_2d().height * scale_width;
      ImGui::Image(reinterpret_cast<ImTextureID>(imgui_set_),
                   ImVec2(window_size.x * .8f, scaled_height * .8f));
    }
    ImGui::TreePop();
  }
}
