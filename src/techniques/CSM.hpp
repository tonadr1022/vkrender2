#pragma once

#include <functional>

#include "CommandEncoder.hpp"
#include "Types.hpp"
#include "vk2/Device.hpp"
#include "vk2/PipelineManager.hpp"
#include "vk2/SamplerCache.hpp"
#include "vk2/Texture.hpp"

struct AABB;

namespace gfx {
class StateTracker;

struct RenderGraph;
class BaseRenderer;
class CSM {
 public:
  using DrawFunc = std::function<void(CmdEncoder&, const mat4& vp)>;
  explicit CSM(BaseRenderer* renderer, DrawFunc draw_fn);
  void add_pass(RenderGraph& rg);

  struct ShadowData {
    std::array<mat4, 5> light_space_matrices;
    vec4 biases;  // min = x, max = y, pcf scale, z, z_far: w
    vec4 cascade_levels;
    uvec4 settings;
  };

  void debug_shadow_pass(StateTracker& state, VkCommandBuffer cmd,
                         const vk2::Sampler& linear_sampler);
  void prepare_frame(RenderGraph& rg, u32 frame_num, const mat4& cam_view, vec3 light_dir,
                     float aspect_ratio, float fov_deg, const AABB& aabb, vec3 view_pos);
  void on_imgui(VkSampler sampler);

  [[nodiscard]] const vk2::Image& get_debug_img() const { return shadow_map_debug_img_; }
  [[nodiscard]] vk2::BufferHandle get_shadow_data_buffer(u32 frame_num) const {
    return shadow_data_bufs_[frame_num % shadow_data_bufs_.size()].handle;
  }
  [[nodiscard]] const AttachmentInfo& get_shadow_map_att_info() const {
    return shadow_map_img_att_info_;
  }

  [[nodiscard]] vk2::ImageHandle get_shadow_map_img() const { return shadow_map_img_; }

 private:
  vk2::ImageHandle shadow_map_img_;
  ShadowData data_{};
  DrawFunc draw_fn_;
  AttachmentInfo shadow_map_img_att_info_;
  vk2::PipelineHandle shadow_depth_pipline_;
  vk2::PipelineHandle shadow_depth_alpha_mask_pipline_;
  vk2::PipelineHandle depth_debug_pipeline_;
  uvec2 shadow_map_res_{};
  u32 cascade_count_{4};
  std::array<vk2::Holder<vk2::BufferHandle>, 2> shadow_data_bufs_;
  vk2::Image shadow_map_debug_img_;
  VkPipelineLayout pipeline_layout_;
  bool debug_render_enabled_{false};
  VkDescriptorSet imgui_set_{};
  static constexpr u32 max_cascade_levels{5};
  std::array<mat4, max_cascade_levels> light_matrices_;
  vk2::ImageHandle curr_shadow_map_img_;
  std::array<vk2::Holder<vk2::ImageViewHandle>, max_cascade_levels> shadow_map_img_views_;
  i32 debug_cascade_idx_{0};
  float shadow_z_near_{.1};
  float shadow_z_far_{225};
  float depth_bias_constant_factor_{.001f};
  float depth_bias_slope_factor_{2.5f};
  bool depth_bias_enabled_{true};
  bool aabb_based_z_far_{true};
  float pcf_scale_{1.};
  float min_bias_{0.001};
  float max_bias_{0.001};
  bool pcf_{true};
  float cascade_linear_factor_{.6};
  float z_mult_{2.75};
  BaseRenderer* renderer_{};
};
}  // namespace gfx
