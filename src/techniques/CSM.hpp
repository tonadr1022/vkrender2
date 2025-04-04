#pragma once

#include <functional>

#include "vk2/Buffer.hpp"
#include "vk2/PipelineManager.hpp"
#include "vk2/SamplerCache.hpp"
#include "vk2/Texture.hpp"

class StateTracker;
class CSM {
 public:
  struct ShadowData {
    std::array<mat4, 5> light_space_matrices;
    vec4 biases;  // min = x, max = y, pcf scale, z, z_far: w
    vec4 cascade_levels;
    uvec4 settings;
  };

  using DrawFunc = std::function<void(const mat4& vp)>;
  explicit CSM(VkPipelineLayout pipeline_layout);
  void debug_shadow_pass(StateTracker& state, VkCommandBuffer cmd,
                         const vk2::Sampler& linear_sampler);
  void render(StateTracker& state, VkCommandBuffer cmd, u32 frame_num, const mat4& cam_view,
              vec3 light_dir, float aspect_ratio, float fov_deg, const DrawFunc& draw);
  void on_imgui(VkSampler sampler);

  [[nodiscard]] const vk2::Texture& get_debug_img() const { return shadow_map_debug_img_; }
  [[nodiscard]] const vk2::Buffer& get_shadow_data_buffer(u32 frame_num) const {
    return shadow_data_bufs_[frame_num % shadow_data_bufs_.size()];
  }
  [[nodiscard]] const vk2::Sampler& get_shadow_sampler() const { return shadow_sampler_; }
  [[nodiscard]] const vk2::Texture& get_shadow_img() const { return shadow_map_img_; }

 private:
  vk2::PipelineHandle shadow_depth_pipline_;
  vk2::PipelineHandle depth_debug_pipeline_;

  uvec2 shadow_map_res_{};
  u32 cascade_count_{4};
  std::array<vk2::Buffer, 2> shadow_data_bufs_;
  vk2::Texture shadow_map_img_;
  vk2::Texture shadow_map_debug_img_;
  VkPipelineLayout pipeline_layout_;
  bool debug_render_enabled_{false};
  VkDescriptorSet imgui_set_{};
  static constexpr u32 max_cascade_levels{5};
  std::array<mat4, max_cascade_levels> light_matrices_;
  std::array<std::optional<vk2::TextureView>, max_cascade_levels> shadow_map_img_views_;
  vk2::Sampler shadow_sampler_;
  i32 debug_cascade_idx_{0};
  float shadow_z_near_{.1};
  float shadow_z_far_{225};
  float depth_bias_constant_factor_{1.25f};
  float depth_bias_slope_factor_{1.75f};
  bool depth_bias_enabled_{true};
  float pcf_scale_{1.};
  float min_bias_{0.001};
  float max_bias_{0.001};
  bool pcf_{true};
  float cascade_linear_factor_{.0};
  float z_mult_{2.75};
};
