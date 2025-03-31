#pragma once

#include <functional>

#include "vk2/PipelineManager.hpp"
#include "vk2/SamplerCache.hpp"
#include "vk2/Texture.hpp"

class StateTracker;
class CSM {
 public:
  using DrawFunc = std::function<void(const mat4& vp)>;
  explicit CSM(VkPipelineLayout pipeline_layout);
  void debug_shadow_pass(StateTracker& state, VkCommandBuffer cmd,
                         const vk2::Sampler& linear_sampler);
  void render(StateTracker& state, VkCommandBuffer cmd, const mat4& cam_view, vec3 light_dir,
              float aspect_ratio, float fov_deg, const DrawFunc& draw);
  void on_imgui(VkDescriptorSet imgui_render_set);

  [[nodiscard]] const vk2::Texture& get_debug_img() const { return shadow_map_debug_img_; }

 private:
  vk2::PipelineHandle shadow_depth_pipline_;
  vk2::PipelineHandle depth_debug_pipeline_;

  uvec2 shadow_map_res_{2048};
  u32 cascade_count_{4};
  vk2::Texture shadow_map_img_;
  vk2::Texture shadow_map_debug_img_;
  VkPipelineLayout pipeline_layout_;
  bool shadow_map_debug_enabled_{true};
  static constexpr u32 max_cascade_levels{5};
  std::array<mat4, max_cascade_levels> light_matrices_;
  std::array<std::optional<vk2::TextureView>, max_cascade_levels> shadow_map_img_views_;
  vk2::Sampler shadow_sampler_;
  i32 debug_cascade_idx_{0};
  float shadow_z_near_{.1};
  float shadow_z_far_{1000};
  float cascade_linear_factor_{.0};
  float z_mult_{10.f};
};
