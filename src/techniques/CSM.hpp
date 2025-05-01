#pragma once

#include <functional>

#include "CommandEncoder.hpp"
#include "Types.hpp"
#include "vk2/PipelineManager.hpp"
#include "vk2/Pool.hpp"
#include "vk2/SamplerCache.hpp"
#include "vk2/Texture.hpp"

struct AABB;

namespace gfx {
class StateTracker;

struct RenderGraph;
struct RenderGraphPass;
class BaseRenderer;
class CSM {
 public:
  using DrawFunc =
      std::function<void(CmdEncoder&, const mat4& vp, bool opaque_alpha, u32 cascade_i)>;
  using AddRenderDependenciesFunc = std::function<void(RenderGraphPass& pass)>;
  explicit CSM(BaseRenderer* renderer, DrawFunc draw_fn, AddRenderDependenciesFunc add_deps_fn);
  void add_pass(RenderGraph& rg);
  static constexpr u32 max_cascade_levels{4};

  struct ShadowData {
    std::array<mat4, max_cascade_levels> light_space_matrices;
    vec4 biases;  // min = x, max = y, pcf scale, z, z_far: w
    vec4 cascade_levels;
    uvec4 settings;
  };

  [[nodiscard]] const mat4& get_cascade_proj_mat(u32 cascade_level) const {
    return light_proj_matrices_[cascade_level];
  }

  void debug_shadow_pass(RenderGraph& rg, const vk2::Sampler& linear_sampler);
  void prepare_frame(RenderGraph& rg, u32 frame_num, const mat4& cam_view, vec3 light_dir,
                     float aspect_ratio, float fov_deg, const AABB& aabb, vec3 view_pos);
  void on_imgui();

  // [[nodiscard]] const vk2::Image& get_debug_img() const { return shadow_map_debug_img_; }
  [[nodiscard]] BufferHandle get_shadow_data_buffer(u32 frame_num) const {
    return shadow_data_bufs_[frame_num % shadow_data_bufs_.size()].handle;
  }
  [[nodiscard]] const AttachmentInfo& get_shadow_map_att_info() const {
    return shadow_map_img_att_info_;
  }

  [[nodiscard]] ImageHandle get_shadow_map_img() const { return shadow_map_img_; }

  [[nodiscard]] u32 get_num_cascade_levels() const { return cascade_count_; }
  void imgui_pass(CmdEncoder& cmd, const vk2::Sampler& sampler, const vk2::Image& tex);

  [[nodiscard]] bool get_debug_render_enabled() const { return debug_render_enabled_; }

  using LightMatrixArray = std::array<mat4, max_cascade_levels>;
  [[nodiscard]] const LightMatrixArray& get_light_matrices() const { return light_matrices_; }

 private:
  ImageHandle shadow_map_img_;
  ShadowData data_{};
  std::array<mat4, max_cascade_levels> light_proj_matrices_;
  DrawFunc draw_fn_;
  AddRenderDependenciesFunc add_deps_fn_;
  AttachmentInfo shadow_map_img_att_info_;
  vk2::PipelineHandle shadow_depth_pipline_;
  vk2::PipelineHandle shadow_depth_alpha_mask_pipline_;
  vk2::PipelineHandle depth_debug_pipeline_;
  uvec2 shadow_map_res_{};
  u32 cascade_count_{4};
  std::array<Holder<BufferHandle>, 2> shadow_data_bufs_;
  Format debug_shadow_img_format_{Format::R16G16B16A16Sfloat};

  VkDescriptorSet imgui_set_{};
  VkImage curr_debug_img_{};
  uvec2 curr_debug_img_size_{};
  LightMatrixArray light_matrices_;
  ImageHandle curr_shadow_map_img_;
  std::array<Holder<ImageViewHandle>, max_cascade_levels> shadow_map_img_views_;
  BaseRenderer* renderer_{};
  i32 debug_cascade_idx_{0};
  float shadow_z_near_{.1};
  float shadow_z_far_{225};
  float depth_bias_constant_factor_{.001f};
  float depth_bias_slope_factor_{2.5f};
  float pcf_scale_{1.};
  float min_bias_{0.001};
  float max_bias_{0.001};
  float cascade_linear_factor_{.6};
  float z_mult_{2.75};
  bool aabb_based_z_far_{true};
  bool depth_bias_enabled_{true};
  bool pcf_{true};
  bool alpha_cutout_enabled_{true};
  bool debug_render_enabled_{false};
};
}  // namespace gfx
