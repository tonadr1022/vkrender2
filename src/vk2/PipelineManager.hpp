#pragma once

#include <vulkan/vulkan_core.h>

#include <cstring>
#include <filesystem>
#include <future>
#include <shared_mutex>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "Types.hpp"
#include "vk2/ShaderCompiler.hpp"

namespace gfx::vk2 {

struct Pipeline {
  VkPipeline pipeline;
  VkPipelineLayout layout;
  bool owns_layout{};
};

// TODO: vertex input, tesselation
// TODO: configurable dynamic state
struct GraphicsPipelineCreateInfo {
  struct Rasterization {
    PolygonMode polygon_mode{PolygonMode::Fill};
    CullMode cull_mode{CullMode::Back};
    FrontFace front_face{FrontFace::CounterClockwise};
    bool depth_clamp{false};
    bool depth_bias{false};
    bool rasterize_discard_enable{false};
    float line_width{1.};
    float depth_bias_constant_factor{};
    float depth_bias_clamp{};
    float depth_bias_slope_factor{};
  };
  struct ColorBlendAttachment {
    bool enable{false};
    BlendFactor src_color_factor;
    BlendFactor dst_color_blend_factor;
    BlendOp color_blend_op;
    BlendFactor src_alpha_factor;
    BlendFactor dst_alpha_blend_factor;
    BlendOp alpha_blend_op;
    ColorComponentFlags color_write_mask{ColorComponentRBit | ColorComponentGBit |
                                         ColorComponentBBit | ColorComponentABit};
  };
  struct Blend {
    bool logic_op_enable{false};
    LogicOp logic_op{LogicOpCopy};
    // TODO: fixed vector
    std::vector<ColorBlendAttachment> attachments;
    float blend_constants[4]{};
  };
  struct Multisample {
    // TODO: flesh out, for now not caring about it
    SampleCountFlagBits rasterization_samples{SampleCount1Bit};
    float min_sample_shading{0.};
    bool sample_shading_enable{false};
    bool alpha_to_coverage_enable{false};
    bool alpha_to_one_enable{false};
  };

  struct StencilOpState {
    StencilOp fail_op{};
    StencilOp pass_op{};
    StencilOp depth_fail_op{};
    CompareOp compare_op{};
    u32 compare_mask{};
    u32 write_mask{};
    u32 reference{};
  };

  struct RenderingInfo {
    std::array<VkFormat, 5> color_formats{};
    VkFormat depth_format{VK_FORMAT_UNDEFINED};
    VkFormat stencil_format{VK_FORMAT_UNDEFINED};
  };
  struct DepthStencil {
    StencilOpState stencil_front{};
    StencilOpState stencil_back{};
    float min_depth_bounds{0.};
    float max_depth_bounds{1.};
    bool depth_test_enable{false};
    bool depth_write_enable{false};
    CompareOp depth_compare_op{CompareOp::Never};
    bool depth_bounds_test_enable{false};
    bool stencil_test_enable{false};
  };

  std::vector<ShaderCreateInfo> shaders;

  // TODO: move elsewhere
  VkPipelineLayout layout{};
  PrimitiveTopology topology{PrimitiveTopology::TriangleList};
  RenderingInfo rendering{};
  Rasterization rasterization{};
  Blend blend{};
  Multisample multisample{};
  DepthStencil depth_stencil{};
  std::vector<VkDynamicState> dynamic_state;

  static constexpr DepthStencil depth_disable() { return DepthStencil{.depth_test_enable = false}; }
  static constexpr DepthStencil depth_enable(bool write_enable, CompareOp op) {
    return DepthStencil{
        .depth_test_enable = true, .depth_write_enable = write_enable, .depth_compare_op = op};
  }
  std::string name;
};
struct ComputePipelineCreateInfo {
  ShaderCreateInfo info;
  std::string name;
};

// TODO: on start up, check last write times for shader dir files that are .h or glsl. compare them
// with a cached list written to disk check whether any pipeline uses them and update if needed.
// pass the dirty ones to pipelne compilation

class PipelineManager {
 public:
  static PipelineManager &get();
  static void init(VkDevice device, std::filesystem::path shader_dir, bool hot_reload,
                   VkPipelineLayout default_layout = nullptr);
  static void shutdown();
  [[nodiscard]] u64 num_pipelines();
  void bind_graphics(VkCommandBuffer cmd, PipelineHandle handle);
  void bind_compute(VkCommandBuffer cmd, PipelineHandle handle);

  [[nodiscard]] PipelineHandle load(const ComputePipelineCreateInfo &info);
  [[nodiscard]] PipelineHandle load(GraphicsPipelineCreateInfo info);

  Pipeline *get(PipelineHandle handle);

  void reload_shaders();
  void reload_pipeline(PipelineHandle handle, bool force);
  void on_imgui();

 private:
  void reload_pipeline_unsafe(PipelineHandle handle, bool force);
  // concurrent
  void on_dirty_files(std::span<std::filesystem::path> dirty_files);
  // end concurrent

  explicit PipelineManager(VkDevice device, std::filesystem::path shader_dir, bool hot_reload,
                           VkPipelineLayout default_layout);
  ~PipelineManager();
  [[nodiscard]] std::string get_shader_path(const std::string &path) const;

  enum class PipelineType : u8 {
    Graphics,
    Compute,
    Mesh,
  };

  struct PipelineAndMetadata {
    Pipeline pipeline;
    std::vector<std::string> shader_paths;
    PipelineType type;
  };

  std::unordered_map<std::string, std::unordered_set<PipelineHandle>>
      shader_name_to_used_pipelines_;
  std::shared_mutex mtx_;
  std::unordered_map<PipelineHandle, PipelineAndMetadata> pipelines_;
  std::unordered_map<PipelineHandle, GraphicsPipelineCreateInfo> graphics_pipeline_infos_;
  std::unordered_map<PipelineHandle, ShaderCreateInfo> compute_pipeline_infos_;

  std::filesystem::path shader_dir_;
  size_t get_pipeline_hash(const GraphicsPipelineCreateInfo &info);
  struct LoadPipelineResult {
    VkPipeline pipeline;
    size_t hash;
  };
  LoadPipelineResult load_graphics_pipeline_impl(const GraphicsPipelineCreateInfo &info,
                                                 bool force);
  LoadPipelineResult load_compute_pipeline_impl(const ShaderCreateInfo &info, bool force);

  ShaderManager shader_manager_;
  std::filesystem::path cache_path_;
  VkPipelineLayout default_pipeline_layout_{};
  VkDevice device_;
};

struct PipelineTask {
  std::future<void> future;
};

}  // namespace gfx::vk2
