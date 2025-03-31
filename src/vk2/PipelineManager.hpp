#pragma once

#include <vulkan/vulkan_core.h>

#include <cstring>
#include <filesystem>
#include <unordered_map>
#include <vector>

#include "vk2/Handle.hpp"
#include "vk2/ShaderCompiler.hpp"

enum ColorComponentFlagBits : u8 {
  ColorComponentRBit = 0x00000001,
  ColorComponentGBit = 0x00000002,
  ColorComponentBBit = 0x00000004,
  ColorComponentABit = 0x00000008,
};
using ColorComponentFlags = u32;

enum class StencilOp : u8 {
  Keep = 0,
  Zero,
  Replace,
  IncrementAndClamp,
  DecrementAndClamp,
  IncrementAndWrap,
  DecrementAndWrap,
};
enum class CompareOp : u8 {
  Never = 0,
  Less,
  Equal,
  LessOrEqual,
  Greater,
  NotEqual,
  GreaterOrEqual,
  Always
};

enum class CullMode : u8 {
  None = 0,
  Front,
  Back,
  FrontAndBack,
};
enum class PolygonMode : u8 {
  Fill = 0,
  Line,
  Point,
};

enum class PrimitiveTopology : u8 {
  PointList,
  LineList,
  LineStrip,
  TriangleList,
  TriangleStrip,
  TriangleFan,
  PatchList
};
enum class BlendFactor : u8 {
  Zero = 0,
  One = 1,
  SrcColor = 2,
  OneMinusSrcColor = 3,
  DstColor = 4,
  OneMinusDstColor = 5,
  SrcAlpha = 6,
  OneMinusSrcAlpha = 7,
  DstAlpha = 8,
  OneMinusDstAlpha = 9,
  ConstantColor = 10,
  OneMinusConstantColor = 11,
  ConstantAlpha = 12,
  OneMinusConstantAlpha = 13,
  SRC_ALPHA_SATURATE = 14,
  Src1Color = 15,
  OneMinusSrc1Color = 16,
  Src1Alpha = 17,
  OneMinusSrc1Alpha = 18,
};

enum class BlendOp : u32 {
  Add = 0,
  Subtract = 1,
  ReverseSubtract = 2,
  Min = 3,
  Max = 4,
  ZeroExt = 1000148000,
  SrcExt = 1000148001,
  DstExt = 1000148002,
  SrcOverExt = 1000148003,
  DstOverExt = 1000148004,
  SrcInExt = 1000148005,
  DstInExt = 1000148006,
  SrcOutExt = 1000148007,
  DstOutExt = 1000148008,
  SrcAtopExt = 1000148009,
  DstAtopExt = 1000148010,
  XorExt = 1000148011,
  MultiplyExt = 1000148012,
  ScreenExt = 1000148013,
  OverlayExt = 1000148014,
  DarkenExt = 1000148015,
  LightenExt = 1000148016,
  ColorDodgeExt = 1000148017,
  ColorBurnExt = 1000148018,
  HardLightExt = 1000148019,
  SoftLightExt = 1000148020,
  DifferenceExt = 1000148021,
  ExclusionExt = 1000148022,
  InvertExt = 1000148023,
  InvertRgbExt = 1000148024,
  LinearDodgeExt = 1000148025,
  LinearBurnExt = 1000148026,
  VividLightExt = 1000148027,
  LinearLightExt = 1000148028,
  PinLightExt = 1000148029,
  HardMixExt = 1000148030,
  HslHueExt = 1000148031,
  HslSaturationExt = 1000148032,
  HslColorExt = 1000148033,
  HslLuminosityExt = 1000148034,
  PlusExt = 1000148035,
  PlusClampedExt = 1000148036,
  PlusClampedAlphaExt = 1000148037,
  PlusDarkerExt = 1000148038,
  MinusExt = 1000148039,
  MinusClampedExt = 1000148040,
  ContrastExt = 1000148041,
  InvertOvgExt = 1000148042,
  RedExt = 1000148043,
  GreenExt = 1000148044,
  BlueExt = 1000148045,
  MaxEnum = 0x7FFFFFFF
};
enum LogicOp : u8 {
  LogicOpClear = 0,
  LogicOpAnd = 1,
  LogicOpAndReverse = 2,
  LogicOpCopy = 3,
  LogicOpAndInverted = 4,
  LogicOpNoOp = 5,
  LogicOpXor = 6,
  LogicOpOr = 7,
  LogicOpNor = 8,
  LogicOpEquivalent = 9,
  LogicOpInvert = 10,
  LogicOpOrReverse = 11,
  LogicOpCopyInverted = 12,
  LogicOpOrInverted = 13,
  LogicOpNand = 14,
  LogicOpSet = 15,
};

enum SampleCountFlagBits : u8 {
  SampleCount1Bit = 0x00000001,
  SampleCount2Bit = 0x00000002,
  SampleCount4Bit = 0x00000004,
  SampleCount8Bit = 0x00000008,
  SampleCount16Bit = 0x00000010,
  SampleCount32Bit = 0x00000020,
  SampleCount64Bit = 0x00000040,
};
using SampleCountFlags = u32;

enum class FrontFace : u8 { CounterClockwise = 0, Clockwise };

namespace vk2 {

struct Pipeline {
  VkPipeline pipeline;
  VkPipelineLayout layout;
  bool owns_layout{};
};

VK2_DEFINE_HANDLE_WITH_NAME(Pipeline, PipelineAndMetadata);

struct ComputePipelineCreateInfo {
  std::filesystem::path path;
  VkPipelineLayout layout{};
  const char *entry_point = "main";
};

// TODO: vertex input, tesselation
// TODO: configurable dynamic state
struct GraphicsPipelineCreateInfo {
  struct Rasterization {
    PolygonMode polygon_mode{PolygonMode::Fill};
    CullMode cull_mode{CullMode::None};
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
    std::initializer_list<ColorBlendAttachment> attachments;
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
    std::span<const VkFormat> color_formats;
    VkFormat depth_format{VK_FORMAT_UNDEFINED};
    VkFormat stencil_format{VK_FORMAT_UNDEFINED};
  };
  struct DepthStencil {
    StencilOpState stencil_front{};
    StencilOpState stencil_back{};
    float min_depth_bounds{};
    float max_depth_bounds{1.};
    bool depth_test_enable{false};
    bool depth_write_enable{false};
    CompareOp depth_compare_op{CompareOp::Never};
    bool depth_bounds_test_enable{false};
    bool stencil_test_enable{false};
  };

  std::filesystem::path vertex_path;
  std::filesystem::path fragment_path;
  // TODO: move elsewhere
  VkPipelineLayout layout{};
  PrimitiveTopology topology{PrimitiveTopology::TriangleList};
  RenderingInfo rendering{};
  Rasterization rasterization{};
  Blend blend{};
  Multisample multisample{};
  DepthStencil depth_stencil{};

  static constexpr DepthStencil depth_disable() { return DepthStencil{.depth_test_enable = false}; }
  static constexpr DepthStencil depth_enable(bool write_enable, CompareOp op) {
    return DepthStencil{
        .depth_test_enable = true, .depth_write_enable = write_enable, .depth_compare_op = op};
  }
};

// TODO: on start up, check last write times for shader dir files that are .h or glsl. compare them
// with a cached list written to disk check whether any pipeline uses them and update if needed.
// pass the dirty ones to pipelne compilation

class PipelineManager {
 public:
  static PipelineManager &get();
  static void init(VkDevice device, std::filesystem::path shader_dir);
  static void shutdown();

  void on_shader_update();
  void bind_graphics(VkCommandBuffer cmd, PipelineHandle handle);
  void bind_compute(VkCommandBuffer cmd, PipelineHandle handle);

  PipelineHandle load_graphics_pipeline(const std::filesystem::path &path,
                                        const char *entry_point = "main");

  // if layout is not provided, spirv is reflected to obtain layout info and create the layout.
  [[nodiscard]] PipelineHandle load_compute_pipeline(const ComputePipelineCreateInfo &info);
  [[nodiscard]] PipelineHandle load_graphics_pipeline(const GraphicsPipelineCreateInfo &info);

  Pipeline *get(PipelineHandle handle);

  void clear_module_cache();

  void destroy_pipeline(PipelineHandle handle);

 private:
  explicit PipelineManager(VkDevice device, std::filesystem::path shader_dir);
  ~PipelineManager();
  VkPipeline create_compute_pipeline(ShaderManager::LoadProgramResult &result,
                                     const char *entry_point = "main");
  [[nodiscard]] std::string get_shader_path(const std::string &path) const;

  struct PipelineAndMetadata {
    Pipeline pipeline;
    std::vector<std::string> shader_paths;
  };

  std::unordered_map<std::string, std::vector<PipelineHandle>> shader_name_to_used_pipelines_;
  std::unordered_map<PipelineHandle, PipelineAndMetadata> pipelines_;

  std::filesystem::path shader_dir_;

  ShaderManager shader_manager_;
  VkDevice device_;
};
}  // namespace vk2
