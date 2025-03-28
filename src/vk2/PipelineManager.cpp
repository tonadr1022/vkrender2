#include "PipelineManager.hpp"

#include <volk.h>
#include <vulkan/vulkan_core.h>

#include <cassert>
#include <ranges>
#include <tracy/Tracy.hpp>

#include "Logger.hpp"
#include "vk2/Hash.hpp"
#include "vk2/Initializers.hpp"
#include "vk2/ShaderCompiler.hpp"
#include "vk2/VkCommon.hpp"

namespace vk2 {

namespace {

constexpr VkStencilOpState convert_stencil_op_state(
    const GraphicsPipelineCreateInfo::StencilOpState& state) {
  return VkStencilOpState{.failOp = static_cast<VkStencilOp>(state.fail_op),
                          .passOp = static_cast<VkStencilOp>(state.pass_op),
                          .depthFailOp = static_cast<VkStencilOp>(state.depth_fail_op),
                          .compareOp = static_cast<VkCompareOp>(state.compare_op),
                          .compareMask = state.compare_mask,
                          .writeMask = state.write_mask,
                          .reference = state.reference};
}
constexpr VkLogicOp convert_logic_op(LogicOp op) { return static_cast<VkLogicOp>(op); }

constexpr VkColorComponentFlags convert_color_component_flags(ColorComponentFlags flags) {
  return flags;
}

constexpr VkBlendOp convert_blend_op(BlendOp op) { return static_cast<VkBlendOp>(op); }

constexpr VkBlendFactor convert_blend_factor(BlendFactor factor) {
  return static_cast<VkBlendFactor>(factor);
}
constexpr VkFrontFace convert_front_face(FrontFace face) {
  switch (face) {
    case FrontFace::Clockwise:
      return VK_FRONT_FACE_CLOCKWISE;
    case FrontFace::CounterClockwise:
      return VK_FRONT_FACE_COUNTER_CLOCKWISE;
  }
}
constexpr VkCullModeFlags convert_cull_mode(CullMode mode) {
  switch (mode) {
    case CullMode::None:
      return VK_CULL_MODE_NONE;
    case CullMode::Back:
      return VK_CULL_MODE_BACK_BIT;
    case CullMode::Front:
      return VK_CULL_MODE_FRONT_BIT;
    case CullMode::FrontAndBack:
      return VK_CULL_MODE_FRONT_AND_BACK;
  }
}
constexpr VkPolygonMode convert_polygon_mode(PolygonMode mode) {
  switch (mode) {
    case PolygonMode::Fill:
      return VK_POLYGON_MODE_FILL;
    case PolygonMode::Line:
      return VK_POLYGON_MODE_LINE;
    case PolygonMode::Point:
      return VK_POLYGON_MODE_POINT;
    default:
      assert(0);
      return VK_POLYGON_MODE_FILL;
  }
}

constexpr VkPrimitiveTopology convert_prim_topology(PrimitiveTopology top) {
  switch (top) {
    case PrimitiveTopology::PointList:
      return VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
    case PrimitiveTopology::LineList:
      return VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
    case PrimitiveTopology::TriangleList:
      return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    case PrimitiveTopology::LineStrip:
      return VK_PRIMITIVE_TOPOLOGY_LINE_STRIP;
    case PrimitiveTopology::TriangleStrip:
      return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP;
    case PrimitiveTopology::TriangleFan:
      return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN;
    case PrimitiveTopology::PatchList:
      assert(0);
      return VK_PRIMITIVE_TOPOLOGY_PATCH_LIST;
  }
}

constexpr VkPipelineColorBlendAttachmentState convert_color_blend_attachment(
    const GraphicsPipelineCreateInfo::ColorBlendAttachment& a) {
  return VkPipelineColorBlendAttachmentState{
      .blendEnable = a.enable,
      .srcColorBlendFactor = convert_blend_factor(a.src_color_factor),
      .dstColorBlendFactor = convert_blend_factor(a.dst_color_blend_factor),
      .colorBlendOp = convert_blend_op(a.color_blend_op),
      .srcAlphaBlendFactor = convert_blend_factor(a.src_alpha_factor),
      .dstAlphaBlendFactor = convert_blend_factor(a.dst_alpha_blend_factor),
      .alphaBlendOp = convert_blend_op(a.alpha_blend_op),
      .colorWriteMask = convert_color_component_flags(a.color_write_mask),
  };
}
}  // namespace

VkPipeline PipelineManager::create_compute_pipeline(ShaderManager::LoadProgramResult& result,
                                                    const char* entry_point) {
  ZoneScoped;
  assert(result.module_cnt == 1 && result.layout);
  VkPipelineShaderStageCreateInfo stage{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
      .stage = VK_SHADER_STAGE_COMPUTE_BIT,
      .module = result.modules[0].module,
      .pName = entry_point};
  VkComputePipelineCreateInfo create_info{.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                                          .stage = stage,
                                          .layout = result.layout};
  VkPipeline pipeline{};
  VK_CHECK(vkCreateComputePipelines(device_, nullptr, 1, &create_info, nullptr, &pipeline));
  return pipeline;
}

namespace {
PipelineManager* instance{};
}

PipelineManager& PipelineManager::get() {
  assert(instance);
  return *instance;
}
void PipelineManager::init(VkDevice device) {
  assert(!instance);
  instance = new PipelineManager(device);
}

void PipelineManager::shutdown() {
  assert(instance);
  delete instance;
}

// TODO: detach ownership of layouts from pipelines
// TODO: multithread
PipelineHandle PipelineManager::load_compute_pipeline(const ComputePipelineCreateInfo& info) {
  ZoneScoped;
  ShaderManager::ShaderCreateInfo shader_create_info = {.path = info.path,
                                                        .stage = VK_SHADER_STAGE_COMPUTE_BIT};
  ShaderManager::LoadProgramResult result =
      shader_manager_.load_program(SPAN1(shader_create_info), info.layout == nullptr);
  if (result.module_cnt != 1) {
    LINFO("no modules generated during compute pipeline creation");
    return PipelineHandle{};
  }
  if (info.layout) {
    result.layout = info.layout;
  }

  VkPipeline pipeline = create_compute_pipeline(result, info.entry_point);
  if (!pipeline) {
    return PipelineHandle{};
  }
  auto handle = std::hash<std::string>{}(info.path.string());

  pipelines_.emplace(handle,
                     PipelineAndMetadata{.pipeline = {.pipeline = pipeline,
                                                      .layout = result.layout,
                                                      .owns_layout = info.layout == nullptr},
                                         .shader_paths = {info.path.string()}});
  return PipelineHandle{handle};
}

Pipeline* PipelineManager::get(PipelineHandle handle) {
  ZoneScoped;
  auto it = pipelines_.find(handle);
  return it != pipelines_.end() ? &it->second.pipeline : nullptr;
}

void PipelineManager::destroy_pipeline(PipelineHandle handle) {
  ZoneScoped;
  // destroy the pipeline
  auto it = pipelines_.find(handle);
  if (it == pipelines_.end()) {
    LERROR("pipeline not found");
    return;
  }
  vkDestroyPipelineLayout(device_, it->second.pipeline.layout, nullptr);
  vkDestroyPipeline(device_, it->second.pipeline.pipeline, nullptr);

  // TODO: only if hot reloading?

  // for all the shader paths used by this pipeline, remove the pipeline from them
  for (const auto& shader_name : it->second.shader_paths) {
    auto it2 = shader_name_to_used_pipelines_.find(shader_name);
    if (it2 != shader_name_to_used_pipelines_.end()) {
      auto& used_pipelines = it2->second;
      for (u64 i = 0; i < used_pipelines.size(); i++) {
        if (used_pipelines[i] == handle) {
          used_pipelines[i] = used_pipelines.back();
          used_pipelines.pop_back();
        }
      }
    }
  }
}

void PipelineManager::on_shader_update() {
  // get all the pipelines that use this shader and reload
}

PipelineManager::~PipelineManager() {
  ZoneScoped;
  shader_manager_.clear_module_cache();
  for (auto& [handle, metadata] : pipelines_) {
    assert(metadata.pipeline.pipeline);
    if (metadata.pipeline.pipeline) {
      vkDestroyPipeline(device_, metadata.pipeline.pipeline, nullptr);
      if (metadata.pipeline.owns_layout) {
        vkDestroyPipelineLayout(device_, metadata.pipeline.layout, nullptr);
      }
    }
  }
}
namespace {

template <typename T>
std::span<const T> to_span(std::initializer_list<T> list) {
  return std::span<const T>(list.begin(), list.size());
}
}  // namespace

PipelineManager::PipelineManager(VkDevice device) : shader_manager_(device), device_(device) {}

PipelineHandle PipelineManager::load_graphics_pipeline(const GraphicsPipelineCreateInfo& info) {
  ShaderManager::ShaderCreateInfo shader_create_infos[] = {
      {.path = info.vertex_path, .stage = VK_SHADER_STAGE_VERTEX_BIT},
      {.path = info.fragment_path, .stage = VK_SHADER_STAGE_FRAGMENT_BIT}};
  ShaderManager::LoadProgramResult result =
      shader_manager_.load_program(ARR_SPAN(shader_create_infos), info.layout == nullptr);
  if (result.module_cnt != 0) {
    assert(result.module_cnt == 2);
  }
  if (result.module_cnt != 2) {
    return {};
  }

  VkPipelineShaderStageCreateInfo stages[] = {
      VkPipelineShaderStageCreateInfo{.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                                      .stage = VK_SHADER_STAGE_VERTEX_BIT,
                                      .module = result.modules[0].module,
                                      .pName = "main"},
      VkPipelineShaderStageCreateInfo{.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                                      .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
                                      .module = result.modules[1].module,
                                      .pName = "main"}};

  VkPipelineInputAssemblyStateCreateInfo input_assembly{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
      .topology = convert_prim_topology(info.topology)};
  VkPipelineRasterizationStateCreateInfo rasterization{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
      .depthClampEnable = info.rasterization.depth_clamp,
      .rasterizerDiscardEnable = info.rasterization.rasterize_discard_enable,
      .polygonMode = convert_polygon_mode(info.rasterization.polygon_mode),
      .cullMode = convert_cull_mode(info.rasterization.cull_mode),
      .frontFace = convert_front_face(info.rasterization.front_face),
      .depthBiasEnable = info.rasterization.depth_bias,
      .depthBiasConstantFactor = info.rasterization.depth_bias_constant_factor,
      .depthBiasClamp = info.rasterization.depth_bias_clamp,
      .depthBiasSlopeFactor = info.rasterization.depth_bias_slope_factor,
      .lineWidth = info.rasterization.line_width};
  assert(info.blend.attachments.size() <= 4);
  std::array<VkPipelineColorBlendAttachmentState, 4> attachments{};
  u32 i = 0;
  u32 attachment_cnt = info.blend.attachments.size();
  for (const auto& attachment : info.blend.attachments) {
    attachments[i++] = convert_color_blend_attachment(attachment);
  }
  // dummy blend attachment if color attachment is specified but no blending
  if (i == 0 && info.rendering.color_formats.size() > 0) {
    attachment_cnt = 1;
    attachments[0] =
        convert_color_blend_attachment(GraphicsPipelineCreateInfo::ColorBlendAttachment{});
  }

  VkPipelineColorBlendStateCreateInfo blend_state{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
      .logicOpEnable = info.blend.logic_op_enable,
      .logicOp = convert_logic_op(info.blend.logic_op),
      .attachmentCount = attachment_cnt,
      .pAttachments = attachments.data()};

  VkPipelineMultisampleStateCreateInfo multisample{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
      .rasterizationSamples =
          static_cast<VkSampleCountFlagBits>(info.multisample.rasterization_samples),
      .sampleShadingEnable = info.multisample.sample_shading_enable,
      .minSampleShading = info.multisample.min_sample_shading,
      .alphaToCoverageEnable = info.multisample.alpha_to_coverage_enable,
      .alphaToOneEnable = info.multisample.alpha_to_one_enable};
  VkPipelineDepthStencilStateCreateInfo depth_stencil{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
      .depthTestEnable = info.depth_stencil.depth_test_enable,
      .depthWriteEnable = info.depth_stencil.depth_write_enable,
      .depthCompareOp = static_cast<VkCompareOp>(info.depth_stencil.depth_compare_op),
      .depthBoundsTestEnable = info.depth_stencil.depth_bounds_test_enable,
      .stencilTestEnable = info.depth_stencil.stencil_test_enable,
      .front = convert_stencil_op_state(info.depth_stencil.stencil_front),
      .back = convert_stencil_op_state(info.depth_stencil.stencil_back),
      .minDepthBounds = info.depth_stencil.min_depth_bounds,
      .maxDepthBounds = info.depth_stencil.max_depth_bounds};

  VkPipelineViewportStateCreateInfo viewport_state{};
  viewport_state.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
  viewport_state.pNext = nullptr;
  viewport_state.viewportCount = 1;
  viewport_state.scissorCount = 1;

  // TODO: configurable dynamic state
  VkDynamicState default_dynamic_state[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
  VkPipelineDynamicStateCreateInfo dynamic_state{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
      .dynamicStateCount = 2,
      .pDynamicStates = default_dynamic_state};
  VkPipelineVertexInputStateCreateInfo vertex_state{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};

  VkPipelineRenderingCreateInfo rendering_info{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
      .colorAttachmentCount = static_cast<u32>(info.rendering.color_formats.size()),
      .pColorAttachmentFormats = info.rendering.color_formats.data(),
      .depthAttachmentFormat = info.rendering.depth_format,
      .stencilAttachmentFormat = info.rendering.stencil_format};
  VkGraphicsPipelineCreateInfo cinfo{.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
                                     .pNext = &rendering_info,
                                     .stageCount = 2,
                                     .pStages = stages,
                                     .pVertexInputState = &vertex_state,
                                     .pInputAssemblyState = &input_assembly,
                                     .pTessellationState = nullptr,
                                     .pViewportState = &viewport_state,
                                     .pRasterizationState = &rasterization,
                                     .pMultisampleState = &multisample,
                                     .pDepthStencilState = &depth_stencil,
                                     .pColorBlendState = &blend_state,
                                     .pDynamicState = &dynamic_state,
                                     .layout = info.layout};
  VkPipeline pipeline;
  VK_CHECK(vkCreateGraphicsPipelines(device_, nullptr, 1, &cinfo, nullptr, &pipeline));
  if (!pipeline) return {};

  auto tup = std::make_tuple(info.vertex_path.string(), info.fragment_path.string());
  auto handle = detail::hashing::hash<decltype(tup)>{}(tup);

  pipelines_.emplace(handle,
                     PipelineAndMetadata{
                         .pipeline = {.pipeline = pipeline,
                                      .layout = result.layout,
                                      .owns_layout = info.layout == nullptr},
                         .shader_paths = {info.vertex_path.string(), info.fragment_path.string()}});

  return PipelineHandle{handle};
}
}  // namespace vk2
