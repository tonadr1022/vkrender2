#include "PipelineManager.hpp"

#include <volk.h>
#include <vulkan/vulkan_core.h>

#include <cassert>
#include <ranges>
#include <tracy/Tracy.hpp>
#include <utility>

#include "core/FixedVector.hpp"
#include "core/Logger.hpp"
#include "imgui.h"
#include "vk2/BindlessResourceAllocator.hpp"
#include "vk2/Hash.hpp"
#include "vk2/Initializers.hpp"
#include "vk2/ShaderCompiler.hpp"
#include "vk2/VkCommon.hpp"
#include "vk2/VkTypes.hpp"

namespace gfx::vk2 {

namespace {

// file is written when program is off
// need to reload it because shaders that include it depend on it
// how to tell?
// file --> write time

// on init: read through cached include write times for every loaded shader

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
  VkPipelineShaderStageCreateInfo stage{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
      .stage = VK_SHADER_STAGE_COMPUTE_BIT,
      .module = result.modules[0],
      .pName = entry_point};
  VkComputePipelineCreateInfo create_info{.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                                          .stage = stage,
                                          .layout = default_pipeline_layout_};
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

void PipelineManager::init(VkDevice device, std::filesystem::path shader_dir, bool hot_reload,
                           VkPipelineLayout default_layout) {
  assert(!instance);
  instance = new PipelineManager(device, std::move(shader_dir), hot_reload, default_layout);
}

void PipelineManager::shutdown() {
  ZoneScoped;
  assert(instance);
  delete instance;
}

// TODO: multithread
PipelineHandle PipelineManager::load_compute(const ShaderCreateInfo& info) {
  // handle is the hash of the create info
  u64 handle;
  VkPipeline pipeline = load_compute_pipeline_impl(info, &handle, false);
  if (!pipeline) {
    return PipelineHandle{};
  }
  pipelines_.emplace(handle,
                     PipelineAndMetadata{.pipeline = {.pipeline = pipeline, .owns_layout = false},
                                         .shader_paths = {info.path.string()},
                                         .type = PipelineType::Compute});
  compute_pipeline_infos_.emplace(PipelineHandle{handle}, info);
  auto full_path = get_shader_path(info.path) + ".glsl";
  shader_name_to_used_pipelines_[full_path].emplace(handle);

  return PipelineHandle{handle};
}

Pipeline* PipelineManager::get(PipelineHandle handle) {
  ZoneScoped;
  auto it = pipelines_.find(handle);
  return it != pipelines_.end() ? &it->second.pipeline : nullptr;
}

PipelineManager::~PipelineManager() {
  ZoneScoped;
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

PipelineManager::PipelineManager(VkDevice device, std::filesystem::path shader_dir, bool hot_reload,
                                 VkPipelineLayout default_layout)
    : shader_dir_(std::move(shader_dir)),
      shader_manager_(
          device, shader_dir_ / ".cache",
          [this](std::span<std::filesystem::path> dirty_files) { on_dirty_files(dirty_files); },
          shader_dir_, hot_reload),
      default_pipeline_layout_(default_layout),
      device_(device) {}

std::string PipelineManager::get_shader_path(const std::string& path) const {
  return shader_dir_ / path;
}

PipelineHandle PipelineManager::load(const GraphicsPipelineCreateInfo& cinfo) {
  // TODO: verify path ends in .vert/.frag
  auto info = cinfo;
  if (info.shaders.size() == 0) {
    return {};
  }
  u64 handle{};
  VkPipeline pipeline = load_graphics_pipeline_impl(info, &handle, false);
  if (!pipeline) {
    return {};
  }
  std::vector<std::string> shader_paths;
  for (const auto& shader_info : info.shaders) {
    if (shader_info.path.empty()) {
      continue;
    }
    shader_paths.emplace_back(shader_info.path.string());
  }
  pipelines_.emplace(handle,
                     PipelineAndMetadata{.pipeline = {.pipeline = pipeline, .owns_layout = false},
                                         .shader_paths = shader_paths,
                                         .type = PipelineType::Graphics});
  graphics_pipeline_infos_.emplace(PipelineHandle{handle}, info);
  for (auto& path : shader_paths) {
    if (path.empty()) {
      continue;
    }
    auto full_glsl_path = get_shader_path(path + ".glsl");
    shader_name_to_used_pipelines_[full_glsl_path].emplace(handle);
  }

  return PipelineHandle{handle};
}

void PipelineManager::bind_graphics(VkCommandBuffer cmd, PipelineHandle handle) {
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, get(handle)->pipeline);
}

void PipelineManager::bind_compute(VkCommandBuffer cmd, PipelineHandle handle) {
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, get(handle)->pipeline);
}

VkPipeline PipelineManager::load_graphics_pipeline_impl(const GraphicsPipelineCreateInfo& info,
                                                        u64* out_info_hash, bool force) {
  u32 stage_cnt = info.shaders.size();
  std::array<std::vector<std::string>, 2> include_files;
  std::array<u64, 2> create_info_hashes;
  ShaderManager::LoadProgramResult result =
      shader_manager_.load_program(info.shaders, create_info_hashes, force);
  if (!result.success) {
    for (u32 i = 0; i < stage_cnt; i++) {
      if (result.modules[i]) {
        vkDestroyShaderModule(device_, result.modules[i], nullptr);
      }
    }
    return {};
  }
  if (out_info_hash) {
    u64 hash = 0;
    for (auto& h : create_info_hashes) {
      detail::hashing::hash_combine(hash, h);
    }
    *out_info_hash = hash;
  }

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
  std::array<VkPipelineColorBlendAttachmentState, 10> attachments{};
  u32 i = 0;
  u32 attachment_cnt = info.blend.attachments.size();
  for (const auto& attachment : info.blend.attachments) {
    attachments[i++] = convert_color_blend_attachment(attachment);
  }
  u32 color_format_cnt = 0;
  for (auto format : info.rendering.color_formats) {
    if (format != VK_FORMAT_UNDEFINED) {
      color_format_cnt++;
    } else {
      break;
    }
  }
  // dummy blend attachment if color attachment is specified but no blending
  if (i == 0 && color_format_cnt > 0) {
    attachment_cnt = color_format_cnt;
    auto default_blend =
        convert_color_blend_attachment(GraphicsPipelineCreateInfo::ColorBlendAttachment{});
    for (u32 i = 0; i < attachment_cnt; i++) {
      attachments[i] = default_blend;
    }
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
  VkPipelineDynamicStateCreateInfo dynamic_state{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};

  util::fixed_vector<VkDynamicState, 100> states;
  if (info.dynamic_state.size() == 0) {
    states = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR, VK_DYNAMIC_STATE_CULL_MODE};
    dynamic_state.dynamicStateCount = states.size();
  } else {
    u32 i = 0;
    for (auto s : info.dynamic_state) {
      states[i++] = s;
    }
    dynamic_state.dynamicStateCount = i;
  }
  dynamic_state.pDynamicStates = states.data();
  VkPipelineVertexInputStateCreateInfo vertex_state{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};

  VkPipelineRenderingCreateInfo rendering_info{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
      .colorAttachmentCount = color_format_cnt,
      .pColorAttachmentFormats = info.rendering.color_formats.data(),
      .depthAttachmentFormat = info.rendering.depth_format,
      .stencilAttachmentFormat = info.rendering.stencil_format};

  std::array<VkPipelineShaderStageCreateInfo, 3> stages;
  stages[0] =
      VkPipelineShaderStageCreateInfo{.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                                      .stage = VK_SHADER_STAGE_VERTEX_BIT,
                                      .module = result.modules[0],
                                      .pName = "main"};
  if (stage_cnt == 2) {
    stages[1] = VkPipelineShaderStageCreateInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
        .module = result.modules[1],
        .pName = "main"};
  }
  VkGraphicsPipelineCreateInfo cinfo{
      .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
      .pNext = &rendering_info,
      .stageCount = stage_cnt,
      .pStages = stages.data(),
      .pVertexInputState = &vertex_state,
      .pInputAssemblyState = &input_assembly,
      .pTessellationState = nullptr,
      .pViewportState = &viewport_state,
      .pRasterizationState = &rasterization,
      .pMultisampleState = &multisample,
      .pDepthStencilState = &depth_stencil,
      .pColorBlendState = &blend_state,
      .pDynamicState = &dynamic_state,
      .layout = info.layout ? info.layout : default_pipeline_layout_};
  VkPipeline pipeline{};
  VK_CHECK(vkCreateGraphicsPipelines(device_, nullptr, 1, &cinfo, nullptr, &pipeline));

  for (u32 i = 0; i < stage_cnt; i++) {
    vkDestroyShaderModule(device_, result.modules[i], nullptr);
  }
  if (!pipeline) return {};

  LINFO("loaded graphics pipeline: {}", info.shaders[0].path.string());

  return pipeline;
}

VkPipeline PipelineManager::load_compute_pipeline_impl(const ShaderCreateInfo& info,
                                                       u64* out_info_hash, bool force) {
  ZoneScoped;
  std::array<std::vector<std::string>, 1> include_files_arr;

  std::array<u64, 1> info_hash;

  ShaderManager::LoadProgramResult result =
      shader_manager_.load_program(SPAN1(info), info_hash, force);
  if (out_info_hash) {
    *out_info_hash = info_hash[0];
  }

  if (!result.success) {
    LINFO("failed to load compute pipeline: {}", info.path.string());
    if (result.modules[0]) {
      vkDestroyShaderModule(device_, result.modules[0], nullptr);
    }
    return {};
  }

  VkPipeline pipeline = create_compute_pipeline(result, info.entry_point.c_str());
  vkDestroyShaderModule(device_, result.modules[0], nullptr);
  LINFO("loaded compute pipeline: {}", info.path.string());
  return pipeline;
}

void PipelineManager::reload_shaders() {
  for (auto& [handle, pipeline] : pipelines_) {
    reload_pipeline(handle, true);
  }
}

void PipelineManager::reload_pipeline(PipelineHandle handle, bool force) {
  auto pipeline_it = pipelines_.find(handle);
  if (pipeline_it == pipelines_.end()) {
    assert(0);
    return;
  }
  auto& pipeline = pipeline_it->second;
  if (pipeline.type == PipelineType::Graphics) {
    auto it = graphics_pipeline_infos_.find(handle);
    assert(it != graphics_pipeline_infos_.end());
    u64 new_hash;
    VkPipeline res = load_graphics_pipeline_impl(it->second, &new_hash, force);
    if (res) {
      if (pipeline.pipeline.pipeline) {
        ResourceAllocator::get().enqueue_delete_pipeline(pipeline.pipeline.pipeline);
      }
      pipeline.pipeline.pipeline = res;
    }

  } else if (pipeline.type == PipelineType::Compute) {
    auto it = compute_pipeline_infos_.find(handle);
    assert(it != compute_pipeline_infos_.end());
    u64 new_hash;
    VkPipeline res = load_compute_pipeline_impl(it->second, &new_hash, force);
    if (res) {
      if (pipeline.pipeline.pipeline) {
        ResourceAllocator::get().enqueue_delete_pipeline(pipeline.pipeline.pipeline);
      }
      pipeline.pipeline.pipeline = res;
    }
  }

  auto it = graphics_pipeline_infos_.find(handle);
  if (it != graphics_pipeline_infos_.end()) {
  }
}

void PipelineManager::on_dirty_files(std::span<std::filesystem::path> dirty_files) {
  for (auto& file : dirty_files) {
    if (file.extension() == ".glsl") {
      auto it = shader_name_to_used_pipelines_.find(file.string());
      if (it != shader_name_to_used_pipelines_.end()) {
        for (PipelineHandle handle : it->second) {
          reload_pipeline(handle, false);
        }
      }
    }
  }
}

size_t PipelineManager::get_pipeline_hash(const GraphicsPipelineCreateInfo& info) {
  size_t hash{};
  for (const auto& shader_info : info.shaders) {
    detail::hashing::hash_combine(hash, shader_info.path.string());
    detail::hashing::hash_combine(hash, shader_info.entry_point);
    for (const auto& define : shader_info.defines) {
      detail::hashing::hash_combine(hash, define);
    }
  }

  return hash;
}

void PipelineManager::on_imgui() {
  if (ImGui::Button("Reload All Shaders")) {
    reload_shaders();
  }
  shader_manager_.on_imgui();
}

}  // namespace gfx::vk2
