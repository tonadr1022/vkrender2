#include "PipelineManager.hpp"

#include <volk.h>
#include <vulkan/vulkan_core.h>

#include <cassert>
#include <fstream>
#include <ranges>
#include <tracy/Tracy.hpp>
#include <utility>

#include "Logger.hpp"
#include "vk2/Hash.hpp"
#include "vk2/Initializers.hpp"
#include "vk2/ShaderCompiler.hpp"
#include "vk2/VkCommon.hpp"

namespace gfx::vk2 {

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
void PipelineManager::init(VkDevice device, std::filesystem::path shader_dir, bool hot_reload,
                           VkPipelineLayout default_layout) {
  assert(!instance);
  instance = new PipelineManager(device, std::move(shader_dir), hot_reload, default_layout);
}

void PipelineManager::shutdown() {
  assert(instance);
  delete instance;
}

// TODO: multithread
PipelineHandle PipelineManager::load_compute_pipeline(const ComputePipelineCreateInfo& info) {
  // TODO: verify that path ends in .comp
  VkPipeline pipeline = load_compute_pipeline_impl(info);
  if (!pipeline) {
    return PipelineHandle{};
  }
  auto handle = std::hash<std::string>{}(info.path.string());
  pipelines_.emplace(handle,
                     PipelineAndMetadata{.pipeline = {.pipeline = pipeline, .owns_layout = false},
                                         .shader_paths = {info.path.string()}});
  compute_pipeline_infos_.emplace(PipelineHandle{handle}, info);
  auto full_path = get_shader_path(info.path) + ".glsl";
  auto it = shader_name_to_used_pipelines_.find(full_path);
  if (it == shader_name_to_used_pipelines_.end()) {
    shader_name_to_used_pipelines_[full_path] = {PipelineHandle{handle}};
  } else {
    it->second.emplace_back(handle);
  }
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
  if (it->second.pipeline.owns_layout) {
    vkDestroyPipelineLayout(device_, it->second.pipeline.layout, nullptr);
  }
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
  pipelines_.erase(it);
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

PipelineManager::PipelineManager(VkDevice device, std::filesystem::path shader_dir, bool hot_reload,
                                 VkPipelineLayout default_layout)
    : file_watcher_(
          shader_dir,
          [this](std::span<std::filesystem::path> dirty_files) { on_dirty_files(dirty_files); },
          std::chrono::milliseconds(250),
          hot_reload ? shader_dir / ".cache" / "dirty_shaders.txt" : ""),
      shader_dir_(std::move(shader_dir)),
      shader_manager_(device),
      default_pipeline_layout_(default_layout),
      device_(device),
      hot_reload_(hot_reload) {
  // TODO: flag for enabled
  if (hot_reload) {
    file_watcher_.start();
  }
  std::filesystem::path shader_include_path = shader_dir / ".cache" / "shader_includes.txt";
  if (std::filesystem::exists(shader_include_path)) {
    std::ifstream ifs(shader_include_path);
    if (ifs.is_open()) {
      std::string shader;
      while (ifs >> shader) {
      }
    }
  }
}

std::string PipelineManager::get_shader_path(const std::string& path) const {
  return shader_dir_ / path;
}
PipelineHandle PipelineManager::load_graphics_pipeline(const GraphicsPipelineCreateInfo& info) {
  // TODO: verify path ends in .vert/.frag
  VkPipeline pipeline = load_graphics_pipeline_impl(info);
  if (!pipeline) {
    return {};
  }

  auto tup = std::make_tuple(info.vertex_path.string(), info.fragment_path.string());
  auto handle = detail::hashing::hash<decltype(tup)>{}(tup);
  pipelines_.emplace(handle,
                     PipelineAndMetadata{
                         .pipeline = {.pipeline = pipeline, .owns_layout = false},
                         .shader_paths = {info.vertex_path.string(), info.fragment_path.string()}});
  graphics_pipeline_infos_.emplace(PipelineHandle{handle}, info);
  auto full_vert_path = get_shader_path(info.vertex_path) + ".glsl";
  auto full_frag_path = get_shader_path(info.fragment_path) + ".glsl";
  // LINFO("{} full frag", full_frag_path);
  auto it = shader_name_to_used_pipelines_.find(full_vert_path);
  if (it == shader_name_to_used_pipelines_.end()) {
    shader_name_to_used_pipelines_[full_vert_path] = {PipelineHandle{handle}};
  } else {
    it->second.emplace_back(handle);
  }
  it = shader_name_to_used_pipelines_.find(full_frag_path);
  if (it == shader_name_to_used_pipelines_.end()) {
    shader_name_to_used_pipelines_[full_frag_path] = {PipelineHandle{handle}};
  } else {
    it->second.emplace_back(handle);
  }
  return PipelineHandle{handle};
}
void PipelineManager::bind_graphics(VkCommandBuffer cmd, PipelineHandle handle) {
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, get(handle)->pipeline);
}

void PipelineManager::bind_compute(VkCommandBuffer cmd, PipelineHandle handle) {
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, get(handle)->pipeline);
}

VkPipeline PipelineManager::load_graphics_pipeline_impl(const GraphicsPipelineCreateInfo& info) {
  std::array<ShaderManager::ShaderCreateInfo, 2> shader_create_infos{
      {{get_shader_path(info.vertex_path), VK_SHADER_STAGE_VERTEX_BIT},
       {get_shader_path(info.fragment_path), VK_SHADER_STAGE_FRAGMENT_BIT}}};
  u32 stage_cnt = 1;
  if (info.fragment_path.string().length()) {
    stage_cnt = 2;
  }

  std::array<std::vector<std::string>, 2> include_files;
  ShaderManager::LoadProgramResult result;
  if (hot_reload_) {
    result = shader_manager_.load_program(std::span(shader_create_infos.data(), stage_cnt), false,
                                          include_files);
  } else {
    result =
        shader_manager_.load_program(std::span(shader_create_infos.data(), stage_cnt), false, {});
  }

  if (result.module_cnt != stage_cnt) {
    assert(result.module_cnt == stage_cnt);
    return nullptr;
  }

  if (hot_reload_) {
    for (u32 i = 0; i < stage_cnt; i++) {
      shader_to_includes_.emplace(shader_create_infos[i].path.string(),
                                  std::move(include_files[i]));
    }
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
  std::array<VkPipelineColorBlendAttachmentState, 4> attachments{};
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
  VkPipelineDynamicStateCreateInfo dynamic_state{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};

  std::array<VkDynamicState, 100> states;
  if (info.dynamic_state.size() == 0) {
    states = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    dynamic_state.dynamicStateCount = 2;
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
                                      .module = result.modules[0].module,
                                      .pName = "main"};
  if (stage_cnt == 2) {
    stages[1] = VkPipelineShaderStageCreateInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
        .module = result.modules[1].module,
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
  if (!pipeline) return {};
  return pipeline;
}

VkPipeline PipelineManager::load_compute_pipeline_impl(const ComputePipelineCreateInfo& info) {
  ZoneScoped;
  ShaderManager::ShaderCreateInfo shader_create_info = {.path = get_shader_path(info.path),
                                                        .stage = VK_SHADER_STAGE_COMPUTE_BIT};
  std::array<std::vector<std::string>, 1> include_files_arr;
  ShaderManager::LoadProgramResult result =
      shader_manager_.load_program(SPAN1(shader_create_info), false, include_files_arr);
  if (result.module_cnt != 1) {
    LINFO("no modules generated during compute pipeline creation");
    return nullptr;
  }
  if (info.layout) {
    result.layout = info.layout;
  } else {
    result.layout = default_pipeline_layout_;
  }
  shader_to_includes_.emplace(shader_create_info.path.string(), std::move(include_files_arr[0]));

  return create_compute_pipeline(result, info.entry_point);
}

void PipelineManager::on_dirty_files(std::span<std::filesystem::path> dirty_files) {
  for (auto& file : dirty_files) {
    if (file.extension() == ".glsl") {
      auto it = shader_name_to_used_pipelines_.find(file.string());
      if (it != shader_name_to_used_pipelines_.end()) {
        for (PipelineHandle handle : it->second) {
          auto pipeline_it = pipelines_.find(handle);
          assert(pipeline_it != pipelines_.end());
          if (pipeline_it == pipelines_.end()) continue;
          auto graphics_it = graphics_pipeline_infos_.find(handle);
          if (graphics_it != graphics_pipeline_infos_.end()) {
            VkPipeline res = load_graphics_pipeline_impl(graphics_it->second);
            if (res) {
              vkDestroyPipeline(device_, pipeline_it->second.pipeline.pipeline, nullptr);
              LINFO("res");
              pipeline_it->second.pipeline.pipeline = res;
            }
          }
          auto compute_it = compute_pipeline_infos_.find(handle);
          if (compute_it != compute_pipeline_infos_.end()) {
            pipeline_it->second.pipeline.pipeline = load_compute_pipeline_impl(compute_it->second);
          }
        }
      }
      // LINFO("dirty shader file: {}", file.string());
    }
  }
}

}  // namespace gfx::vk2
