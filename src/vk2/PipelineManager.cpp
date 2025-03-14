#include "PipelineManager.hpp"

#include <volk.h>
#include <vulkan/vulkan_core.h>

#include <cassert>
#include <tracy/Tracy.hpp>

#include "Logger.hpp"
#include "vk2/Initializers.hpp"
#include "vk2/ShaderCompiler.hpp"
#include "vk2/VkCommon.hpp"

namespace vk2 {

VkPipeline PipelineManager::load_compute_pipeline(ShaderManager::LoadShaderResult& result,
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

PipelineHandle PipelineManager::load_compute_pipeline(const std::filesystem::path& path,
                                                      const char* entry_point) {
  ZoneScoped;
  ShaderManager::ShaderCreateInfo info = {.path = path, .stage = VK_SHADER_STAGE_COMPUTE_BIT};
  ShaderManager::LoadShaderResult result = shader_manager_.load_shader(SPAN1(info));
  if (result.module_cnt == 0) {
    return PipelineHandle{};
  }

  VkPipeline pipeline = load_compute_pipeline(result, entry_point);
  if (!pipeline) {
    return PipelineHandle{};
  }
  auto handle = std::hash<std::string>{}(path.string());
  pipelines_.emplace(
      handle, PipelineAndMetadata{.pipeline = {.pipeline = pipeline, .layout = result.layout},
                                  .shader_paths = {path.string()}});
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
      vkDestroyPipelineLayout(device_, metadata.pipeline.layout, nullptr);
    }
  }
  // clear_module_cache();
}

PipelineManager::PipelineManager(VkDevice device) : shader_manager_(device), device_(device) {}

}  // namespace vk2
