#include "PipelineManager.hpp"

#include <volk.h>
#include <vulkan/vulkan_core.h>

#include <cassert>

#include "Logger.hpp"
#include "vk2/Initializers.hpp"
#include "vk2/ShaderCompiler.hpp"
#include "vk2/VkCommon.hpp"

namespace vk2 {

VkPipeline PipelineManager::load_compute_pipeline(ShaderManager::LoadShaderResult& result,
                                                  const char* entry_point) {
  assert(result.module_cnt == 1 && result.layout);
  VkPipelineShaderStageCreateInfo stage{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
      .stage = VK_SHADER_STAGE_COMPUTE_BIT,
      .module = result.modules[0],
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
  instance = new PipelineManager;
  instance->init_impl(device);
}

void PipelineManager::shutdown() {
  assert(instance);
  instance->shutdown_impl();
  delete instance;
}

PipelineHandle PipelineManager::load_compute_pipeline(const std::filesystem::path& path,
                                                      const char* entry_point) {
  ShaderManager::ShaderCreateInfo info = {.path = path, .stage = VK_SHADER_STAGE_COMPUTE_BIT};
  ShaderManager::LoadShaderResult result = vk2::ShaderManager::get().load_shader(SPAN1(info));
  if (result.module_cnt == 0) {
    return PipelineHandle{};
  }
  VkPipeline res = load_compute_pipeline(result, entry_point);
  if (!res) {
    return PipelineHandle{};
  }
  auto handle = std::hash<std::string>{}(path.string());
  pipelines_.emplace(
      handle, PipelineAndMetadata{.pipeline = {.pipeline = res}, .shader_paths = {path.string()}});
  return PipelineHandle{handle};
}

void PipelineManager::clear_module_cache() {
  for (auto& [path, module] : module_cache_) {
    vkDestroyShaderModule(device_, module, nullptr);
  }
  module_cache_.clear();
}

VkShaderModule PipelineManager::get_module(const std::filesystem::path&, VkShaderStageFlagBits) {
  return nullptr;
  // auto it = module_cache_.find(path);
  // if (it != module_cache_.end()) {
  //   assert(it->second);
  //   return it->second;
  // }
  // VkShaderModule module = vk2::ShaderManager::get().load_shader(path, stage);
  // if (!module) {
  //   return nullptr;
  // }
  // module_cache_.emplace(path, module);
  // return module;
}

Pipeline* PipelineManager::get(PipelineHandle handle) {
  auto it = pipelines_.find(handle);
  return it != pipelines_.end() ? &it->second.pipeline : nullptr;
}

void PipelineManager::destroy_pipeline(PipelineHandle handle) {
  // destroy the pipeline
  auto it = pipelines_.find(handle);
  if (it == pipelines_.end()) {
    LERROR("pipeline not found");
    return;
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

  // auto it = pipelines_.find(handle);
  // assert(it != pipelines_.end());
  // if (it != pipelines_.end()) {
  //   vkDestroyPipeline(device_, it->second.pipeline.pipeline, nullptr);
  //   pipelines_.erase(it);
  // }
}
void PipelineManager::shutdown_impl() {
  // for (auto& p : pipelines_.data) {
  //   if (p.pipeline.pipeline) {
  //     LINFO("destroying pipeline");
  //     vkDestroyPipeline(device_, p.pipeline.pipeline, nullptr);
  //   }
  // }
  for (auto& [handle, metadata] : pipelines_) {
    assert(metadata.pipeline.pipeline);
    if (metadata.pipeline.pipeline) {
      LINFO("destroying pipeline");
      vkDestroyPipeline(device_, metadata.pipeline.pipeline, nullptr);
    }
  }
  // clear_module_cache();
}

void PipelineManager::on_shader_update() {
  // get all the pipelines that use this shader and reload
}
void PipelineManager::init_impl(VkDevice device) { device_ = device; }
}  // namespace vk2
