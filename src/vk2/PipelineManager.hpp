#pragma once

#include <vulkan/vulkan_core.h>

#include <cstring>
#include <filesystem>
#include <queue>
#include <unordered_map>
#include <vector>

#include "Common.hpp"
#include "vk2/Handle.hpp"
#include "vk2/ShaderCompiler.hpp"

namespace vk2 {

struct Pipeline {
  VkPipeline pipeline;
};

template <typename T>
struct FreeListPool {
  using HandleT = u32;
  std::vector<T> data;
  std::queue<HandleT> free_list;
  HandleT emplace(T &&val) {
    HandleT handle = next_handle();
    assert(handle <= data.size());
    if (handle == data.size()) {
      data.emplace_back(std::forward<T>(val));
    } else {
      data[handle] = std::move(val);
    }
    return handle;
  }
  void free(HandleT handle) {
    assert(handle >= 0 && handle < data.size());
    data[handle].~T();
    memset(&data[handle], 0, sizeof(T));
    free_list.push(handle);
  }

  T &get(HandleT handle) {
    assert(handle >= 0 && handle < data.size());
    return data[handle];
  }

  [[nodiscard]] const T &get(HandleT handle) const {
    assert(handle >= 0 && handle < data.size());
    return data[handle];
  }

  FreeListPool() = default;
  explicit FreeListPool(u32 size) {
    data.resize(size);
    for (u32 i = 0; i < size; i++) {
      free_list.push(size);
    }
  }
  // lazy, this shouldn't be moved around though
  FreeListPool(FreeListPool &&) = delete;
  FreeListPool(const FreeListPool &) = delete;
  FreeListPool &operator=(FreeListPool &&) = delete;
  FreeListPool &operator=(const FreeListPool &) = delete;

 private:
  HandleT nxt_handle_cntr_{};
  HandleT next_handle() {
    if (free_list.size()) {
      HandleT handle = free_list.front();
      free_list.pop();
      return handle;
    }
    HandleT handle = ++nxt_handle_cntr_;
    nxt_handle_cntr_ = (nxt_handle_cntr_ + 1) % UINT32_MAX;
    return handle;
  }
};

VK2_DEFINE_HANDLE_WITH_NAME(Pipeline, PipelineAndMetadata);
template <typename T>
using Ptr = std::shared_ptr<T>;

class PipelineManager {
 public:
  static PipelineManager &get();
  static void init(VkDevice device);
  static void shutdown();

  void on_shader_update();

  PipelineHandle load_compute_pipeline(const std::filesystem::path &path,
                                       const char *entry_point = "main");

  Pipeline *get(PipelineHandle handle);

  void clear_module_cache();

  void destroy_pipeline(PipelineHandle handle);

 private:
  void shutdown_impl();
  void init_impl(VkDevice device);
  VkPipeline load_compute_pipeline(ShaderManager::LoadShaderResult &result,
                                   const char *entry_point = "main");

  struct PipelineAndMetadata {
    Pipeline pipeline;
    std::vector<std::string> shader_paths;
  };
  std::unordered_map<std::string, std::vector<PipelineHandle>> shader_name_to_used_pipelines_;
  std::unordered_map<PipelineHandle, PipelineAndMetadata> pipelines_;

  // FreeListPool<std::unique_ptr<PipelineAndMetadata>> pipelines_{30};
  VkShaderModule get_module(const std::filesystem::path &path, VkShaderStageFlagBits stage);
  std::unordered_map<std::string, VkShaderModule> module_cache_;
  VkDevice device_;
};
}  // namespace vk2
