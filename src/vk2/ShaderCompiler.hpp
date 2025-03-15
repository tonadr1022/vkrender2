#pragma once
#include <vulkan/vulkan_core.h>

#include <filesystem>
#include <unordered_map>
#include <vector>

#include "Common.hpp"
#include "vk2/Descriptors.hpp"

namespace vk2 {

class ShaderManager {
 public:
  static void init(VkDevice device);
  static void shutdown();
  static ShaderManager& get();

  struct ShaderCreateInfo {
    std::filesystem::path path;
    VkShaderStageFlagBits stage;
  };
  struct LoadShaderResult {
    static constexpr int max_stages = 4;
    std::array<VkShaderModule, max_stages> modules;
    u64 module_cnt{};
    VkPipelineLayout layout{};
  };

  LoadShaderResult load_shader(std::span<ShaderCreateInfo> shader_create_infos);
  VkShaderModule load_shader(const std::filesystem::path& path, VkShaderStageFlagBits stage);

 private:
  struct CompileToSpirvResult {
    std::vector<uint32_t> binary_data;
  };
  bool get_spirv_binary(const std::filesystem::path& path, VkShaderStageFlagBits stage,
                        CompileToSpirvResult& result);
  void init_impl(VkDevice device);
  void shutdown_impl();
  VkDevice device_;
  struct ShaderMetadata {
    std::filesystem::file_time_type last_spirv_write;
  };
  DescriptorSetLayoutCache layout_cache_;
  std::unordered_map<u64, ShaderMetadata> shader_metadata_;
  u64 hash(const std::filesystem::path& path, VkShaderStageFlagBits stage);
};

}  // namespace vk2
