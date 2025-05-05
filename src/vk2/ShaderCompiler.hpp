#pragma once
#include <vulkan/vulkan_core.h>

#include <filesystem>
#include <mutex>
#include <span>
#include <unordered_map>
#include <vector>

#include "Common.hpp"
#include "vk2/Descriptors.hpp"

namespace gfx::vk2 {

struct DescriptorSetLayoutData {
  uint32_t set_number;
  VkDescriptorSetLayoutCreateInfo create_info;
  std::vector<VkDescriptorSetLayoutBinding> bindings;
};

struct ShaderReflectData {
  std::array<DescriptorSetLayoutData, 4> set_layouts;
  uint32_t set_layout_cnt{0};
  VkPushConstantRange range;
  bool has_pc_range{};
  VkShaderStageFlags shader_stage;
};

struct ShaderModule {
  ShaderReflectData refl_data;
  VkShaderModule module;
};

class ShaderManager {
 public:
  explicit ShaderManager(VkDevice device, std::filesystem::path spirv_hash_path);
  ~ShaderManager();

  struct ShaderCreateInfo {
    std::filesystem::path path;
    VkShaderStageFlagBits stage;
  };

  struct LoadProgramResult {
    static constexpr int max_stages = 4;
    std::array<ShaderModule, max_stages> modules;
    u64 module_cnt{};
    VkPipelineLayout layout{};
  };
  LoadProgramResult load_program(std::span<ShaderCreateInfo> shader_create_infos,
                                 std::span<std::vector<std::string>> include_files);

 private:
  std::unordered_map<std::filesystem::path, uint64_t> spirv_hash_cache_;
  std::filesystem::path spirv_hash_path_;
  struct CompileToSpirvResult {
    std::vector<uint32_t> binary_data;
  };
  bool get_spirv_binary(const std::filesystem::path& path, VkShaderStageFlagBits stage,
                        CompileToSpirvResult& result, std::vector<std::string>* include_files);
  bool get_spirv_binary(const std::filesystem::path& path, VkShaderStageFlagBits stage,
                        CompileToSpirvResult& result, bool needs_new,
                        std::vector<std::string>* include_files);
  VkDevice device_;
  struct ShaderMetadata {
    std::filesystem::file_time_type last_spirv_write;
  };
  bool get_dirty_stages(std::span<ShaderCreateInfo> infos, std::span<bool> dirty_flags);
  DescriptorSetLayoutCache layout_cache_;
  std::unordered_map<u64, ShaderMetadata> shader_metadata_;
  std::mutex mtx_;
  u64 hash(const std::filesystem::path& path, VkShaderStageFlagBits stage);
};

}  // namespace gfx::vk2
