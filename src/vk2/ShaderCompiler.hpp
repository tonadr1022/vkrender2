#pragma once
#include <vulkan/vulkan_core.h>

#include <filesystem>
#include <mutex>
#include <span>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "Common.hpp"
#include "util/FileWatcher.hpp"

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

enum class ShaderType : u8 { None, Vertex, Fragment, Compute };

struct ShaderCreateInfo {
  std::filesystem::path path;
  ShaderType type;
  std::vector<std::string> defines;
  std::string entry_point{"main"};
};

class ShaderManager {
 public:
  using OnDirtyFileFunc = std::function<void(std::span<std::filesystem::path>)>;
  ShaderManager(VkDevice device, std::filesystem::path shader_cache_dir,
                OnDirtyFileFunc on_dirty_files_fn, std::filesystem::path shader_dir,
                bool hot_reload);
  ~ShaderManager();
  void on_imgui();

  struct LoadProgramResult {
    static constexpr int max_stages = 4;
    std::array<ShaderModule, max_stages> modules;
    bool success{};
  };
  LoadProgramResult load_program(std::span<const ShaderCreateInfo> shader_create_infos,
                                 std::span<u64> out_create_info_hashes, bool force);
  void invalidate_cache();

 private:
  void init();
  util::FileWatcher file_watcher_;
  OnDirtyFileFunc on_dirty_files_fn_;
  std::filesystem::path shader_dir_;
  std::filesystem::path shader_cache_dir_;
  std::filesystem::path shader_hash_cache_path_;
  struct CompileToSpirvResult {
    std::vector<uint32_t> binary_data;
  };
  VkDevice device_;
  std::mutex mtx_;

  std::unordered_map<std::filesystem::path, std::unordered_set<std::filesystem::path>>
      include_graph_nodes_;
  bool is_stale(const std::filesystem::path& node);
  bool compile_glsl_to_spirv(const std::string& path, VkShaderStageFlagBits stage,
                             std::vector<u32>& out_binary,
                             std::vector<std::string>* included_files);
  bool shader_debug_mode_{};
  bool hot_reload_{};
};

}  // namespace gfx::vk2
