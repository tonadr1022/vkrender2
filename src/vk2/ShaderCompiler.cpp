#include "ShaderCompiler.hpp"

#include <glslang/Public/ResourceLimits.h>
#include <glslang/Public/ShaderLang.h>
#include <glslang/SPIRV/GlslangToSpv.h>
#include <volk.h>
#include <vulkan/vk_enum_string_helper.h>
#include <vulkan/vulkan_core.h>

#include <cassert>
#include <filesystem>
#include <fstream>
#include <print>
#include <string>
#include <tracy/Tracy.hpp>
#include <unordered_set>

#include "Common.hpp"
#include "core/FixedVector.hpp"
#include "core/Logger.hpp"
#include "vk2/Hash.hpp"
#include "vk2/VkCommon.hpp"

// src for file includes:
// https://github.com/JuanDiegoMontoya/Frogfood/blob/main/src/Fvog/Shader2.cpp

namespace {

bool load_entire_file(const std::string& path, std::vector<uint32_t>& result);

}  // namespace

namespace gfx::vk2 {

namespace {

VkShaderStageFlagBits convert_shader_stage(ShaderType type) {
  switch (type) {
    case gfx::vk2::ShaderType::Compute:
      return VK_SHADER_STAGE_COMPUTE_BIT;
    case gfx::vk2::ShaderType::Fragment:
      return VK_SHADER_STAGE_FRAGMENT_BIT;
    case gfx::vk2::ShaderType::Vertex:
      return VK_SHADER_STAGE_VERTEX_BIT;
    default:
      return VkShaderStageFlagBits{};
  }
}

size_t hash_shader_info(const ShaderCreateInfo& info, bool debug_mode) {
  size_t hash{};
  for (const auto& define : info.defines) {
    detail::hashing::hash_combine(hash, define);
  }
  detail::hashing::hash_combine(hash, info.entry_point);
  detail::hashing::hash_combine(hash, info.path.string());
  detail::hashing::hash_combine(hash, (u32)info.type);
  detail::hashing::hash_combine(hash, debug_mode);
  return hash;
}

}  // namespace

ShaderManager::LoadProgramResult ShaderManager::load_program(
    std::span<const ShaderCreateInfo> shader_create_infos, std::span<u64> out_create_info_hashes,
    bool force) {
  ZoneScoped;
  // TODO: thread safe
  LoadProgramResult result{};
  if (shader_create_infos.empty()) {
    return result;
  }

  util::fixed_vector<CompileToSpirvResult, LoadProgramResult::max_stages> spirv_binaries;

  u32 out_create_info_hash_idx = 0;
  for (const auto& cinfo : shader_create_infos) {
    auto full_path = cinfo.path.is_relative() ? shader_dir_ / cinfo.path : cinfo.path;
    size_t new_hash = hash_shader_info(cinfo, shader_debug_mode_);
    out_create_info_hashes[out_create_info_hash_idx++] = new_hash;

    auto& spirv_load_result = spirv_binaries.emplace_back();
    auto glsl_path = full_path.string() + ".glsl";
    auto spv_path = full_path.string() + '.' + std::to_string(new_hash) + ".spv";
    if (!std::filesystem::exists(glsl_path)) {
      LERROR("glsl file does not exist for shader: {}", glsl_path);
      return result;
    }

    bool need_new_spirv =
        force || !std::filesystem::exists(spv_path) ||
        std::filesystem::last_write_time(spv_path) < std::filesystem::last_write_time(glsl_path);
    if (!need_new_spirv) {
      auto it = spirv_include_timestamps_.find(spv_path);
      if (it == spirv_include_timestamps_.end()) {
        need_new_spirv = true;
      } else {
        for (auto& [filename, write_time] : it->second) {
          if (std::filesystem::last_write_time(filename) > write_time) {
            need_new_spirv = true;
            break;
          }
        }
      }
    }

    if (need_new_spirv) {
      std::unordered_set<std::string> all_included_files;
      if (!compile_glsl_to_spirv(glsl_path, convert_shader_stage(cinfo.type),
                                 spirv_load_result.binary_data, all_included_files,
                                 cinfo.defines)) {
        return result;
      }

      {
        std::ofstream file(spv_path, std::ios::binary);
        if (!file.is_open()) {
          return result;
        }
        file.write(reinterpret_cast<const char*>(spirv_load_result.binary_data.data()),
                   spirv_load_result.binary_data.size() * sizeof(u32));
      }

      std::vector<std::pair<std::string, std::filesystem::file_time_type>> write_times;
      write_times.reserve(all_included_files.size());
      for (const auto& included_file : all_included_files) {
        write_times.emplace_back(included_file, std::filesystem::last_write_time(included_file));
      }
      spirv_include_timestamps_[spv_path] = std::move(write_times);

    } else {
      load_entire_file(spv_path, spirv_load_result.binary_data);
    }
  }

  for (u64 i = 0; i < shader_create_infos.size(); i++) {
    const std::string& path = shader_create_infos[i].path.string();
    VkShaderModuleCreateInfo create_info{
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = spirv_binaries[i].binary_data.size() * sizeof(u32),
        .pCode = spirv_binaries[i].binary_data.data()};
    VK_CHECK(vkCreateShaderModule(device_, &create_info, nullptr, &result.modules[i]));
  }
  result.success = true;
  return result;
}

ShaderManager::ShaderManager(VkDevice device, std::filesystem::path shader_cache_dir,
                             OnDirtyFileFunc on_dirty_files_fn, std::filesystem::path shader_dir,
                             bool hot_reload)
    : file_watcher_(
          shader_dir, {".glsl"},
          [this](std::span<std::filesystem::path> dirty_files) {
            if (on_dirty_files_fn_) on_dirty_files_fn_(dirty_files);
          },
          std::chrono::milliseconds(250), hot_reload),
      on_dirty_files_fn_(std::move(on_dirty_files_fn)),
      shader_dir_(std::move(shader_dir)),
      shader_cache_dir_(std::move(shader_cache_dir)),
      shader_hash_cache_path_(shader_cache_dir_ / "shader_hash_cache.txt"),
      device_(device),
      hot_reload_(true) {
  init();
}
void ShaderManager::init() {
  if (!std::filesystem::exists(shader_cache_dir_)) {
    std::filesystem::create_directory(shader_cache_dir_);
  }

  {
    std::ifstream ifs(shader_cache_dir_ / include_graph_data_filename);
    if (ifs.is_open()) {
      u64 node_count;
      ifs >> node_count;
      std::string filename;
      std::string filename2;
      u64 included_by_count;
      while (ifs >> filename >> included_by_count) {
        auto it = include_graph_nodes_.find(filename);
        if (it == include_graph_nodes_.end()) {
          include_graph_nodes_[filename] = {};
          it = include_graph_nodes_.find(filename);
        }
        for (u64 i = 0; i < included_by_count; i++) {
          ifs >> filename2;
          it->second.emplace(filename2);
        }
      }
    }
  }
  {
    std::ifstream ifs(shader_cache_dir_ / spirv_include_write_times_filename);
    if (ifs.is_open()) {
      u64 num_spirv_files;
      ifs >> num_spirv_files;
      spirv_include_timestamps_.reserve(num_spirv_files);
      std::string spv_filename;
      u64 num_includes;
      std::string included_filename;
      u64 included_write_time;
      std::vector<std::pair<std::string, std::filesystem::file_time_type>> write_times;
      for (u64 spirv_file_i = 0; spirv_file_i < num_spirv_files; spirv_file_i++) {
        if (!(ifs >> spv_filename >> num_includes)) {
          break;
        }
        write_times.clear();
        write_times.reserve(num_includes);
        for (u64 include_i = 0; include_i < num_includes; include_i++) {
          if (!(ifs >> included_filename >> included_write_time)) {
            break;
          }
          write_times.emplace_back(
              included_filename,
              std::filesystem::file_time_type(std::chrono::nanoseconds(included_write_time)));
        }
        spirv_include_timestamps_.emplace(spv_filename, std::move(write_times));
      }
    }
  }

  glslang::InitializeProcess();

  if (hot_reload_) {
    file_watcher_.start();
  }
}

ShaderManager::~ShaderManager() {
  ZoneScoped;
  {
    std::ofstream ofs(shader_cache_dir_ / include_graph_data_filename);
    if (ofs.is_open()) {
      ofs << include_graph_nodes_.size() << '\n';
      for (const auto& [filename, included_bys] : include_graph_nodes_) {
        if (!std::filesystem::exists(filename)) continue;
        ofs << filename.string() << ' ' << included_bys.size() << ' ';
        for (const auto& included_by : included_bys) {
          if (!std::filesystem::exists(included_by)) continue;
          ofs << included_by.string() << ' ';
        }
        ofs << '\n';
      }
    }
  }
  {
    std::ofstream ofs(shader_cache_dir_ / spirv_include_write_times_filename);
    if (ofs.is_open()) {
      ofs << spirv_include_timestamps_.size() << '\n';
      for (auto& [spv_filename, write_times] : spirv_include_timestamps_) {
        ofs << spv_filename << ' ' << write_times.size() << '\n';
        for (auto& [included_filename, writetime] : write_times) {
          ofs << included_filename << ' '
              << static_cast<size_t>(writetime.time_since_epoch().count()) << ' ';
        }
      }
    }
  }
  glslang::FinalizeProcess();
}

namespace {
constexpr EShLanguage vk_shader_stage_to_glslang(VkShaderStageFlagBits stage) {
  switch (stage) {
    case VkShaderStageFlagBits::VK_SHADER_STAGE_VERTEX_BIT:
      return EShLanguage::EShLangVertex;
    case VkShaderStageFlagBits::VK_SHADER_STAGE_FRAGMENT_BIT:
      return EShLanguage::EShLangFragment;
    case VkShaderStageFlagBits::VK_SHADER_STAGE_COMPUTE_BIT:
      return EShLanguage::EShLangCompute;
    case VkShaderStageFlagBits::VK_SHADER_STAGE_RAYGEN_BIT_KHR:
      return EShLanguage::EShLangRayGen;
    case VkShaderStageFlagBits::VK_SHADER_STAGE_MISS_BIT_KHR:
      return EShLanguage::EShLangMiss;
    case VkShaderStageFlagBits::VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR:
      return EShLanguage::EShLangClosestHit;
    case VkShaderStageFlagBits::VK_SHADER_STAGE_ANY_HIT_BIT_KHR:
      return EShLanguage::EShLangAnyHit;
    case VkShaderStageFlagBits::VK_SHADER_STAGE_INTERSECTION_BIT_KHR:
      return EShLanguage::EShLangIntersect;
    default:
      return EShLanguage::EShLangCount;
  }
  return static_cast<EShLanguage>(-1);
}

}  // namespace

}  // namespace gfx::vk2

namespace {

class IncludeHandler final : public glslang::TShader::Includer {
 public:
  explicit IncludeHandler(const std::filesystem::path& source_path) {
    currentIncluderDir_ /= source_path.parent_path();
    source_path_ = source_path;
  }

  glslang::TShader::Includer::IncludeResult* includeLocal(
      const char* requested_source, [[maybe_unused]] const char* requesting_source,
      [[maybe_unused]] size_t include_depth) override {
    ZoneScoped;
    assert(std::filesystem::path(requested_source).is_relative());
    auto full_requested_source = currentIncluderDir_ / requested_source;
    currentIncluderDir_ = full_requested_source.parent_path();
    std::ifstream file{full_requested_source};
    if (!file) {
      throw std::runtime_error("File not found");
    }
    auto content_ptr = std::make_unique<std::string>(std::istreambuf_iterator<char>(file),
                                                     std::istreambuf_iterator<char>());
    auto* content = content_ptr.get();
    auto source_path_ptr = std::make_unique<std::string>(requested_source);
    contentStrings_.emplace_back(std::move(content_ptr));
    sourcePathStrings_.emplace_back(std::move(source_path_ptr));

    auto canonical_requested_source =
        std::filesystem::weakly_canonical(std::filesystem::path(full_requested_source)).string();
    all_included_files.emplace(canonical_requested_source);

    std::filesystem::path canonical_requesting_source =
        include_depth == 1
            ? source_path_
            : std::filesystem::weakly_canonical(std::filesystem::path(requesting_source));
    file_include_graph_reverse[full_requested_source].emplace(canonical_requesting_source);
    // file_include_graph[canonical_requesting_source].emplace(canonical_requested_source);

    return new glslang::TShader::Includer::IncludeResult(requested_source, content->c_str(),
                                                         content->size(), nullptr);
  }

  size_t number_of_path_components(std::filesystem::path path) {
    ZoneScoped;
    size_t parents = 0;
    while (!path.empty()) {
      parents++;
      path = path.parent_path();
    }
    return parents > 0 ? parents - 1 : 0;
  }

  void releaseInclude(glslang::TShader::Includer::IncludeResult* data) override {
    ZoneScoped;
    for (size_t i = 0; i < number_of_path_components(data->headerName); i++) {
      currentIncluderDir_ = currentIncluderDir_.parent_path();
    }
    delete data;
  }

  // std::unordered_map<std::string, std::unordered_set<std::string>> file_include_graph;
  std::unordered_map<std::string, std::unordered_set<std::string>> file_include_graph_reverse;
  std::unordered_set<std::string> all_included_files;

 private:
  // Acts like a stack that we "push" path components to when include{Local, System} are invoked,
  // and "pop" when releaseInclude is invoked
  std::filesystem::path currentIncluderDir_;
  std::filesystem::path source_path_;
  std::vector<std::unique_ptr<std::string>> contentStrings_;
  std::vector<std::unique_ptr<std::string>> sourcePathStrings_;
};

// returns true on success

bool load_entire_file(const std::string& path, std::vector<uint32_t>& result) {
  ZoneScoped;
  std::ifstream file(path, std::ios::binary);
  if (!file.is_open()) {
    // TODO: maybe not critical lol
    LCRITICAL("failed to open file: {}", path);
    exit(1);
  }
  file.seekg(0, std::ios::end);
  auto len = file.tellg();
  result.resize(len / sizeof(uint32_t));
  file.seekg(0, std::ios::beg);
  file.read(reinterpret_cast<char*>(result.data()), len);
  file.close();
  return true;
}

std::optional<std::string> load_entire_file(const std::string& path) {
  std::ifstream file(path, std::ios::in | std::ios::binary);
  if (!file) {
    return std::nullopt;
  }

  std::ostringstream contents;
  contents << file.rdbuf();
  return contents.str();
}

}  // namespace

namespace gfx::vk2 {

bool ShaderManager::compile_glsl_to_spirv(const std::string& path, VkShaderStageFlagBits stage,
                                          std::vector<u32>& out_binary,
                                          std::unordered_set<std::string>& all_included_files,
                                          std::span<const std::string> defines) {
  ZoneScoped;
  LINFO("compiling glsl: {}", path);
  if (!std::filesystem::exists(path)) {
    LERROR("path does not exist: {}", path);
  }
  constexpr auto compiler_messages =
      EShMessages(EShMsgSpvRules | EShMsgVulkanRules | EShMsgDebugInfo);
  auto glslang_stage = vk_shader_stage_to_glslang(stage);

  glslang::TShader shader(glslang_stage);
  auto glsl_text_result = load_entire_file(path);
  if (!glsl_text_result.has_value()) {
    LERROR("failed to read file: {}", path);
    return false;
  }
  const auto& glsl_text = glsl_text_result.value();
  const char* sources[] = {glsl_text.c_str()};
  int lengths[] = {static_cast<int>(glsl_text.size())};
  const char* names[] = {path.c_str()};
  shader.setStringsWithLengthsAndNames(sources, lengths, names, 1);
  shader.setEnvInput(glslang::EShSourceGlsl, glslang_stage, glslang::EShClientVulkan, 100);
  shader.setEnvClient(glslang::EShClientVulkan, glslang::EShTargetVulkan_1_2);
  shader.setEnvTarget(glslang::EShTargetLanguage::EShTargetSpv, glslang::EShTargetSpv_1_5);
  std::string preamble;
  if (defines.size()) {
    std::stringstream ss;
    for (const auto& define : defines) {
      ss << define << "\n";
    }
    preamble = ss.str();
  }

  if (preamble.size()) {
    shader.setPreamble(preamble.c_str());
  }

  shader.setOverrideVersion(460);

  IncludeHandler includer(path);
  bool success = shader.parse(GetDefaultResources(), 460, EProfile::ECoreProfile, false, false,
                              compiler_messages, includer);
  if (!success) {
    LERROR("path: {}", path);
    LERROR("Failed to parse GLSL shader:\nShader info log:\n{}\nInfo Debug log:\n{}",
           shader.getInfoLog(), shader.getInfoDebugLog());
    return false;
  }

  all_included_files.insert(includer.all_included_files.begin(), includer.all_included_files.end());
  auto& dep_graph = includer.file_include_graph_reverse;
  for (const auto& [filename, included_bys] : dep_graph) {
    auto it = include_graph_nodes_.find(filename);
    if (it == include_graph_nodes_.end()) {
      include_graph_nodes_[filename] = {};
      it = include_graph_nodes_.find(filename);
    }
    for (const auto& included_by : included_bys) {
      it->second.emplace(included_by);
    }
  }

  glslang::TProgram program;
  program.addShader(&shader);

  if (!program.link(compiler_messages)) {
    LERROR("Failed to link GLSL program:\nProgram info log:\n{}\nInfo Debug log:\n{}",
           program.getInfoLog(), program.getInfoDebugLog());
    return false;
  }

  // Equivalent to -gVS
  // https://github.com/KhronosGroup/glslang/blob/vulkan-sdk-1.3.283.0/StandAlone/StandAlone.cpp#L998-L1016
  // TODO: only in debug?
  auto options = glslang::SpvOptions{
      .generateDebugInfo = true,
      .stripDebugInfo = false,
      .disableOptimizer = true,
      .optimizeSize = false,
      .disassemble = true,
      .emitNonSemanticShaderDebugInfo = true,
      .emitNonSemanticShaderDebugSource = true,
  };

  {
    ZoneScopedN("GlslangToSpv");
    out_binary.clear();
    spv::SpvBuildLogger logger;
    glslang::GlslangToSpv(*shader.getIntermediate(), out_binary, &logger, &options);
    auto logger_messages = logger.getAllMessages();
    if (!logger_messages.empty()) {
      LINFO("spv logger messages: {}", logger_messages);
    }
  }
  return true;
}

void ShaderManager::invalidate_cache() { include_graph_nodes_.clear(); }

}  // namespace gfx::vk2
