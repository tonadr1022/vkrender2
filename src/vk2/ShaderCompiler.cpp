#include "ShaderCompiler.hpp"

#include <glslang/Public/ResourceLimits.h>
#include <glslang/Public/ShaderLang.h>
#include <glslang/SPIRV/GlslangToSpv.h>
#include <spirv-reflect/spirv_reflect.h>
#include <volk.h>
#include <vulkan/vk_enum_string_helper.h>
#include <vulkan/vulkan_core.h>

#include <cassert>
#include <filesystem>
#include <fstream>
#include <print>
#include <string>
#include <tracy/Tracy.hpp>

#include "Common.hpp"
#include "Logger.hpp"
#include "vk2/Hash.hpp"
#include "vk2/VkCommon.hpp"

// heavy inspiration/stealing from:
// https://github.com/JuanDiegoMontoya/Frogfood/blob/main/src/Fvog/Shader2.cpp

namespace {

size_t number_of_path_components(std::filesystem::path path) {
  size_t parents = 0;
  while (!path.empty()) {
    parents++;
    path = path.parent_path();
  }
  return parents > 0 ? parents - 1 : 0;  // The path will contain a filename, which we will ignore
}

std::string load_file2(const std::filesystem::path& path) {
  std::ifstream file{path};
  if (!file) {
    throw std::runtime_error("File not found");
  }
  return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
}

class IncludeHandler final : public glslang::TShader::Includer {
 public:
  explicit IncludeHandler(const std::filesystem::path& source_path) {
    currentIncluderDir_ /= source_path.parent_path();
  }

  glslang::TShader::Includer::IncludeResult* includeLocal(
      const char* requested_source, [[maybe_unused]] const char* requesting_source,
      [[maybe_unused]] size_t include_depth) override {
    ZoneScoped;
    assert(std::filesystem::path(requested_source).is_relative());
    auto full_requested_source = currentIncluderDir_ / requested_source;
    currentIncluderDir_ = full_requested_source.parent_path();

    auto content_ptr = std::make_unique<std::string>(load_file2(full_requested_source));
    auto* content = content_ptr.get();
    auto source_path_ptr = std::make_unique<std::string>(requested_source);
    // auto sourcePath = sourcePathPtr.get();

    contentStrings_.emplace_back(std::move(content_ptr));
    sourcePathStrings_.emplace_back(std::move(source_path_ptr));

    return new glslang::TShader::Includer::IncludeResult(requested_source, content->c_str(),
                                                         content->size(), nullptr);
  }

  void releaseInclude(glslang::TShader::Includer::IncludeResult* data) override {
    for (size_t i = 0; i < number_of_path_components(data->headerName); i++) {
      currentIncluderDir_ = currentIncluderDir_.parent_path();
    }
    delete data;
  }

 private:
  // Acts like a stack that we "push" path components to when include{Local, System} are invoked,
  // and "pop" when releaseInclude is invoked
  std::filesystem::path currentIncluderDir_;
  std::vector<std::unique_ptr<std::string>> contentStrings_;
  std::vector<std::unique_ptr<std::string>> sourcePathStrings_;
};

bool load_shader_bytes(const std::string& path, std::vector<uint32_t>& result) {
  std::ifstream file(path, std::ios::binary);
  if (!file.is_open()) {
    LERROR("failed to open file: {}", path);
    return false;
  }
  file.seekg(0, std::ios::end);
  auto len = file.tellg();
  result.resize(len / sizeof(uint32_t));
  file.seekg(0, std::ios::beg);
  file.read(reinterpret_cast<char*>(result.data()), len);
  file.close();
  return true;
}

std::optional<std::string> load_file(const std::filesystem::path& path) {
  std::ifstream file{path, std::ios::binary};
  if (!file) {
    return std::nullopt;
  }

  file.seekg(0, std::ios::end);
  std::size_t size = file.tellg();
  file.seekg(0, std::ios::beg);

  std::string content(size, '\0');
  if (!file.read(content.data(), size)) {
    return std::nullopt;
  }
  return content;
}

struct DescriptorSetLayoutData {
  uint32_t set_number;
  VkDescriptorSetLayoutCreateInfo create_info;
  std::vector<VkDescriptorSetLayoutBinding> bindings;
};
struct ShaderCreateData {
  // NOTE: expand these if ever needed
  static constexpr const int max_pc_ranges = 4;
  std::array<DescriptorSetLayoutData, 4> set_layouts;
  uint32_t set_layout_cnt{0};
  std::array<VkPushConstantRange, max_pc_ranges> pc_ranges;
  uint32_t pc_range_cnt{0};
  VkShaderStageFlags shader_stages;
  uint32_t shader_stage_cnt{0};
};

VkPipelineLayout create_layout(VkDevice device, ShaderCreateData& data,
                               vk2::DescriptorSetLayoutCache& layout_cache) {
  // for each set, merge the bindings together
  constexpr int max_descriptor_sets = 4;
  constexpr int max_set_layout_bindings = 100;

  std::array<DescriptorSetLayoutData, max_descriptor_sets> merged_layout_datas{};
  std::array<VkDescriptorSetLayout, max_descriptor_sets> combined_set_layouts{};
  std::array<uint64_t, max_descriptor_sets> merged_layout_hashes{};
  // go through each of the possible set indices
  for (uint32_t set_number = 0; set_number < max_descriptor_sets; set_number++) {
    // set merged layout info
    auto& merged_layout = merged_layout_datas[set_number];
    merged_layout.set_number = set_number;

    std::array<std::pair<VkDescriptorSetLayoutBinding, bool>, max_set_layout_bindings>
        used_bindings{};
    for (uint32_t unmerged_set_ly_idx = 0; unmerged_set_ly_idx < data.set_layout_cnt;
         unmerged_set_ly_idx++) {
      const auto& unmerged_set_layout = data.set_layouts[unmerged_set_ly_idx];
      // if the unmerged set corresponds, merge it
      if (unmerged_set_layout.set_number != set_number) continue;

      // for each of the bindings, if it's already used, add the shader stage since it doesn't need
      // to be duplicated, otherwise mark used and set it
      for (const auto& binding : unmerged_set_layout.bindings) {
        if (binding.binding >= max_set_layout_bindings) {
          LERROR("exceed max layout bindings for pipeline with shader");
          return nullptr;
        }

        if (used_bindings[binding.binding].second) {
          used_bindings[binding.binding].first.stageFlags |= binding.stageFlags;
        } else {
          used_bindings[binding.binding].second = true;
          used_bindings[binding.binding].first = binding;
        }
      }
    }
    // add the used bindings to the merged layout
    for (uint32_t i = 0; i < max_set_layout_bindings; i++) {
      if (used_bindings[i].second) {
        merged_layout.bindings.push_back(used_bindings[i].first);
      }
    }
    std::ranges::sort(merged_layout.bindings,
                      [](VkDescriptorSetLayoutBinding& a, VkDescriptorSetLayoutBinding& b) {
                        return a.binding < b.binding;
                      });
    merged_layout.create_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    merged_layout.create_info.bindingCount = merged_layout.bindings.size();
    merged_layout.create_info.pBindings = merged_layout.bindings.data();
    merged_layout.create_info.flags = 0;
    merged_layout.create_info.pNext = nullptr;
    if (merged_layout.create_info.bindingCount > 0) {
      auto res = layout_cache.create_layout(device, merged_layout.create_info);
      merged_layout_hashes[set_number] = res.hash;
      combined_set_layouts[set_number] = res.layout;
    } else {
      merged_layout_hashes[set_number] = 0;
      combined_set_layouts[set_number] = VK_NULL_HANDLE;
    }
  }

  // use dummy layouts for the case when sets are akin to: [null, 1, null, null]
  // set cnt is the last valid set, so in the above case, 1
  uint32_t set_cnt = 0;
  for (uint32_t i = 0; i < max_descriptor_sets; i++) {
    if (combined_set_layouts[i] != VK_NULL_HANDLE) {
      set_cnt = i + 1;
    } else {
      combined_set_layouts[i] = layout_cache.dummy_layout();
    }
  }

  VkPipelineLayoutCreateInfo pipeline_layout_info{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      // TODO: flags?
      .flags = 0,
      .setLayoutCount = set_cnt,
      .pSetLayouts = set_cnt == 0 ? nullptr : combined_set_layouts.data(),
      .pushConstantRangeCount = data.pc_range_cnt,
      .pPushConstantRanges = data.pc_ranges.data(),
  };
  VkPipelineLayout out_layout;
  VK_CHECK(vkCreatePipelineLayout(device, &pipeline_layout_info, nullptr, &out_layout));
  return out_layout;
}

}  // namespace

namespace vk2 {

namespace {
ShaderManager* g_shader_compiler{};
}

ShaderManager& ShaderManager::get() {
  assert(g_shader_compiler);
  return *g_shader_compiler;
}

void ShaderManager::init(VkDevice device) {
  assert(!g_shader_compiler);
  g_shader_compiler = new ShaderManager;
  g_shader_compiler->init_impl(device);
}

void ShaderManager::shutdown() {
  assert(g_shader_compiler);
  g_shader_compiler->shutdown_impl();
  delete g_shader_compiler;
}

void ShaderManager::init_impl(VkDevice device) {
  device_ = device;
  layout_cache_.init(device);
  glslang::InitializeProcess();
}

void ShaderManager::shutdown_impl() {
  layout_cache_.shutdown();
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

// returns true on success
bool compile_glsl_to_spirv(std::string path, VkShaderStageFlagBits stage,
                           std::vector<u32>& out_binary) {
  ZoneScoped;
  if (!std::filesystem::exists(path)) {
    LERROR("path does not exist: {}", path);
  }
  constexpr auto compiler_messages =
      EShMessages(EShMessages::EShMsgSpvRules | EShMessages::EShMsgVulkanRules);
  auto glslang_stage = vk_shader_stage_to_glslang(stage);

  glslang::TShader shader(glslang_stage);
  auto glsl_text_result = load_file(path);
  if (!glsl_text_result.has_value()) {
    LERROR("failed to read file: {}", path);
    return false;
  }
  const auto& glsl_text = glsl_text_result.value();
  // TODO: fix this
  char* shader_source = new char[glsl_text.size() + 1];
  std::memcpy(shader_source, glsl_text.c_str(), glsl_text.size());
  shader_source[glsl_text.size()] = '\0';
  int ss = strlen(shader_source);
  shader.setStringsWithLengths(&shader_source, &ss, 1);
  shader.setEnvInput(glslang::EShSourceGlsl, glslang_stage, glslang::EShClientVulkan, 100);
  shader.setEnvClient(glslang::EShClientVulkan, glslang::EShTargetVulkan_1_3);
  shader.setEnvTarget(glslang::EShTargetLanguage::EShTargetSpv, glslang::EShTargetSpv_1_6);
  std::string preamble = "#extension GL_GOOGLE_include_directive : enable\n";
  shader.setPreamble(preamble.c_str());
  shader.setOverrideVersion(460);

  IncludeHandler includer(path);
  bool success = shader.parse(GetDefaultResources(), 460, EProfile::ECoreProfile, false, false,
                              compiler_messages, includer);
  if (!success) {
    LERROR("Failed to parse GLSL shader:\nShader info log:\n{}\nInfo Debug log:\n{}",
           shader.getInfoLog(), shader.getInfoDebugLog());
    return false;
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
      .generateDebugInfo = true, .stripDebugInfo = false,
      // .disableOptimizer = true,
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

}  // namespace

#define TRY(x)                               \
  do {                                       \
    if ((x) != SPV_REFLECT_RESULT_SUCCESS) { \
      LERROR("spirv reflection failed");     \
      return false;                          \
    }                                        \
  } while (0)

namespace {
bool reflect_shader(std::vector<u32>& binary, ShaderCreateData& out_data) {
  spv_reflect::ShaderModule refl_module{binary};
  u32 cnt;
  TRY(refl_module.EnumerateDescriptorSets(&cnt, nullptr));
  std::vector<SpvReflectDescriptorSet*> refl_sets(cnt);
  TRY(refl_module.EnumerateDescriptorSets(&cnt, refl_sets.data()));

  TRY(refl_module.EnumeratePushConstantBlocks(&cnt, nullptr));
  std::vector<SpvReflectBlockVariable*> refl_pc_blocks(cnt);
  TRY(refl_module.EnumeratePushConstantBlocks(&cnt, refl_pc_blocks.data()));
  if (cnt > 1) {
    LERROR("shader module has > 1 push constant block, not supported.");
    return false;
  }
  if (cnt > 0) {
    out_data.pc_ranges[out_data.pc_range_cnt++] = VkPushConstantRange{
        .stageFlags = static_cast<VkShaderStageFlags>(refl_module.GetShaderStage()),
        .offset = refl_pc_blocks[0]->offset,
        .size = refl_pc_blocks[0]->size};
  }

  for (const SpvReflectDescriptorSet* set : refl_sets) {
    const SpvReflectDescriptorSet& refl_set = *set;
    DescriptorSetLayoutData& layout = out_data.set_layouts[out_data.set_layout_cnt++];
    layout.bindings.resize(refl_set.binding_count);
    for (uint32_t binding_idx = 0; binding_idx < refl_set.binding_count; binding_idx++) {
      auto& binding = layout.bindings[binding_idx];
      const auto& refl_binding = *(refl_set.bindings[binding_idx]);
      binding.binding = refl_binding.binding;
      binding.descriptorType = static_cast<VkDescriptorType>(refl_binding.descriptor_type);
      binding.descriptorCount = 1;
      for (uint32_t dim_idx = 0; dim_idx < refl_binding.array.dims_count; dim_idx++) {
        binding.descriptorCount *= refl_binding.array.dims[dim_idx];
      }
      binding.stageFlags = static_cast<VkShaderStageFlagBits>(refl_module.GetShaderStage());
    }
    layout.set_number = refl_set.set;
    layout.create_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layout.create_info.bindingCount = refl_set.binding_count;
    // NOTE: must move the vector and not copy, or the pBindings will be invalidated
    layout.create_info.pBindings = layout.bindings.data();
  }
  out_data.shader_stages |= refl_module.GetShaderStage();
  out_data.shader_stage_cnt++;
  return true;
}

}  // namespace

// VkShaderModule ShaderManager::load_shader(const std::filesystem::path& path,
//                                           VkShaderStageFlagBits stage) {
//   // TODO: flag for whether to compile to spirv or fail if not exists. don't always need glsl
//   // compilation?
//
//   CompileToSpirvResult spirv;
//   if (!get_spirv_binary(path, stage, spirv)) {
//     return nullptr;
//   }
//   VkShaderModule module;
//   VkShaderModuleCreateInfo create_info{.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
//                                        .codeSize = spirv.binary_data.size() * sizeof(u32),
//                                        .pCode = spirv.binary_data.data()};
//   vkCreateShaderModule(device_, &create_info, nullptr, &module);
//
//   return module;
// }

bool ShaderManager::get_spirv_binary(const std::filesystem::path& path, VkShaderStageFlagBits stage,
                                     CompileToSpirvResult& result) {
  assert(path.extension() != ".spv" && path.extension() != ".glsl");
  auto glsl_path = std::filesystem::path(path.string() + ".glsl");
  auto spv_path = std::filesystem::path(path.string() + ".spv");

  if (!std::filesystem::exists(glsl_path)) {
    LERROR("glsl file does not exist for shader: {}", glsl_path.string());
    return false;
  }

  // if spirv is older than glsl, need new spirv
  bool needs_new_spirv =
      !std::filesystem::exists(spv_path) ||
      std::filesystem::last_write_time(spv_path) < std::filesystem::last_write_time(glsl_path);
  if (needs_new_spirv) {
    bool success = compile_glsl_to_spirv(glsl_path, stage, result.binary_data);
    if (!success) {
      return false;
    }
    std::ofstream file(spv_path, std::ios::binary);
    if (!file.is_open()) {
      LERROR("failed to open file");
      return false;
    }
    file.write(reinterpret_cast<const char*>(result.binary_data.data()),
               result.binary_data.size() * sizeof(u32));
  }
  return load_shader_bytes(spv_path, result.binary_data);
}

ShaderManager::LoadShaderResult ShaderManager::load_shader(
    std::span<ShaderCreateInfo> shader_create_infos) {
  // TODO: thread safe
  LoadShaderResult result{};
  if (shader_create_infos.empty()) {
    LERROR("ShaderManager::load_shader: no shaders");
    return result;
  }
  assert(shader_create_infos.size() < LoadShaderResult::max_stages);
  // std::array<u64, max_stages> hashes{};
  // bool needs_new_spirv = false;
  // for (u64 i = 0; i < shader_create_infos.size(); i++) {
  //   hashes[i] = hash(shader_create_infos[i].path, shader_create_infos[i].stage);
  //   auto it = shader_metadata_.find(hashes[i]);
  //   if (it == shader_metadata_.end()) {
  //     needs_new_spirv = true;
  //     break;
  //   }
  //
  //   auto glsl_path = std::filesystem::path(shader_create_infos[i].path.string() + ".glsl");
  //   if (it->second.last_spirv_write < std::filesystem::last_write_time(glsl_path)) {
  //     needs_new_spirv = true;
  //     break;
  //   }
  // }

  std::array<CompileToSpirvResult, LoadShaderResult::max_stages> spirv_binaries;
  for (u64 i = 0; i < shader_create_infos.size(); i++) {
    if (!get_spirv_binary(shader_create_infos[i].path, shader_create_infos[i].stage,
                          spirv_binaries[i])) {
      return result;
    }
  }

  ShaderCreateData data;
  for (u64 i = 0; i < shader_create_infos.size(); i++) {
    if (!reflect_shader(spirv_binaries[i].binary_data, data)) {
      LERROR("failed to reflect spirv binary: {}", shader_create_infos[i].path.string());
      return result;
    }
  }

  result.module_cnt = shader_create_infos.size();
  for (u64 i = 0; i < shader_create_infos.size(); i++) {
    VkShaderModuleCreateInfo create_info{
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = spirv_binaries[i].binary_data.size() * sizeof(u32),
        .pCode = spirv_binaries[i].binary_data.data()};
    VK_CHECK(vkCreateShaderModule(device_, &create_info, nullptr, &result.modules[i]));
    LINFO("made modules");
  }

  result.layout = create_layout(device_, data, layout_cache_);
  if (!result.layout) {
    LERROR("Failed to create pipeline layout for pipeline with shader stage 0: {}",
           shader_create_infos[0].path.string());
  }
  LINFO("made layout");
  return result;
}

u64 ShaderManager::hash(const std::filesystem::path& path, VkShaderStageFlagBits stage) {
  auto v = std::make_tuple(path.string(), stage);
  return vk2::detail::hashing::hash<decltype(v)>{}(v);
}

VkShaderModule ShaderManager::load_shader(const std::filesystem::path& path,
                                          VkShaderStageFlagBits stage) {
  CompileToSpirvResult res;
  get_spirv_binary(path, stage, res);
  VkShaderModuleCreateInfo info{.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
  info.codeSize = res.binary_data.size() * sizeof(u32);
  info.pCode = res.binary_data.data();
  VkShaderModule mod;
  VK_CHECK(vkCreateShaderModule(device_, &info, nullptr, &mod));
  return mod;
}

}  // namespace vk2
