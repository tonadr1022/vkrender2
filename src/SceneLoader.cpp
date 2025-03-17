#include "SceneLoader.hpp"

#include <vulkan/vulkan_core.h>

#include <tracy/Tracy.hpp>

#include "Logger.hpp"
#include "fastgltf/core.hpp"
#include "vk2/Texture.hpp"

namespace gfx {

namespace {

VkFilter gltf_to_vk_filter(fastgltf::Filter filter) {
  switch (filter) {
    // nearest
    case fastgltf::Filter::Nearest:
    case fastgltf::Filter::NearestMipMapNearest:
    case fastgltf::Filter::NearestMipMapLinear:
      return VK_FILTER_NEAREST;

      // linear
    case fastgltf::Filter::Linear:
    case fastgltf::Filter::LinearMipMapLinear:
    case fastgltf::Filter::LinearMipMapNearest:
    default:
      return VK_FILTER_LINEAR;
  }
}

VkSamplerMipmapMode gltf_to_vk_mipmap_mode(fastgltf::Filter filter) {
  switch (filter) {
    case fastgltf::Filter::NearestMipMapNearest:
    case fastgltf::Filter::LinearMipMapNearest:
      return VK_SAMPLER_MIPMAP_MODE_LINEAR;
    case fastgltf::Filter::NearestMipMapLinear:
    case fastgltf::Filter::LinearMipMapLinear:
    default:
      return VK_SAMPLER_MIPMAP_MODE_NEAREST;
  }
}

VkSamplerAddressMode gltf_to_vk_wrap_mode(fastgltf::Wrap wrap) {
  switch (wrap) {
    case fastgltf::Wrap::ClampToEdge:
      return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    case fastgltf::Wrap::MirroredRepeat:
      return VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
    default:
      return VK_SAMPLER_ADDRESS_MODE_REPEAT;
  }
}

}  // namespace

std::optional<LoadedSceneData> load_gltf(const std::filesystem::path& path) {
  if (!std::filesystem::exists(path)) {
    LINFO("Failed to load glTF: directory {} does not exist", path.string());
    return std::nullopt;
  }

  constexpr auto supported_extensions = fastgltf::Extensions::KHR_mesh_quantization |
                                        fastgltf::Extensions::KHR_texture_transform |
                                        fastgltf::Extensions::KHR_materials_variants |
                                        fastgltf::Extensions::KHR_materials_emissive_strength;

  constexpr auto gltf_options =
      fastgltf::Options::DontRequireValidAssetMember | fastgltf::Options::AllowDouble |
      fastgltf::Options::LoadExternalBuffers | fastgltf::Options::GenerateMeshIndices |
      fastgltf::Options::DecomposeNodeMatrices;

  fastgltf::Parser parser{supported_extensions};
  auto gltf_file = fastgltf::GltfDataBuffer::FromPath(path);
  auto parent_path = std::filesystem::path(path).parent_path();
  auto load_ret = parser.loadGltf(gltf_file.get(), parent_path, gltf_options);
  if (!load_ret) {
    LINFO("Failed to load glTF\n\tpath: {}\n\terror: {}\n", path.string(),
          fastgltf::getErrorMessage(load_ret.error()));
    return std::nullopt;
  }

  fastgltf::Asset gltf = std::move(load_ret.get());

  {
    ZoneScopedN("gltf load samplers");
    std::vector<vk2::Sampler> samplers;
    samplers.reserve(gltf.samplers.size());
    for (auto& sampler : gltf.samplers) {
      samplers.emplace_back(VkSamplerCreateInfo{
          .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
          .magFilter = gltf_to_vk_filter(sampler.magFilter.value_or(fastgltf::Filter::Linear)),
          .minFilter = gltf_to_vk_filter(sampler.magFilter.value_or(fastgltf::Filter::Linear)),
          .mipmapMode = gltf_to_vk_mipmap_mode(
              sampler.minFilter.value_or(fastgltf::Filter::LinearMipMapLinear)),
          .addressModeU = gltf_to_vk_wrap_mode(sampler.wrapS),
          .addressModeV = gltf_to_vk_wrap_mode(sampler.wrapT),
          .addressModeW = gltf_to_vk_wrap_mode(fastgltf::Wrap::Repeat),
          .minLod = -1000,
          .maxLod = 100,
          .borderColor = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK,
      });
    }
  }

  return std::nullopt;
}

}  // namespace gfx
