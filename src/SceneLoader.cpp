#include "SceneLoader.hpp"

#include <vulkan/vulkan_core.h>

#include <fastgltf/core.hpp>
#include <fastgltf/glm_element_traits.hpp>
#include <fastgltf/tools.hpp>
#include <fastgltf/types.hpp>
#include <tracy/Tracy.hpp>

#include "Logger.hpp"
#include "vk2/Buffer.hpp"

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

void populate_indices(const fastgltf::Primitive& primitive, const fastgltf::Asset& gltf,
                      std::vector<uint32_t>& indices) {
  if (!primitive.indicesAccessor.has_value()) {
    return;
  }
  const auto& index_accessor = gltf.accessors[primitive.indicesAccessor.value()];
  indices.resize(indices.size() + index_accessor.count);
  size_t i = 0;
  fastgltf::iterateAccessor<u32>(gltf, index_accessor,
                                 [&](uint32_t index) { indices[i++] = index; });
}

void populate_vertices(const fastgltf::Primitive& primitive, const fastgltf::Asset& gltf,
                       std::vector<Vertex>& vertices) {
  const auto* pos_attrib = primitive.findAttribute("POSITION");
  if (pos_attrib == primitive.attributes.end()) {
    return;
  }
  const auto& pos_accessor = gltf.accessors[pos_attrib->accessorIndex];
  vertices.resize(vertices.size() + pos_accessor.count);
  size_t i = 0;
  for (glm::vec3 pos : fastgltf::iterateAccessor<glm::vec3>(gltf, pos_accessor)) {
    vertices[i++].pos = pos;
  }

  const auto* uv_attrib = primitive.findAttribute("TEXCOORD_0");
  if (uv_attrib != primitive.attributes.end()) {
    const auto& accessor = gltf.accessors[uv_attrib->accessorIndex];
    i = 0;
    for (glm::vec2 uv : fastgltf::iterateAccessor<glm::vec2>(gltf, accessor)) {
      vertices[i++].uv_x = uv.x;
      vertices[i++].uv_y = uv.y;
    }
  }

  const auto* normal_attrib = primitive.findAttribute("NORMAL");
  if (normal_attrib != primitive.attributes.end()) {
    const auto& normal_accessor = gltf.accessors[normal_attrib->accessorIndex];
    i = 0;
    for (glm::vec3 normal : fastgltf::iterateAccessor<glm::vec3>(gltf, normal_accessor)) {
      vertices[i++].normal = normal;
    }
  }
}

}  // namespace

std::optional<LoadedSceneBaseData> load_gltf_base(const std::filesystem::path& path) {
  std::optional<LoadedSceneBaseData> result = std::nullopt;
  if (!std::filesystem::exists(path)) {
    LINFO("Failed to load glTF: directory {} does not exist", path.string());
    return result;
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
    return result;
  }

  fastgltf::Asset gltf = std::move(load_ret.get());
  result = LoadedSceneBaseData{};

  {
    ZoneScopedN("gltf load samplers");
    result->samplers.reserve(gltf.samplers.size());
    for (auto& sampler : gltf.samplers) {
      result->samplers.emplace_back(VkSamplerCreateInfo{
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

  // meshes
  for (const auto& gltf_mesh : gltf.meshes) {
    for (const auto& gltf_prim : gltf_mesh.primitives) {
      populate_indices(gltf_prim, gltf, result->indices);
      populate_vertices(gltf_prim, gltf, result->vertices);
    }
  }

  // get a staging buffer (for now create, later: reuse)

  return result;
}

std::optional<LoadedSceneData> load_gltf(const std::filesystem::path& path) {
  auto base_scene_data_ret = load_gltf_base(path);
  if (!base_scene_data_ret.has_value()) {
    return {};
  }
  LoadedSceneBaseData& base_scene_data = base_scene_data_ret.value();
  u64 vertices_size = base_scene_data.vertices.size() * sizeof(Vertex);
  u64 indices_size = base_scene_data.indices.size() * sizeof(u32);
  vk2::Buffer staging_buf = vk2::create_staging_buffer(vertices_size + indices_size);
  assert(staging_buf.mapped_data());
  memcpy(staging_buf.mapped_data(), base_scene_data.vertices.data(), vertices_size);
  memcpy((char*)staging_buf.mapped_data() + vertices_size, base_scene_data.indices.data(),
         indices_size);
  vk2::Buffer vertex_buffer = vk2::Buffer{vk2::BufferCreateInfo{
      .size = vertices_size,
      .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
      .alloc_flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT}};
  vk2::Buffer index_buffer = vk2::Buffer{vk2::BufferCreateInfo{
      .size = indices_size,
      .usage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
  }};

  return LoadedSceneData{.vertex_buffer = std::move(vertex_buffer),
                         .index_buffer = std::move(index_buffer),
                         .samplers = std::move(base_scene_data.samplers)};
}

}  // namespace gfx
