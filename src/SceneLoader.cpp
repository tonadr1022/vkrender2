#include "SceneLoader.hpp"

#include <ktx.h>
#include <ktxvulkan.h>
#include <volk.h>
#include <vulkan/vk_enum_string_helper.h>
#include <vulkan/vulkan_core.h>

#include <cstddef>
#include <cstdint>
#include <cstring>

#include "Types.hpp"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wredundant-move"
#include <fastgltf/core.hpp>
#include <fastgltf/glm_element_traits.hpp>
#include <fastgltf/tools.hpp>
#include <fastgltf/types.hpp>
#pragma GCC diagnostic pop
#include <future>
#include <optional>

#include "BS_thread_pool.hpp"
#include "Scene.hpp"
#include "StateTracker.hpp"
#include "ThreadPool.hpp"
#include "core/Timer.hpp"
#include "shaders/common.h.glsl"
#include "vk2/Device.hpp"

// #include "ThreadPool.hpp"

#include "stb_image.h"
#include "vk2/Texture.hpp"

#define GLM_ENABLE_EXPERIMENTAL
#include <mikktspace.h>

#include <glm/gtx/quaternion.hpp>
#include <tracy/Tracy.hpp>

#include "core/Logger.hpp"
#include "vk2/Buffer.hpp"

// ktx loading inspired/ripped from:
// https://github.com/JuanDiegoMontoya/Frogfood/blob/be82e484baab02b7ce3e80d36eb7c9291d97ebcb/src/Fvog/detail/ApiToEnum2.cpp#L4
namespace {

enum class PBRImageUsage : u8 {
  BaseColor,
  Normal,
  MetallicRoughness,
  OccRoughnessMetallic,
  Emissive,
  Occlusion
};

}  // namespace
namespace gfx {

namespace {

void calc_aabb(AABB& aabb, const void* vertices, size_t len, size_t stride, size_t offset) {
  aabb.min = vec3{std::numeric_limits<float>::max()};
  aabb.max = vec3{std::numeric_limits<float>::lowest()};
  for (size_t i = 0; i < len; i++) {
    const glm::vec3& pos = *reinterpret_cast<const glm::vec3*>(static_cast<const char*>(vertices) +
                                                               (i * stride) + offset);
    aabb.min = glm::min(aabb.min, pos);
    aabb.max = glm::max(aabb.max, pos);
  }
}

void load_tangents(const std::string& path, std::vector<Vertex>& vertices) {
  ZoneScoped;
  std::ifstream file(path, std::ios::binary);
  assert(file.is_open());

  file.seekg(0, std::ios::end);
  auto len = file.tellg();
  file.seekg(0, std::ios::beg);
  std::vector<vec3> tangents(len / sizeof(vec3));
  file.read(reinterpret_cast<char*>(tangents.data()), len);
  file.close();
  if (tangents.size() != vertices.size()) {
    LERROR("invalid tangents loaded");
    return;
  }
  u64 i = 0;
  for (auto& vertex : vertices) {
    vertex.tangent = vec4{tangents[i++], 0.};
  }
}

void save_tangents(const std::string& path, const std::vector<Vertex>& vertices) {
  ZoneScoped;
  std::ofstream file(path, std::ios::binary);
  assert(file.is_open());
  std::vector<vec3> tangents;
  tangents.reserve(vertices.size());
  for (const auto& vertex : vertices) {
    tangents.emplace_back(vertex.tangent);
  }
  file.write(reinterpret_cast<const char*>(tangents.data()), sizeof(glm::vec3) * tangents.size());
}

template <typename IndexType>
void calc_tangents(std::span<Vertex> vertices, std::span<IndexType> indices) {
  ZoneScoped;
  SMikkTSpaceContext ctx{};
  SMikkTSpaceInterface interface{};
  ctx.m_pInterface = &interface;

  struct MyCtx {
    MyCtx(std::span<Vertex> vertices, std::span<IndexType>& indices)
        : vertices(vertices), indices(indices), num_faces(indices.size() / 3) {}
    std::span<Vertex> vertices;
    std::span<IndexType> indices;
    size_t num_faces{};
    int face_size = 3;
    Vertex& get_vertex(int face_idx, int vert_idx) {
      return vertices[indices[(face_idx * face_size) + vert_idx]];
    }
  };

  MyCtx my_ctx{vertices, indices};
  ctx.m_pUserData = &my_ctx;

  interface.m_getNumFaces = [](const SMikkTSpaceContext* ctx) -> int {
    return reinterpret_cast<MyCtx*>(ctx->m_pUserData)->num_faces;
  };
  // assuming GL_TRIANGLES until it becomes an issue
  interface.m_getNumVerticesOfFace = [](const SMikkTSpaceContext* ctx, const int) {
    return reinterpret_cast<MyCtx*>(ctx->m_pUserData)->face_size;
  };

  interface.m_getPosition = [](const SMikkTSpaceContext* ctx, float fvPosOut[], const int iFace,
                               const int iVert) {
    MyCtx& my_ctx = *reinterpret_cast<MyCtx*>(ctx->m_pUserData);
    Vertex& vertex = my_ctx.get_vertex(iFace, iVert);
    fvPosOut[0] = vertex.pos.x;
    fvPosOut[1] = vertex.pos.y;
    fvPosOut[2] = vertex.pos.z;
  };
  interface.m_getNormal = [](const SMikkTSpaceContext* ctx, float fvNormOut[], const int iFace,
                             const int iVert) {
    MyCtx& my_ctx = *reinterpret_cast<MyCtx*>(ctx->m_pUserData);
    Vertex& vertex = my_ctx.get_vertex(iFace, iVert);
    fvNormOut[0] = vertex.normal.x;
    fvNormOut[1] = vertex.normal.y;
    fvNormOut[2] = vertex.normal.z;
  };
  interface.m_getTexCoord = [](const SMikkTSpaceContext* ctx, float fvTexcOut[], const int iFace,
                               const int iVert) {
    MyCtx& my_ctx = *reinterpret_cast<MyCtx*>(ctx->m_pUserData);
    Vertex& vertex = my_ctx.get_vertex(iFace, iVert);
    fvTexcOut[0] = vertex.uv_x;
    fvTexcOut[1] = vertex.uv_y;
  };
  interface.m_setTSpaceBasic = [](const SMikkTSpaceContext* ctx, const float fvTangent[],
                                  const float, const int iFace, const int iVert) {
    MyCtx& my_ctx = *reinterpret_cast<MyCtx*>(ctx->m_pUserData);
    Vertex& vertex = my_ctx.get_vertex(iFace, iVert);
    vertex.tangent.x = fvTangent[0];
    vertex.tangent.y = fvTangent[1];
    vertex.tangent.z = fvTangent[2];
    // vertex.tangent.w = fSign;
  };
  genTangSpaceDefault(&ctx);
}
std::vector<uint8_t> read_file(const std::string& full_path) {
  std::ifstream file(full_path, std::ios::binary | std::ios::ate);
  if (!file) {
    throw std::runtime_error("Failed to open file: " + full_path);
  }

  std::streamsize size = file.tellg();
  file.seekg(0, std::ios::beg);

  std::vector<uint8_t> buffer(size);
  if (!file.read(reinterpret_cast<char*>(buffer.data()), size)) {
    throw std::runtime_error("Failed to read file: " + full_path);
  }

  return buffer;
}

// void decompose_matrix(const glm::mat4& m, glm::vec3& pos, glm::quat& rot, glm::vec3& scale) {
//   pos = m[3];
//   for (int i = 0; i < 3; i++) scale[i] = glm::length(glm::vec3(m[i]));
//   const glm::mat3 rot_mtx(glm::vec3(m[0]) / scale[0], glm::vec3(m[1]) / scale[1],
//                           glm::vec3(m[2]) / scale[2]);
//   rot = glm::quat_cast(rot_mtx);
// }
//
// VkFilter gltf_to_vk_filter(fastgltf::Filter filter) {
//   switch (filter) {
//     // nearest
//     case fastgltf::Filter::Nearest:
//     case fastgltf::Filter::NearestMipMapNearest:
//     case fastgltf::Filter::NearestMipMapLinear:
//       return VK_FILTER_NEAREST;
//
//       // linear
//     case fastgltf::Filter::Linear:
//     case fastgltf::Filter::LinearMipMapLinear:
//     case fastgltf::Filter::LinearMipMapNearest:
//     default:
//       return VK_FILTER_LINEAR;
//   }
// }
//
// VkSamplerMipmapMode gltf_to_vk_mipmap_mode(fastgltf::Filter filter) {
//   switch (filter) {
//     case fastgltf::Filter::NearestMipMapNearest:
//     case fastgltf::Filter::LinearMipMapNearest:
//       return VK_SAMPLER_MIPMAP_MODE_LINEAR;
//     case fastgltf::Filter::NearestMipMapLinear:
//     case fastgltf::Filter::LinearMipMapLinear:
//     default:
//       return VK_SAMPLER_MIPMAP_MODE_NEAREST;
//   }
// }
//
// VkSamplerAddressMode gltf_to_vk_wrap_mode(fastgltf::Wrap wrap) {
//   switch (wrap) {
//     case fastgltf::Wrap::ClampToEdge:
//       return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
//     case fastgltf::Wrap::MirroredRepeat:
//       return VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
//     default:
//       return VK_SAMPLER_ADDRESS_MODE_REPEAT;
//   }
// }

void set_node_transform_from_gltf_node(NodeData& new_node, const fastgltf::Node& gltf_node) {
  std::visit(fastgltf::visitor{[&new_node](fastgltf::TRS matrix) {
                                 glm::vec3 trans(matrix.translation[0], matrix.translation[1],
                                                 matrix.translation[2]);
                                 glm::quat rot(matrix.rotation[3], matrix.rotation[0],
                                               matrix.rotation[1], matrix.rotation[2]);
                                 glm::vec3 scale(matrix.scale[0], matrix.scale[1], matrix.scale[2]);
                                 new_node.local_transform = glm::translate(glm::mat4{1}, trans) *
                                                            glm::toMat4(rot) *
                                                            glm::scale(glm::mat4{1}, scale);
                               },
                               [&new_node](fastgltf::math::fmat4x4 matrix) {
                                 for (int row = 0; row < 4; row++) {
                                   for (int col = 0; col < 4; col++) {
                                     new_node.local_transform[row][col] = matrix[row][col];
                                   }
                                 }
                               }},
             gltf_node.transform);
}

void update_node_transforms(std::vector<NodeData>& nodes, std::vector<u32>& root_node_indices) {
  // node idx, parent idx
  std::vector<std::pair<u32, u32>> to_refresh;
  to_refresh.reserve(root_node_indices.size());
  for (size_t i = 0; i < nodes.size(); i++) {
    auto& node = nodes[i];
    to_refresh.clear();
    if (node.parent_idx == NodeData::null_idx) {
      root_node_indices.emplace_back(i);
      to_refresh.emplace_back(i, NodeData::null_idx);
    }

    while (!to_refresh.empty()) {
      auto [node_idx, parent_idx] = to_refresh.back();
      to_refresh.pop_back();
      auto& node = nodes[node_idx];
      node.world_transform = parent_idx == NodeData::null_idx
                                 ? node.local_transform
                                 : nodes[parent_idx].world_transform * node.local_transform;
      for (auto c : node.children_indices) {
        to_refresh.emplace_back(c, node_idx);
      }
    }
  }
}

void load_scene_graph_data(LoadedSceneBaseData& result, fastgltf::Asset& gltf,
                           u32 default_mat_idx) {
  ZoneScoped;
  auto& scene_graph_data = result.scene_graph_data;
  // aggregate nodes
  std::vector<u32> prim_offsets_of_meshes(gltf.meshes.size());
  {
    u32 offset = 0;
    for (u32 mesh_idx = 0; mesh_idx < gltf.meshes.size(); mesh_idx++) {
      prim_offsets_of_meshes[mesh_idx] = offset;
      offset += gltf.meshes[mesh_idx].primitives.size();
    }
  }
  result.scene_graph_data.node_datas.reserve(gltf.nodes.size());
  std::vector<u32> camera_node_indices;
  for (u32 node_idx = 0; node_idx < gltf.nodes.size(); node_idx++) {
    fastgltf::Node& gltf_node = gltf.nodes[node_idx];
    NodeData new_node;
    if (auto mesh_idx = gltf_node.meshIndex.value_or(NodeData::null_idx);
        mesh_idx != NodeData::null_idx) {
      u32 prim_idx = 0;
      for (const auto& primitive : gltf.meshes[mesh_idx].primitives) {
        new_node.meshes.emplace_back(NodeData::MeshData{
            .mesh_idx = prim_offsets_of_meshes[mesh_idx] + prim_idx++,
            .material_id = static_cast<u16>(primitive.materialIndex.value_or(default_mat_idx)),
        });
        auto& mat = result.materials[new_node.meshes.back().material_id];
        new_node.meshes.back().pass_flags = mat.get_pass_flags();
      }
      result.scene_graph_data.mesh_node_indices.emplace_back(node_idx);
    }
    if (auto cam_idx = gltf_node.cameraIndex.value_or(NodeData::null_idx);
        cam_idx != NodeData::null_idx) {
      camera_node_indices.emplace_back(cam_idx);
    }
    new_node.name = gltf_node.name;
    set_node_transform_from_gltf_node(new_node, gltf_node);
    result.scene_graph_data.node_datas.emplace_back(new_node);
  }

  // link children/parents
  for (u64 node_idx = 0; node_idx < gltf.nodes.size(); node_idx++) {
    auto& gltf_node = gltf.nodes[node_idx];
    auto& scene_node = result.scene_graph_data.node_datas[node_idx];
    scene_node.children_indices.reserve(gltf_node.children.size());
    for (auto child_idx : gltf_node.children) {
      scene_node.children_indices.emplace_back(child_idx);
      result.scene_graph_data.node_datas[child_idx].parent_idx = node_idx;
    }
  }

  // assign root nodes
  for (u64 node_idx = 0; node_idx < scene_graph_data.node_datas.size(); node_idx++) {
    auto& node = scene_graph_data.node_datas[node_idx];
    if (node.parent_idx == NodeData::null_idx) {
      scene_graph_data.root_node_indices.emplace_back(node_idx);
    }
  }
  update_node_transforms(scene_graph_data.node_datas, scene_graph_data.root_node_indices);
}

struct CpuImageData {
  enum class Type : u8 { None, KTX2, JPEG, PNG, DDS };
  u32 w, h, d, components;
  Format format{};
  Type type{};
  void* data{};  // points to type specific data (stb image, ktx_tex ptr, etc)
};

CpuImageData::Type convert_cpu_img_type(fastgltf::MimeType type) {
  switch (type) {
    case fastgltf::MimeType::KTX2:
      return CpuImageData::Type::KTX2;
    case fastgltf::MimeType::DDS:
      return CpuImageData::Type::DDS;
    case fastgltf::MimeType::JPEG:
      return CpuImageData::Type::JPEG;
    case fastgltf::MimeType::PNG:
      return CpuImageData::Type::PNG;
    default:
      return CpuImageData::Type::None;
  }
}

void load_image(CpuImageData& result, CpuImageData::Type type, const void* data, u64 size,
                bool srgb) {
  result.type = type;
  switch (type) {
    case CpuImageData::Type::KTX2: {
      ktxTexture2* ktx_tex{};
      if (auto result =
              ktxTexture2_CreateFromMemory(reinterpret_cast<const ktx_uint8_t*>(data), size,
                                           KTX_TEXTURE_CREATE_LOAD_IMAGE_DATA_BIT, &ktx_tex);
          result != KTX_SUCCESS) {
        assert(0);
      }
      assert(ktx_tex->pData && ktx_tex->dataSize);

      u32 components = ktxTexture2_GetNumComponents(ktx_tex);
      result.components = components;
      result.w = ktx_tex->baseWidth;
      result.h = ktx_tex->baseHeight;
      result.d = ktx_tex->baseDepth;
      result.data = ktx_tex;
      if (ktxTexture2_NeedsTranscoding(ktx_tex)) {
        ktx_transcode_fmt_e ktx_transcode_format{};
        if (components == 4 || components == 3) {
          ktx_transcode_format = KTX_TTF_BC7_RGBA;
          result.format = srgb ? Format::Bc7SrgbBlock : Format::Bc7UnormBlock;
        } else if (components == 2) {
          ktx_transcode_format = KTX_TTF_BC5_RG;
          result.format = Format::Bc5UnormBlock;
        } else if (components == 1) {
          ktx_transcode_format = KTX_TTF_BC4_R;
          result.format = Format::Bc4UnormBlock;
        }
        // determine target format
        if (auto result =
                ktxTexture2_TranscodeBasis(ktx_tex, ktx_transcode_format, KTX_TF_HIGH_QUALITY);
            result != KTX_SUCCESS) {
          assert(false);
        }
      } else {
        assert(0);
      }
      break;
    }
    case CpuImageData::Type::JPEG:
    case CpuImageData::Type::PNG: {
      int w, h, channels;
      auto* pixels = stbi_load_from_memory(reinterpret_cast<const unsigned char*>(data),
                                           static_cast<int>(size), &w, &h, &channels, 4);
      result.w = w;
      result.h = h;
      result.components = channels;
      result.data = pixels;
      if (result.components == 4 || result.components == 3) {
        result.format = srgb ? Format::R8G8B8A8Srgb : Format::R8G8B8A8Unorm;
      } else if (result.components == 2) {
        result.format = srgb ? Format::R8G8Srgb : Format::R8G8Unorm;
      } else if (result.components == 1) {
        result.format = srgb ? Format::R8Srgb : Format::R8Unorm;
      }
      break;
    }
    default:
      assert(0);
  }
}

void load_cpu_img_data(const fastgltf::Asset& asset, const fastgltf::Image& image,
                       const std::filesystem::path& directory, CpuImageData& result,
                       PBRImageUsage usage) {
  ZoneScoped;
  bool is_srgb_usage = usage == PBRImageUsage::BaseColor || usage == PBRImageUsage::Emissive;
  std::visit(fastgltf::visitor{
                 [&](const fastgltf::sources::Array& arr) {
                   load_image(result, convert_cpu_img_type(arr.mimeType), arr.bytes.data(),
                              arr.bytes.size_bytes(), is_srgb_usage);
                 },
                 [&](const fastgltf::sources::Vector& vector) {
                   load_image(result, convert_cpu_img_type(vector.mimeType), vector.bytes.data(),
                              vector.bytes.size() * sizeof(std::byte), is_srgb_usage);
                 },
                 [&](const fastgltf::sources::URI& file_path) {
                   assert(file_path.fileByteOffset == 0);
                   const std::string path(file_path.uri.path().begin(), file_path.uri.path().end());
                   auto full_path = directory / path;
                   if (!std::filesystem::exists(full_path)) {
                     LERROR("glTF Image load fail: path does not exist {}", full_path.string());
                   }
                   auto bytes = read_file(full_path);
                   const auto& ext = full_path.extension().string();
                   CpuImageData::Type type = ext == ".ktx2"   ? CpuImageData::Type::KTX2
                                             : ext == ".png"  ? CpuImageData::Type::PNG
                                             : ext == ".jpeg" ? CpuImageData::Type::JPEG
                                             : ext == ".jpg"  ? CpuImageData::Type::JPEG
                                             : ext == ".png"  ? CpuImageData::Type::PNG
                                             : ext == ".dds"  ? CpuImageData::Type::DDS
                                                              : CpuImageData::Type::None;
                   load_image(result, type, bytes.data(), bytes.size(), is_srgb_usage);
                 },
                 [&](const fastgltf::sources::BufferView& view) {
                   const auto& buffer_view = asset.bufferViews[view.bufferViewIndex];
                   const auto& buffer = asset.buffers[buffer_view.bufferIndex];
                   std::visit(fastgltf::visitor{
                                  [](auto&) {},
                                  [&](const fastgltf::sources::Array& arr) {
                                    load_image(result, convert_cpu_img_type(view.mimeType),
                                               arr.bytes.data() + buffer_view.byteOffset,
                                               buffer_view.byteLength, is_srgb_usage);
                                  },
                                  [&](const fastgltf::sources::Vector& vector) {
                                    load_image(result, convert_cpu_img_type(view.mimeType),
                                               vector.bytes.data() + buffer_view.byteOffset,
                                               buffer_view.byteLength, is_srgb_usage);
                                  },
                              },
                              buffer.data);
                 },
                 [](auto&) {
                   LERROR("not valid image path uh oh spaghettio");
                   assert(0);
                 }},
             image.data);
}

}  // namespace

std::optional<LoadedSceneBaseData> load_gltf_base(const std::filesystem::path& path,
                                                  const DefaultMaterialData& default_mat) {
  ZoneScoped;
  std::optional<LoadedSceneBaseData> result = std::nullopt;
  if (!std::filesystem::exists(path)) {
    LERROR("Failed to load glTF: directory {} does not exist", path.string());
    return result;
  }

  constexpr auto supported_extensions =
      fastgltf::Extensions::KHR_texture_basisu | fastgltf::Extensions::KHR_mesh_quantization |
      fastgltf::Extensions::KHR_texture_transform | fastgltf::Extensions::KHR_materials_variants |
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
    LERROR("Failed to load glTF\n\tpath: {}\n\terror: {}\n", path.string(),
           fastgltf::getErrorMessage(load_ret.error()));
    return result;
  }

  fastgltf::Asset gltf = std::move(load_ret.get());
  result = LoadedSceneBaseData{};

  std::vector<PBRImageUsage> img_usages(gltf.images.size(), PBRImageUsage::BaseColor);
  {
    ZoneScopedN("determine usages");
    auto set_usage = [&img_usages](const fastgltf::Texture& tex, PBRImageUsage usage) {
      std::size_t idx = tex.basisuImageIndex.value_or(tex.imageIndex.value_or(UINT32_MAX));
      if (idx != UINT32_MAX) {
        img_usages[idx] = usage;
      }
    };
    for (const auto& gltf_mat : gltf.materials) {
      if (gltf_mat.pbrData.baseColorTexture.has_value()) {
        set_usage(gltf.textures[gltf_mat.pbrData.baseColorTexture.value().textureIndex],
                  PBRImageUsage::BaseColor);
      }
      if (gltf_mat.pbrData.metallicRoughnessTexture.has_value()) {
        set_usage(gltf.textures[gltf_mat.pbrData.metallicRoughnessTexture.value().textureIndex],
                  PBRImageUsage::MetallicRoughness);
      }
      if (gltf_mat.emissiveTexture.has_value()) {
        set_usage(gltf.textures[gltf_mat.emissiveTexture.value().textureIndex],
                  PBRImageUsage::Emissive);
      }
      if (gltf_mat.normalTexture.has_value()) {
        set_usage(gltf.textures[gltf_mat.normalTexture.value().textureIndex],
                  PBRImageUsage::Normal);
      }
      if (gltf_mat.occlusionTexture.has_value()) {
        set_usage(gltf.textures[gltf_mat.occlusionTexture.value().textureIndex],
                  PBRImageUsage::Occlusion);
      }
      if (gltf_mat.packedOcclusionRoughnessMetallicTextures) {
        if (gltf_mat.packedOcclusionRoughnessMetallicTextures->occlusionRoughnessMetallicTexture
                .has_value()) {
          set_usage(gltf.textures[gltf_mat.packedOcclusionRoughnessMetallicTextures
                                      ->occlusionRoughnessMetallicTexture.value()
                                      .textureIndex],
                    PBRImageUsage::OccRoughnessMetallic);
        }
      }
    }
  }
  std::vector<CpuImageData> images(gltf.images.size());
  std::vector<std::future<void>> futures(images.size());
  {
    ZoneScopedN("load images");
    for (u64 i = 0; i < images.size(); i++) {
      futures[i] = threads::pool.submit_task([i, &gltf, &parent_path, &images, &img_usages]() {
        load_cpu_img_data(gltf, gltf.images[i], parent_path, images[i], img_usages[i]);
        // upload to gpu
      });
    }
    for (auto& f : futures) {
      f.get();
    }
    futures.clear();
  }

  struct ImgUploadInfo {
    uvec3 extent{};
    size_t size;
    void* data;
    size_t staging_offset;
    u32 level;
    u32 img_idx;
  };
  std::vector<ImgUploadInfo> img_upload_infos;
  img_upload_infos.reserve(images.size());
  result->textures.reserve(images.size());
  size_t staging_offset{};
  for (auto& img : images) {
    // TODO: diff types, dds
    if (img.type == CpuImageData::Type::KTX2) {
      auto* ktx = (ktxTexture2*)img.data;
      assert(ktx->numLevels > 0);
      u64 tot = 0;
      for (u32 level = 0; level < ktx->numLevels; level++) {
        size_t level_offset;
        ktxTexture_GetImageOffset(ktxTexture(ktx), level, 0, 0, &level_offset);
        u32 w = std::max(img.w >> level, 1u);
        u32 h = std::max(img.h >> level, 1u);
        size_t size = img_to_buffer_size(img.format, {w, h, 1});
        tot += size;
        img_upload_infos.emplace_back(
            ImgUploadInfo{.extent = {w, h, 1},
                          .size = size,
                          .data = ktx->pData + level_offset,
                          .staging_offset = staging_offset,
                          .level = level,
                          .img_idx = static_cast<u32>(result->textures.size())});
        staging_offset += size;
      }
      result->textures.emplace_back(
          get_device().create_image(ImageDesc{.type = ImageDesc::Type::TwoD,
                                              .format = img.format,
                                              .dims = {img.w, img.h, 1},
                                              .mip_levels = ktx->numLevels,
                                              .bind_flags = BindFlag::ShaderResource}));
      assert(tot == ktx->dataSize);
      (void)tot;

    } else {
      // TODO: mip gen?
      size_t size = img_to_buffer_size(img.format, {img.w, img.h, 1});
      img_upload_infos.emplace_back(
          ImgUploadInfo{.extent = {img.w, img.w, 1},
                        .size = size,
                        .data = img.data,
                        .staging_offset = staging_offset,
                        .level = 0,
                        .img_idx = static_cast<u32>(result->textures.size())});
      result->textures.emplace_back(
          get_device().create_image(ImageDesc{.type = ImageDesc::Type::TwoD,
                                              .format = img.format,
                                              .dims = {img.w, img.h, 1},
                                              .mip_levels = 1,
                                              .bind_flags = BindFlag::ShaderResource}));
      staging_offset += size;
    }
  }

  assert(result->textures.size() == images.size());
  {
    ZoneScopedN("upload images");
    constexpr i32 max_batch_upload_size = 1024ull * 1024 * 1024;  // 1 GB
    i32 batch_upload_size = std::min<i32>(max_batch_upload_size, staging_offset);
    assert(batch_upload_size < max_batch_upload_size);
    i32 bytes_remaining = staging_offset;
    u64 img_i{};
    u64 curr_staging_offset = 0;
    u64 start_copy_idx{};
    auto copy_cmd = get_device().transfer_copy_allocator_.allocate(batch_upload_size);
    StateTracker state;
    auto flush_uploads = [&]() {
      for (u64 i = 0; i < img_i; i++) {
        if (futures[i].valid()) {
          futures[i].get();
        } else {
          break;
        }
      }
      futures.clear();
      auto end_copy_idx = img_i - 1;
      {
        state.reset(copy_cmd.transfer_cmd_buf);
        for (u64 i = start_copy_idx; i <= end_copy_idx; i++) {
          const auto& img_upload = img_upload_infos[i];
          state.transition(get_device().get_image(result->textures[img_upload.img_idx])->image(),
                           VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        }
        state.flush_barriers();
        for (u64 i = start_copy_idx; i <= end_copy_idx; i++) {
          const auto& img_upload = img_upload_infos[i];
          const auto& texture = *get_device().get_image(result->textures[img_upload.img_idx]);
          VkBufferImageCopy2 img_copy{
              .sType = VK_STRUCTURE_TYPE_BUFFER_IMAGE_COPY_2,
              .bufferOffset = img_upload.staging_offset - curr_staging_offset,
              // .bufferRowLength = (img_upload.extent.x + 3) & ~3,  // Align to BC7
              // 4x4 blocks .bufferImageHeight = (img_upload.extent.y + 3) & ~3,
              .bufferRowLength = 0,
              .bufferImageHeight = 0,
              .imageSubresource =
                  {
                      .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                      .mipLevel = img_upload.level,
                      .layerCount = 1,
                  },
              .imageExtent = VkExtent3D{img_upload.extent.x, img_upload.extent.y, 1}};
          VkCopyBufferToImageInfo2 img_copy_info{
              .sType = VK_STRUCTURE_TYPE_COPY_BUFFER_TO_IMAGE_INFO_2_KHR,
              .srcBuffer = get_device().get_buffer(copy_cmd.staging_buffer)->buffer(),
              .dstImage = texture.image(),
              .dstImageLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
              .regionCount = 1,
              .pRegions = &img_copy,
          };
          vkCmdCopyBufferToImage2KHR(copy_cmd.transfer_cmd_buf, &img_copy_info);
        }
        for (u64 i = start_copy_idx; i <= end_copy_idx; i++) {
          state.transition(
              get_device().get_image(result->textures[img_upload_infos[i].img_idx])->image(),
              VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
              VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
              VK_IMAGE_LAYOUT_READ_ONLY_OPTIMAL);
        }
        state.flush_barriers();
        // TODO: thread safe
        get_device().transfer_copy_allocator_.submit(copy_cmd);
      }

      curr_staging_offset += max_batch_upload_size;
      start_copy_idx = img_i;
    };
    while (bytes_remaining > 0) {
      // if (bytes_remaining - img_upload_infos[img_i].size < 0) {
      //   LINFO("flush uploads");
      //   flush_uploads();
      // }

      futures.emplace_back(
          threads::pool.submit_task([img_i, &img_upload_infos, curr_staging_offset, &copy_cmd]() {
            const auto& img_upload = img_upload_infos[img_i];
            if (get_device().get_buffer(copy_cmd.staging_buffer)->size() <
                img_upload.staging_offset + img_upload.size) {
              assert(0);
            } else {
              memcpy((char*)get_device().get_buffer(copy_cmd.staging_buffer)->mapped_data() +
                         img_upload.staging_offset - curr_staging_offset,
                     img_upload.data, img_upload.size);
            }
          }));
      bytes_remaining -= img_upload_infos[img_i].size;
      img_i++;
    }

    flush_uploads();
    // TODO: remove?
    for (auto& f : futures) {
      if (f.valid()) f.get();
    }
    futures.clear();
  }
  {
    ZoneScopedN("free imgs");
    for (auto& image : images) {
      futures.push_back(threads::pool.submit_task([&image]() {
        switch (image.type) {
          case CpuImageData::Type::KTX2:
            ktxTexture_Destroy(ktxTexture(image.data));
            break;
          case CpuImageData::Type::JPEG:
          case CpuImageData::Type::PNG:
            stbi_image_free(image.data);
            break;
          default:
            assert(0);
        }
      }));
    }
    for (u64 i = 0; i < images.size(); i++) {
      futures[i].get();
    }
    futures.clear();
  }
  for (auto& f : futures) {
    if (f.valid()) {
      f.get();
    }
  }
  futures.clear();

  {
    ZoneScopedN("load gltf materials");
    result->materials.reserve(gltf.materials.size());
    for (size_t i = 0; i < gltf.materials.size(); i++) {
      const auto& gltf_mat = gltf.materials[i];
      auto get_idx = [&gltf, &default_mat, &result](const fastgltf::TextureInfo& info) -> u32 {
        const auto& tex = gltf.textures[info.textureIndex];
        auto gltf_idx = tex.basisuImageIndex.value_or(tex.imageIndex.value_or(UINT32_MAX));
        if (gltf_idx != UINT32_MAX) {
          return get_device().get_bindless_idx(result->textures[gltf_idx].handle,
                                               SubresourceType::Shader);
        }
        LERROR("uh oh, no texture for gltf material");
        return default_mat.white_img_handle;
      };
      Material mat{.ids1 = uvec4{0}, .ids2 = uvec4(0)};
      auto base_col = gltf_mat.pbrData.baseColorFactor;
      mat.albedo_factors = {base_col.x(), base_col.y(), base_col.z(), base_col.w()};
      mat.pbr_factors.x = gltf_mat.pbrData.metallicFactor;
      mat.pbr_factors.y = gltf_mat.pbrData.roughnessFactor;

      if (gltf_mat.doubleSided) {
        mat.ids2.w |= MATERIAL_DOUBLE_SIDED_BIT;
      }

      if (gltf_mat.pbrData.baseColorTexture.has_value()) {
        mat.ids1.x = get_idx(gltf_mat.pbrData.baseColorTexture.value());
      }
      if (gltf_mat.normalTexture.has_value()) {
        mat.ids1.y = get_idx(gltf_mat.normalTexture.value());
      }
      if (gltf_mat.pbrData.metallicRoughnessTexture.has_value()) {
        mat.ids1.z = get_idx(gltf_mat.pbrData.metallicRoughnessTexture.value());
      }
      if (gltf_mat.emissiveTexture.has_value()) {
        mat.ids1.w = get_idx(gltf_mat.emissiveTexture.value());
      }
      if (gltf_mat.occlusionTexture.has_value()) {
        mat.ids2.x = get_idx(gltf_mat.occlusionTexture.value());
      } else if (gltf_mat.packedOcclusionRoughnessMetallicTextures &&
                 gltf_mat.packedOcclusionRoughnessMetallicTextures
                     ->occlusionRoughnessMetallicTexture.has_value()) {
        mat.ids2.x = get_idx(gltf_mat.packedOcclusionRoughnessMetallicTextures
                                 ->occlusionRoughnessMetallicTexture.value());
        mat.ids2.w |= PACKED_OCCLUSION_ROUGHNESS_METALLIC;
      }
      if (gltf_mat.alphaMode == fastgltf::AlphaMode::Mask) {
        mat.ids2.w |= MATERIAL_ALPHA_MODE_MASK_BIT;
      } else if (gltf_mat.alphaMode == fastgltf::AlphaMode::Blend) {
        mat.ids2.w |= MATERIAL_TRANSPARENT_BIT;
      }

      mat.emissive_factors = vec4(gltf_mat.emissiveFactor.x(), gltf_mat.emissiveFactor.y(),
                                  gltf_mat.emissiveFactor.z(), 0);
      mat.emissive_factors *= gltf_mat.emissiveStrength;

      result->materials.emplace_back(mat);
    }
  }
  {
    ZoneScopedN("gltf load geometry");
    size_t total_num_gltf_primitives = 0;
    for (const auto& m : gltf.meshes) {
      total_num_gltf_primitives += m.primitives.size();
    }
    result->mesh_draw_infos.resize(total_num_gltf_primitives);
    futures.resize(total_num_gltf_primitives);

    u32 num_indices{};
    u32 num_vertices{};
    {
      u32 primitive_idx{0};
      u32 mesh_idx{0};
      u32 index_offset{};
      u32 vertex_offset{};
      for (const auto& gltf_mesh : gltf.meshes) {
        for (const auto& gltf_prim : gltf_mesh.primitives) {
          u32 first_index = index_offset;
          const auto& index_accessor = gltf.accessors[gltf_prim.indicesAccessor.value()];
          u32 index_count = index_accessor.count;
          index_offset += index_count;
          u32 first_vertex = vertex_offset;
          const auto* pos_attrib = gltf_prim.findAttribute("POSITION");
          if (pos_attrib == gltf_prim.attributes.end()) {
            return {};
          }
          u32 vertex_count = gltf.accessors[pos_attrib->accessorIndex].count;
          vertex_offset += vertex_count;
          num_vertices += vertex_count;
          num_indices += index_count;
          result->mesh_draw_infos[primitive_idx++] = {.first_index = first_index,
                                                      .index_count = index_count,
                                                      .first_vertex = first_vertex,
                                                      .vertex_count = vertex_count,
                                                      .mesh_idx = mesh_idx};
        }
        mesh_idx++;
      };
    }

    result->indices.resize(num_indices);
    result->vertices.resize(num_vertices);

    bool has_tangents = gltf.meshes[0].primitives[0].findAttribute("TANGENT") !=
                        gltf.meshes[0].primitives[0].attributes.end();
    std::filesystem::path tangents_path =
        parent_path / (path.filename().stem().string() + "_tangents.bin");
    bool loaded_tangents_from_disk = false;
    if (!has_tangents) {
      // load from disk
      if (std::filesystem::exists(tangents_path)) {
        load_tangents(tangents_path, result->vertices);
        loaded_tangents_from_disk = true;
      }
    }
    {
      futures.clear();
      u64 mesh_draw_idx = 0;
      for (u64 mesh_idx = 0; mesh_idx < gltf.meshes.size(); mesh_idx++) {
        for (u64 primitive_idx = 0; primitive_idx < gltf.meshes[mesh_idx].primitives.size();
             primitive_idx++, mesh_draw_idx++) {
          futures.emplace_back(threads::pool.submit_task([&gltf, mesh_idx, primitive_idx, &result,
                                                          mesh_draw_idx,
                                                          loaded_tangents_from_disk]() {
            ZoneScopedN("gltf process primitives");
            const auto& primitive = gltf.meshes[mesh_idx].primitives[primitive_idx];
            const auto& index_accessor = gltf.accessors[primitive.indicesAccessor.value()];
            auto& mesh_draw_info = result->mesh_draw_infos[mesh_draw_idx];
            u32 start_idx = mesh_draw_info.first_index;
            fastgltf::iterateAccessorWithIndex<u32>(
                gltf, index_accessor,
                [&](uint32_t index, u32 i) { result->indices[start_idx + i] = index; });
            const auto* pos_attrib = primitive.findAttribute("POSITION");
            if (pos_attrib == primitive.attributes.end()) {
              assert(0);
            }
            u64 start_i = mesh_draw_info.first_vertex;
            const auto& pos_accessor = gltf.accessors[pos_attrib->accessorIndex];
            fastgltf::iterateAccessorWithIndex<vec3>(
                gltf, pos_accessor,
                [&result, start_i](vec3 pos, u32 i) { result->vertices[start_i + i].pos = pos; });

            glm::vec3 min;
            glm::vec3 max;

            bool has_min = false, has_max = false;
            if (pos_accessor.min.has_value()) {
              has_min = true;
              const auto& val = pos_accessor.min.value();
              min = glm::vec3(val.get<double>(0), val.get<double>(1), val.get<double>(2));
            }
            if (pos_accessor.max.has_value()) {
              has_max = true;
              const auto& val = pos_accessor.max.value();
              max = glm::vec3(val.get<double>(0), val.get<double>(1), val.get<double>(2));
            }

            // If min/max are available, calculate bounds directly
            if (has_min && has_max) {
              mesh_draw_info.aabb = {.min = min, .max = max};
            } else {
              assert(0 && "why does this gltf not have bounds lmao noob");
              // calculate bounds from vertices if accessor min/max not set
              calc_aabb(mesh_draw_info.aabb, &result->vertices[mesh_draw_info.first_vertex],
                        pos_accessor.count, sizeof(gfx::Vertex), offsetof(gfx::Vertex, pos));
            }

            const auto* uv_attrib = primitive.findAttribute("TEXCOORD_0");
            if (uv_attrib != primitive.attributes.end()) {
              const auto& accessor = gltf.accessors[uv_attrib->accessorIndex];

              assert(accessor.count == pos_accessor.count);
              u64 i = start_i;
              auto range = fastgltf::iterateAccessor<glm::vec2>(gltf, accessor);
              for (const glm::vec2& uv : range) {
                result->vertices[i].uv_x = uv.x;
                result->vertices[i++].uv_y = uv.y;
              }
            }
            const auto* tangent_attrib = primitive.findAttribute("TANGENT");
            if (tangent_attrib != primitive.attributes.end()) {
              const auto& accessor = gltf.accessors[tangent_attrib->accessorIndex];
              u64 i = start_i;
              assert(accessor.count == pos_accessor.count);
              if (accessor.type == fastgltf::AccessorType::Vec3) {
                auto range = fastgltf::iterateAccessor<glm::vec3>(gltf, accessor);
                for (const glm::vec3& tangent : range) {
                  result->vertices[i++].tangent = vec4(tangent, 0.);
                }
              } else if (accessor.type == fastgltf::AccessorType::Vec4) {
                auto range = fastgltf::iterateAccessor<glm::vec4>(gltf, accessor);
                for (const glm::vec4& tangent : range) {
                  result->vertices[i++].tangent = tangent;
                }
              }
            } else if (!loaded_tangents_from_disk) {
              calc_tangents<u32>(
                  std::span(result->vertices.data() + (start_i), mesh_draw_info.vertex_count),
                  std::span(result->indices.data() + (start_idx), mesh_draw_info.index_count));
            }

            const auto* normal_attrib = primitive.findAttribute("NORMAL");
            if (normal_attrib != primitive.attributes.end()) {
              const auto& normal_accessor = gltf.accessors[normal_attrib->accessorIndex];
              u64 i = start_i;
              assert(normal_accessor.count == pos_accessor.count);
              auto range = fastgltf::iterateAccessor<glm::vec3>(gltf, normal_accessor);
              for (const glm::vec3& normal : range) {
                result->vertices[i++].normal = normal;
              }
            }
          }));
        }
      }
      for (auto& f : futures) {
        if (f.valid()) {
          f.get();
        }
      }
      if (!has_tangents && !loaded_tangents_from_disk) {
        save_tangents(tangents_path, result->vertices);
      }
    }
  }

  load_scene_graph_data(*result, gltf, 0);

  return result;
}

std::optional<LoadedSceneData> load_gltf(const std::filesystem::path& path,
                                         const DefaultMaterialData& default_mat) {
  ZoneScoped;
  PrintTimerMS t;
  auto base_scene_data_ret = load_gltf_base(path, default_mat);
  if (!base_scene_data_ret.has_value()) {
    return {};
  }

  LoadedSceneBaseData& base_scene_data = base_scene_data_ret.value();
  return LoadedSceneData{
      .scene_graph_data = std::move(base_scene_data.scene_graph_data),
      .materials = std::move(base_scene_data.materials),
      .textures = std::move(base_scene_data.textures),
      .mesh_draw_infos = std::move(base_scene_data.mesh_draw_infos),
      .vertices = std::move(base_scene_data.vertices),
      .indices = std::move(base_scene_data.indices),
  };
}

namespace loader {

std::optional<CPUHDRImageData> load_hdr(const std::filesystem::path& path, int num_components,
                                        bool flip) {
  std::optional<CPUHDRImageData> res;
  if (!std::filesystem::exists(path)) {
    LINFO("path does not exist: {}", path.string());
    return res;
  }
  res = CPUHDRImageData{};
  int w, h, channels;

  stbi_set_flip_vertically_on_load(flip);
  res->data = stbi_loadf(path.c_str(), &w, &h, &channels, num_components);
  assert(res->data);
  res->w = w;
  res->h = h;
  res->channels = channels;
  return res;
}

void free_hdr(CPUHDRImageData& img_data) {
  assert(img_data.data);
  if (img_data.data) {
    stbi_image_free(img_data.data);
  }
  img_data.data = nullptr;
}

}  // namespace loader

PassFlags Material::get_pass_flags() const {
  PassFlags flags{};
  if (ids2.w & MATERIAL_ALPHA_MODE_MASK_BIT) {
    flags |= PassFlags_OpaqueAlpha;
  } else if (ids2.w & MATERIAL_TRANSPARENT_BIT) {
    flags |= PassFlags_Transparent;
  } else {
    flags |= PassFlags_Opaque;
  }
  return flags;
}

bool Material::is_double_sided() const { return (ids2.w & MATERIAL_DOUBLE_SIDED_BIT); }

}  // namespace gfx
