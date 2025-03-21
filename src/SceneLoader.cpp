#include "SceneLoader.hpp"

#include <ktx.h>
#include <ktxvulkan.h>
#include <volk.h>
#include <vulkan/vulkan_core.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fastgltf/core.hpp>
#include <fastgltf/glm_element_traits.hpp>
#include <fastgltf/tools.hpp>
#include <fastgltf/types.hpp>
#include <future>

#include "Scene.hpp"
#include "StateTracker.hpp"
#include "ThreadPool.hpp"
#include "VkRender2.hpp"
#include "vk2/StagingBufferPool.hpp"
#include "vk2/VkCommon.hpp"

// #include "ThreadPool.hpp"

#include "stb_image.h"
#include "vk2/Texture.hpp"

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/quaternion.hpp>
#include <tracy/Tracy.hpp>

#include "Logger.hpp"
#include "vk2/Buffer.hpp"

// ktx loading inspired/ripped from:
// https://github.com/JuanDiegoMontoya/Frogfood/blob/be82e484baab02b7ce3e80d36eb7c9291d97ebcb/src/Fvog/detail/ApiToEnum2.cpp#L4
namespace gfx {

namespace {

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

u32 populate_indices(const fastgltf::Primitive& primitive, const fastgltf::Asset& gltf,
                     std::vector<uint32_t>& indices) {
  if (!primitive.indicesAccessor.has_value()) {
    return 0;
  }
  u64 first_idx = indices.size();
  const auto& index_accessor = gltf.accessors[primitive.indicesAccessor.value()];
  indices.resize(indices.size() + index_accessor.count);
  size_t i = first_idx;
  fastgltf::iterateAccessor<u32>(gltf, index_accessor,
                                 [&](uint32_t index) { indices[i++] = index; });
  return index_accessor.count;
}

u32 populate_vertices(const fastgltf::Primitive& primitive, const fastgltf::Asset& gltf,
                      std::vector<Vertex>& vertices) {
  const auto* pos_attrib = primitive.findAttribute("POSITION");
  if (pos_attrib == primitive.attributes.end()) {
    assert(0);
    return 0;
  }
  u64 start_i = vertices.size();
  const auto& pos_accessor = gltf.accessors[pos_attrib->accessorIndex];
  vertices.reserve(vertices.size() + pos_accessor.count);
  for (glm::vec3 pos : fastgltf::iterateAccessor<glm::vec3>(gltf, pos_accessor)) {
    vertices.emplace_back(Vertex{.pos = pos});
  }

  const auto* uv_attrib = primitive.findAttribute("TEXCOORD_0");
  if (uv_attrib != primitive.attributes.end()) {
    const auto& accessor = gltf.accessors[uv_attrib->accessorIndex];

    assert(accessor.count == pos_accessor.count);
    u64 i = start_i;
    for (glm::vec2 uv : fastgltf::iterateAccessor<glm::vec2>(gltf, accessor)) {
      vertices[i].uv_x = uv.x;
      vertices[i++].uv_y = uv.y;
    }
  }

  const auto* normal_attrib = primitive.findAttribute("NORMAL");
  if (normal_attrib != primitive.attributes.end()) {
    const auto& normal_accessor = gltf.accessors[normal_attrib->accessorIndex];
    u64 i = start_i;
    assert(normal_accessor.count == pos_accessor.count);
    for (glm::vec3 normal : fastgltf::iterateAccessor<glm::vec3>(gltf, normal_accessor)) {
      vertices[i++].normal = normal;
    }
  }
  return pos_accessor.count;
}

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
  struct RefreshEntry {
    u32 node_idx{NodeData::null_idx};
    u32 parent_idx{NodeData::null_idx};
  };
  // TODO: don't allocate here
  std::vector<RefreshEntry> to_refresh;
  to_refresh.reserve(nodes.size());
  for (u64 node_idx = 0; node_idx < root_node_indices.size(); node_idx++) {
    auto& node = nodes[node_idx];
    if (node.parent_idx == NodeData::null_idx) {
      to_refresh.emplace_back(node_idx, NodeData::null_idx);
    }
  }

  while (!to_refresh.empty()) {
    RefreshEntry entry = to_refresh.back();
    to_refresh.pop_back();
    mat4 parent_world_transform =
        entry.parent_idx == NodeData::null_idx ? mat4{1} : nodes[entry.parent_idx].world_transform;
    auto& node = nodes[entry.node_idx];
    node.world_transform = parent_world_transform * node.local_transform;
    for (auto c : node.children_indices) {
      to_refresh.emplace_back(c, entry.node_idx);
    }
  }
}

void load_scene_graph_data(SceneLoadData& result, fastgltf::Asset& gltf) {
  ZoneScoped;
  // aggregate nodes
  std::vector<u32> prim_offsets_of_meshes(gltf.meshes.size());
  {
    u32 offset = 0;
    for (u32 mesh_idx = 0; mesh_idx < gltf.meshes.size(); mesh_idx++) {
      prim_offsets_of_meshes[mesh_idx] = offset;
      offset += gltf.meshes[mesh_idx].primitives.size();
    }
  }
  result.node_datas.reserve(gltf.nodes.size());
  for (u32 node_idx = 0; node_idx < gltf.nodes.size(); node_idx++) {
    fastgltf::Node& gltf_node = gltf.nodes[node_idx];
    NodeData new_node;
    if (auto mesh_idx = gltf_node.meshIndex.value_or(NodeData::null_idx);
        mesh_idx != NodeData::null_idx) {
      u32 prim_idx = 0;
      for (const auto& primitive : gltf.meshes[mesh_idx].primitives) {
        new_node.meshes.emplace_back(
            NodeData::MeshData{.material_id = static_cast<u32>(primitive.materialIndex.value_or(0)),
                               .mesh_idx = prim_offsets_of_meshes[mesh_idx] + prim_idx++});
      }
    }
    if (gltf_node.meshIndex.has_value()) {
      result.mesh_node_indices.emplace_back(node_idx);
    }
    new_node.name = gltf_node.name;
    set_node_transform_from_gltf_node(new_node, gltf_node);
    result.node_datas.emplace_back(new_node);
  }

  // link children/parents
  for (u64 node_idx = 0; node_idx < gltf.nodes.size(); node_idx++) {
    auto& gltf_node = gltf.nodes[node_idx];
    auto& scene_node = result.node_datas[node_idx];
    scene_node.children_indices.reserve(gltf_node.children.size());
    for (auto child_idx : gltf_node.children) {
      scene_node.children_indices.emplace_back(child_idx);
      result.node_datas[child_idx].parent_idx = node_idx;
    }
  }

  // assign root nodes
  for (u64 node_idx = 0; node_idx < result.node_datas.size(); node_idx++) {
    auto& node = result.node_datas[node_idx];
    if (node.parent_idx == NodeData::null_idx) {
      result.root_node_indices.emplace_back(node_idx);
    }
  }
  update_node_transforms(result.node_datas, result.root_node_indices);
}

// void load_samplers(const fastgltf::Asset& gltf, std::vector<vk2::Sampler>& samplers) {
//   ZoneScoped;
//   samplers.reserve(gltf.samplers.size());
//   for (const auto& sampler : gltf.samplers) {
//     samplers.emplace_back(VkSamplerCreateInfo{
//         .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
//         .magFilter = gltf_to_vk_filter(sampler.magFilter.value_or(fastgltf::Filter::Linear)),
//         .minFilter = gltf_to_vk_filter(sampler.magFilter.value_or(fastgltf::Filter::Linear)),
//         .mipmapMode = gltf_to_vk_mipmap_mode(
//             sampler.minFilter.value_or(fastgltf::Filter::LinearMipMapLinear)),
//         .addressModeU = gltf_to_vk_wrap_mode(sampler.wrapS),
//         .addressModeV = gltf_to_vk_wrap_mode(sampler.wrapT),
//         .addressModeW = gltf_to_vk_wrap_mode(fastgltf::Wrap::Repeat),
//         .minLod = -1000,
//         .maxLod = 100,
//         .borderColor = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK,
//     });
//   }
// }

struct CpuImageData {
  u32 w, h, channels;
  bool is_ktx{};
  VkFormat format;
  std::unique_ptr<unsigned char[], decltype([](unsigned char* p) { stbi_image_free(p); })>
      non_ktx_data;
  std::unique_ptr<ktxTexture2, decltype([](ktxTexture2* p) { ktxTexture_Destroy(ktxTexture(p)); })>
      ktx_data;
};

enum class ImageUsage : u8 {
  BaseColor,
  Normal,
  MetallicRoughness,
  OccRoughnessMetallic,
  Emissive,
  Occlusion
};

void load_cpu_img_data(const fastgltf::Asset& asset, const fastgltf::Image& image,
                       const std::filesystem::path&, CpuImageData& result, ImageUsage usage) {
  auto get_vk_format = [&](bool is_ktx) {
    switch (usage) {
      case ImageUsage::BaseColor:
      case ImageUsage::Emissive:
        return is_ktx ? VK_FORMAT_BC7_SRGB_BLOCK : VK_FORMAT_R8G8B8A8_SRGB;
      case ImageUsage::MetallicRoughness:
      case ImageUsage::Occlusion:
      case ImageUsage::Normal:
      case ImageUsage::OccRoughnessMetallic:
        return is_ktx ? VK_FORMAT_BC7_UNORM_BLOCK : VK_FORMAT_R8G8B8A8_UNORM;
    }
  };
  auto load_non_ktx = [&](const void* data, u64 size) {
    int w, h, channels;
    auto* pixels = stbi_load_from_memory(reinterpret_cast<const unsigned char*>(data),
                                         static_cast<int>(size), &w, &h, &channels, 4);
    result.w = w;
    result.h = h;
    result.channels = channels;
    result.non_ktx_data.reset(pixels);
    result.format = get_vk_format(false);
  };
  auto load_ktx = [&](const void* data, u64 size) {
    ktxTexture2* ktx_tex{};
    if (auto result =
            ktxTexture2_CreateFromMemory(reinterpret_cast<const ktx_uint8_t*>(data), size,
                                         KTX_TEXTURE_CREATE_LOAD_IMAGE_DATA_BIT, &ktx_tex);
        result != KTX_SUCCESS) {
      assert(0);
    }
    result.ktx_data.reset(ktx_tex);
    result.w = ktx_tex->baseWidth;
    result.h = ktx_tex->baseHeight;
    result.channels = ktxTexture2_GetNumComponents(ktx_tex);
    result.format = get_vk_format(true);
    if (ktxTexture2_NeedsTranscoding(ktx_tex)) {
      ZoneScopedN("Transcode KTX 2 Texture");
      ktx_transcode_fmt_e ktx_transcode_format{};
      switch (usage) {
        case ImageUsage::BaseColor:
        case ImageUsage::Emissive:
        case ImageUsage::MetallicRoughness:
        case ImageUsage::Occlusion:
        case ImageUsage::Normal:
        case ImageUsage::OccRoughnessMetallic:
          ktx_transcode_format = KTX_TTF_BC7_RGBA;
          break;
      }
      if (auto result =
              ktxTexture2_TranscodeBasis(ktx_tex, ktx_transcode_format, KTX_TF_HIGH_QUALITY);
          result != KTX_SUCCESS) {
        assert(false);
      }
    } else {
      result.format = static_cast<VkFormat>(ktx_tex->vkFormat);
    }
  };

  std::visit(
      fastgltf::visitor{
          [&](const fastgltf::sources::Array& arr) {
            result.is_ktx = arr.mimeType == fastgltf::MimeType::KTX2;
            if (result.is_ktx) {
              load_ktx(arr.bytes.data(), arr.bytes.size() * sizeof(std::byte));
            } else {
              load_non_ktx(arr.bytes.data(), arr.bytes.size_bytes());
            }
          },
          [&](const fastgltf::sources::Vector& vector) {
            result.is_ktx = vector.mimeType == fastgltf::MimeType::KTX2;
            if (result.is_ktx) {
              load_ktx(vector.bytes.data(), vector.bytes.size() * sizeof(std::byte));
            } else {
              load_non_ktx(vector.bytes.data(), vector.bytes.size() * sizeof(std::byte));
            }
          },
          [&](const fastgltf::sources::URI&) {
            assert(0);
            // assert(file_path.fileByteOffset == 0);
            // const std::string path(file_path.uri.path().begin(),
            // file_path.uri.path().end()); auto full_path = directory / path;
            // if
            // (!std::filesystem::exists(full_path)) {
            //   LERROR("glTF Image load fail: path does not exist {}",
            //   full_path.string());
            // }
            // if (file_path.mimeType == fastgltf::MimeType::KTX2) {
            //   result.is_ktx = true;
            //   LERROR("URI ktx unhandled");
            // } else if (file_path.mimeType == fastgltf::MimeType::PNG ||
            //            file_path.mimeType == fastgltf::MimeType::JPEG) {
            //   data = stbi_load(full_path.string().c_str(), &w, &h,
            //   &channels, 4);
            // } else {
            //   LERROR("unhandled uri type for loading image");
            // }
          },
          [&](const fastgltf::sources::BufferView& view) {
            result.is_ktx = view.mimeType == fastgltf::MimeType::KTX2;
            const auto& buffer_view = asset.bufferViews[view.bufferViewIndex];
            const auto& buffer = asset.buffers[buffer_view.bufferIndex];
            std::visit(
                fastgltf::visitor{
                    [](auto&) {},
                    [&](const fastgltf::sources::Array& arr) {
                      if (result.is_ktx) {
                        load_ktx(arr.bytes.data(), arr.bytes.size() * sizeof(std::byte));
                      } else {
                        load_non_ktx(arr.bytes.data(), arr.bytes.size_bytes());
                      }
                    },
                    [&](const fastgltf::sources::Vector& vector) {
                      if (result.is_ktx) {
                        load_ktx(vector.bytes.data(), vector.bytes.size() * sizeof(std::byte));
                      } else {
                        load_non_ktx(vector.bytes.data(), vector.bytes.size() * sizeof(std::byte));
                      }
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

std::optional<LoadedSceneBaseData> load_gltf_base(const std::filesystem::path& path) {
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
      fastgltf::Options::DecomposeNodeMatrices | fastgltf::Options::LoadExternalImages;

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

  std::vector<ImageUsage> img_usages(gltf.images.size(), ImageUsage::BaseColor);
  {
    ZoneScopedN("determine usages");
    auto set_usage = [&img_usages](const fastgltf::Texture& tex, ImageUsage usage) {
      std::size_t idx = tex.basisuImageIndex.value_or(tex.imageIndex.value_or(UINT32_MAX));
      if (idx != UINT32_MAX) {
        img_usages[idx] = usage;
      }
    };
    for (const auto& gltf_mat : gltf.materials) {
      if (gltf_mat.pbrData.baseColorTexture.has_value()) {
        set_usage(gltf.textures[gltf_mat.pbrData.baseColorTexture.value().textureIndex],
                  ImageUsage::BaseColor);
      }
      if (gltf_mat.pbrData.metallicRoughnessTexture.has_value()) {
        set_usage(gltf.textures[gltf_mat.pbrData.metallicRoughnessTexture.value().textureIndex],
                  ImageUsage::MetallicRoughness);
      }
      if (gltf_mat.emissiveTexture.has_value()) {
        set_usage(gltf.textures[gltf_mat.emissiveTexture.value().textureIndex],
                  ImageUsage::Emissive);
      }
      if (gltf_mat.normalTexture.has_value()) {
        set_usage(gltf.textures[gltf_mat.normalTexture.value().textureIndex], ImageUsage::Normal);
      }
      if (gltf_mat.occlusionTexture.has_value()) {
        set_usage(gltf.textures[gltf_mat.occlusionTexture.value().textureIndex],
                  ImageUsage::Occlusion);
      }
      if (gltf_mat.packedOcclusionRoughnessMetallicTextures) {
        if (gltf_mat.packedOcclusionRoughnessMetallicTextures->occlusionRoughnessMetallicTexture
                .has_value()) {
          set_usage(gltf.textures[gltf_mat.packedOcclusionRoughnessMetallicTextures
                                      ->occlusionRoughnessMetallicTexture.value()
                                      .textureIndex],
                    ImageUsage::OccRoughnessMetallic);
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
    if (img.is_ktx) {
      assert(img.ktx_data && "need ktx data if the img is ktx");
      assert(!img.non_ktx_data);
      auto* ktx = img.ktx_data.get();
      assert(ktx->numLevels > 0);
      u64 tot = 0;
      for (u32 level = 0; level < ktx->numLevels; level++) {
        size_t level_offset;
        ktxTexture_GetImageOffset(ktxTexture(ktx), level, 0, 0, &level_offset);
        u32 w = std::max(img.w >> level, 1u);
        u32 h = std::max(img.h >> level, 1u);
        size_t size = vk2::img_to_buffer_size(img.format, {w, h, 1});
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

      result->textures.emplace_back(vk2::create_texture_2d_mip(
          img.format, {img.w, img.h, 1}, vk2::TextureUsage::ReadOnly, ktx->numLevels));
      assert(tot == ktx->dataSize);
    } else {
      // TODO: mip gen?
      size_t size = vk2::img_to_buffer_size(img.format, {img.w, img.h, 1});
      assert(img.non_ktx_data);
      assert(!img.ktx_data);
      img_upload_infos.emplace_back(
          ImgUploadInfo{.extent = {img.w, img.w, 1},
                        .size = size,
                        .data = img.non_ktx_data.get(),
                        .staging_offset = staging_offset,
                        .level = 0,
                        .img_idx = static_cast<u32>(result->textures.size())});
      result->textures.emplace_back(vk2::create_texture_2d_mip(img.format, {img.w, img.h, 1},
                                                               vk2::TextureUsage::ReadOnly, 1));
      staging_offset += size;
    }
  }

  assert(result->textures.size() == images.size());
  {
    ZoneScopedN("upload images");
    constexpr size_t max_batch_upload_size = 1'000'000'000;
    size_t batch_upload_size = std::min(max_batch_upload_size, staging_offset);
    assert(batch_upload_size < max_batch_upload_size);
    vk2::Buffer* img_staging_buf = vk2::StagingBufferPool::get().acquire(batch_upload_size);
    size_t bytes_remaining = staging_offset;
    u64 img_i{};
    u64 curr_staging_offset = 0;
    u64 start_copy_idx{};
    // u64 end_copy_idx{};
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
      VkRender2::get().immediate_submit([start_copy_idx, end_copy_idx = img_i - 1,
                                         &img_upload_infos, &img_staging_buf, &result, &state,
                                         curr_staging_offset](VkCommandBuffer cmd) {
        state.reset(cmd);
        for (u64 i = start_copy_idx; i <= end_copy_idx; i++) {
          const auto& img_upload = img_upload_infos[i];
          state.transition(result->textures[img_upload.img_idx].image(),
                           VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        }
        state.flush_barriers();

        for (u64 i = start_copy_idx; i <= end_copy_idx; i++) {
          const auto& img_upload = img_upload_infos[i];
          const auto& texture = result->textures[img_upload.img_idx];
          vkCmdCopyBufferToImage2KHR(
              cmd, vk2::addr(VkCopyBufferToImageInfo2{
                       .sType = VK_STRUCTURE_TYPE_COPY_BUFFER_TO_IMAGE_INFO_2_KHR,
                       .srcBuffer = img_staging_buf->buffer(),
                       .dstImage = texture.image(),
                       .dstImageLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                       .regionCount = 1,
                       .pRegions = vk2::addr(VkBufferImageCopy2{
                           .sType = VK_STRUCTURE_TYPE_BUFFER_IMAGE_COPY_2,
                           .bufferOffset = img_upload.staging_offset - curr_staging_offset,
                           .bufferRowLength = 0,
                           .bufferImageHeight = 0,
                           .imageSubresource =
                               {
                                   .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                                   .mipLevel = img_upload.level,
                                   .layerCount = 1,
                               },
                           .imageExtent = VkExtent3D{img_upload.extent.x, img_upload.extent.y, 1}

                       })

                   }));
        }
        for (u64 i = start_copy_idx; i <= end_copy_idx; i++) {
          state.transition(result->textures[img_upload_infos[i].img_idx].image(),
                           VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                           VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                           VK_IMAGE_LAYOUT_READ_ONLY_OPTIMAL);
        }
        state.flush_barriers();
      });

      start_copy_idx = img_i;
    };
    while (bytes_remaining > 0) {
      if (bytes_remaining - img_upload_infos[img_i].size < 0) {
        flush_uploads();
      }

      futures.emplace_back(threads::pool.submit_task(
          [img_i, &img_upload_infos, curr_staging_offset, &img_staging_buf]() {
            const auto& img_upload = img_upload_infos[img_i];
            if (img_staging_buf->size() < img_upload.staging_offset + img_upload.size) {
              assert(0);
            } else {
              memcpy((char*)img_staging_buf->mapped_data() + img_upload.staging_offset -
                         curr_staging_offset,
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

  result->materials.reserve(gltf.materials.size());
  for (size_t i = 0; i < gltf.materials.size(); i++) {
    const auto& gltf_mat = gltf.materials[i];
    ZoneScopedN("determine usages");
    auto get_idx = [&gltf](const fastgltf::TextureInfo& info, u32 fallback) {
      const auto& tex = gltf.textures[info.textureIndex];
      return tex.basisuImageIndex.value_or(tex.imageIndex.value_or(fallback));
    };
    // TODO: pass default data as args
    u32 default_img_idx =
        VkRender2::get().get_default_data().white_img->view().sampled_img_resource().handle;
    Material mat{.albedo_idx = default_img_idx, .normal_idx = default_img_idx};

    if (gltf_mat.pbrData.baseColorTexture.has_value()) {
      mat.albedo_idx = get_idx(gltf_mat.pbrData.baseColorTexture.value(), default_img_idx);
    }
    if (gltf_mat.normalTexture.has_value()) {
      mat.normal_idx = get_idx(gltf_mat.normalTexture.value(), default_img_idx);
    }

    // TODO: others

    result->materials.emplace_back(mat);
  }
  // TODO: fix bad
  {
    for (auto& m : result->materials) {
      m.albedo_idx = result->textures[m.albedo_idx].view().sampled_img_resource().handle;
      m.normal_idx = result->textures[m.normal_idx].view().sampled_img_resource().handle;
    }
  }
  // std::array<std::future<void>, 2> scene_load_futures;
  // u32 scene_load_future_idx = 0;
  // scene_load_futures[scene_load_future_idx++] = threads::pool.submit_task([&]() { });
  size_t total_num_gltf_primitives = 0;
  for (const auto& m : gltf.meshes) {
    total_num_gltf_primitives += m.primitives.size();
  }
  result->mesh_draw_infos.resize(total_num_gltf_primitives);

  u32 primitive_idx{0};
  u32 mesh_idx{0};
  for (const auto& gltf_mesh : gltf.meshes) {
    for (const auto& gltf_prim : gltf_mesh.primitives) {
      u32 first_index = result->indices.size();
      u32 index_count = populate_indices(gltf_prim, gltf, result->indices);
      u32 first_vertex = result->vertices.size();
      u32 vertex_count = populate_vertices(gltf_prim, gltf, result->vertices);
      result->mesh_draw_infos[primitive_idx++] = {.first_index = first_index,
                                                  .index_count = index_count,
                                                  .first_vertex = first_vertex,
                                                  .vertex_count = vertex_count,
                                                  .mesh_idx = mesh_idx};
    }
    mesh_idx++;
  };
  // scene_load_futures[scene_load_future_idx++] = threads::pool.submit_task([&result,
  // &gltf]() {});

  load_scene_graph_data(result->scene_graph_data, gltf);
  // for (auto& f : scene_load_futures) {
  //   f.get();
  // }

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
  vk2::Buffer* staging = vk2::StagingBufferPool::get().acquire(vertices_size + indices_size);
  memcpy(staging->mapped_data(), base_scene_data.vertices.data(), vertices_size);
  memcpy((char*)staging->mapped_data() + vertices_size, base_scene_data.indices.data(),
         indices_size);
  return LoadedSceneData{
      .scene_graph_data = std::move(base_scene_data.scene_graph_data),
      .samplers = std::move(base_scene_data.samplers),
      .materials = std::move(base_scene_data.materials),
      .textures = std::move(base_scene_data.textures),
      .mesh_draw_infos = std::move(base_scene_data.mesh_draw_infos),
      .vert_idx_staging = staging,
      .vertices_size = vertices_size,
      .indices_size = indices_size,
  };
}
}  // namespace gfx
