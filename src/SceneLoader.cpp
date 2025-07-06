#include "SceneLoader.hpp"

#include <ktx.h>
#include <ktxvulkan.h>
#include <volk.h>
#include <vulkan/vk_enum_string_helper.h>
#include <vulkan/vulkan_core.h>

#include <algorithm>
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
#include <glm/gtc/type_ptr.hpp>
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

struct CalcTangentsVertexInfo {
  struct BaseOffset {
    void* base;
    u32 offset;
    u32 stride;
  };
  BaseOffset pos;
  BaseOffset normal;
  BaseOffset uv_x;
  BaseOffset uv_y;
  BaseOffset tangent;
};

template <typename IndexT>
void calc_tangents(const CalcTangentsVertexInfo& info, std::span<IndexT> indices) {
  ZoneScoped;
  SMikkTSpaceContext ctx{};
  SMikkTSpaceInterface interface{};
  ctx.m_pInterface = &interface;

  struct MyCtx {
    MyCtx(const CalcTangentsVertexInfo& info, std::span<IndexT>& indices)
        : info(info), indices(indices), num_faces(indices.size() / 3) {}
    CalcTangentsVertexInfo info;
    std::span<IndexT> indices;
    size_t num_faces{};
    int face_size = 3;
    int get_index(int face_i, int vert_i) { return indices[(face_i * face_size) + vert_i]; }
  };

  MyCtx my_ctx{info, indices};
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
    u32 idx = my_ctx.get_index(iFace, iVert);
    vec3* pos_ptr =
        reinterpret_cast<vec3*>(reinterpret_cast<char*>(my_ctx.info.pos.base) +
                                (idx * my_ctx.info.pos.stride) + my_ctx.info.pos.offset);
    vec3& pos = *pos_ptr;
    fvPosOut[0] = pos.x;
    fvPosOut[1] = pos.y;
    fvPosOut[2] = pos.z;
  };

  interface.m_getNormal = [](const SMikkTSpaceContext* ctx, float fvNormOut[], const int iFace,
                             const int iVert) {
    MyCtx& my_ctx = *reinterpret_cast<MyCtx*>(ctx->m_pUserData);
    u32 idx = my_ctx.get_index(iFace, iVert);
    vec3* normal_ptr =
        reinterpret_cast<vec3*>(reinterpret_cast<char*>(my_ctx.info.normal.base) +
                                (idx * my_ctx.info.normal.stride) + my_ctx.info.normal.offset);
    vec3& normal = *normal_ptr;
    fvNormOut[0] = normal.x;
    fvNormOut[1] = normal.y;
    fvNormOut[2] = normal.z;
  };
  interface.m_getTexCoord = [](const SMikkTSpaceContext* ctx, float fvTexcOut[], const int iFace,
                               const int iVert) {
    MyCtx& my_ctx = *reinterpret_cast<MyCtx*>(ctx->m_pUserData);
    u32 idx = my_ctx.get_index(iFace, iVert);
    auto* uv_x_ptr =
        reinterpret_cast<float*>(reinterpret_cast<char*>(my_ctx.info.uv_x.base) +
                                 (idx * my_ctx.info.uv_x.stride) + my_ctx.info.uv_x.offset);
    auto* uv_y_ptr =
        reinterpret_cast<float*>(reinterpret_cast<char*>(my_ctx.info.uv_y.base) +
                                 (idx * my_ctx.info.uv_y.stride) + my_ctx.info.uv_y.offset);
    fvTexcOut[0] = *uv_x_ptr;
    fvTexcOut[1] = *uv_y_ptr;
  };

  interface.m_setTSpaceBasic = [](const SMikkTSpaceContext* ctx, const float fvTangent[],
                                  const float, const int iFace, const int iVert) {
    MyCtx& my_ctx = *reinterpret_cast<MyCtx*>(ctx->m_pUserData);
    u32 idx = my_ctx.get_index(iFace, iVert);
    vec3* tangent_ptr =
        reinterpret_cast<vec3*>(reinterpret_cast<char*>(my_ctx.info.tangent.base) +
                                (idx * my_ctx.info.tangent.stride) + my_ctx.info.tangent.offset);
    vec3& tangent = *tangent_ptr;
    tangent.x = fvTangent[0];
    tangent.y = fvTangent[1];
    tangent.z = fvTangent[2];
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

void set_node_transform_from_gltf_node(mat4& local_transform, NodeTransform& transform_data,
                                       const fastgltf::Node& gltf_node) {
  if (std::holds_alternative<fastgltf::math::fmat4x4>(gltf_node.transform)) {
    const auto& mat_data = std::get<fastgltf::math::fmat4x4>(gltf_node.transform);
    local_transform = glm::make_mat4(mat_data.data());
    decompose_matrix(local_transform, transform_data.translation, transform_data.rotation,
                     transform_data.scale);
  } else {
    const auto& trs = std::get<fastgltf::TRS>(gltf_node.transform);
    transform_data.translation = glm::make_vec3(trs.translation.data());
    transform_data.rotation =
        glm::quat(trs.rotation[3], trs.rotation[0], trs.rotation[1], trs.rotation[2]);
    transform_data.scale = glm::make_vec3(trs.scale.data());
    transform_data.to_mat4(local_transform);
  }
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

i32 add_node(Scene2& scene, i32 parent, i32 level) {
  size_t node_i = scene.hierarchies.size();
  scene.local_transforms.emplace_back(1);
  scene.global_transforms.emplace_back(1);
  scene.node_mesh_indices.emplace_back(-1);
  scene.node_transforms.emplace_back();
  scene.node_flags.emplace_back(0);
  scene.hierarchies.push_back(Hierarchy{.parent = parent});
  // if parent exists, update it
  if (parent > -1) {
    i32 first_child = scene.hierarchies[parent].first_child;
    if (first_child == -1) {
      // no sibling, node is first child of parent and own last sibling
      scene.hierarchies[parent].first_child = node_i;
      scene.hierarchies[node_i].last_sibling = node_i;
    } else {
      // sibling exists, traverse to find last sibling
      i32 last = scene.hierarchies[first_child].last_sibling;
      if (last <= -1) {
        for (last = first_child; scene.hierarchies[last].next_sibling != -1;
             last = scene.hierarchies[last].next_sibling);
      }
      scene.hierarchies[last].next_sibling = node_i;
      scene.hierarchies[first_child].last_sibling = node_i;
    }
  }
  scene.hierarchies[node_i].level = level;
  scene.hierarchies[node_i].next_sibling = -1;
  scene.hierarchies[node_i].first_child = -1;
  scene.hierarchies[node_i].last_sibling = -1;
  return node_i;
}

void traverse(Scene2& scene, fastgltf::Asset& gltf, const Material& default_material,
              const std::vector<Material>& materials, LoadedSceneBaseData& result,
              std::vector<int>& gltf_node_i_to_node_i,
              const std::vector<u32>& prim_offsets_of_meshes) {
  struct NodeStackEntry {
    int gltf_node_i;
    int parent_i;
    int level;
  };
  std::vector<NodeStackEntry> to_add_node_stack;
  int root_node = add_node(scene, -1, 0);
  scene.node_to_node_name_idx.emplace(root_node, scene.node_names.size());
  scene.node_names.emplace_back("Root node");

  auto& scene_node_indices = gltf.scenes[gltf.defaultScene.value_or(0)].nodeIndices;
  to_add_node_stack.reserve(scene_node_indices.size());

  for (size_t i = scene_node_indices.size(); i > 0; i--) {
    auto gltf_node_i = scene_node_indices[i - 1];
    // to_add_node_stack.emplace_back(
    //     NodeStackEntry{.gltf_node_i = static_cast<int>(gltf_node_i), .parent_i = -1, .level =
    //     0});
    to_add_node_stack.emplace_back(NodeStackEntry{
        .gltf_node_i = static_cast<int>(gltf_node_i), .parent_i = root_node, .level = 1});
  }

  while (to_add_node_stack.size() > 0) {
    int gltf_node_i = to_add_node_stack.back().gltf_node_i;
    int parent_i = to_add_node_stack.back().parent_i;
    int level = to_add_node_stack.back().level;
    to_add_node_stack.pop_back();

    assert((size_t)gltf_node_i < gltf_node_i_to_node_i.size());
    const auto& gltf_node = gltf.nodes[gltf_node_i];
    i32 new_node = add_node(scene, parent_i, level);
    assert(gltf_node_i_to_node_i[gltf_node_i] == -1);
    gltf_node_i_to_node_i[gltf_node_i] = new_node;
    set_node_transform_from_gltf_node(scene.local_transforms[new_node],
                                      scene.node_transforms[new_node], gltf.nodes[gltf_node_i]);
    if (gltf_node.name.size() > 0) {
      scene.node_to_node_name_idx.emplace(new_node, scene.node_names.size());
      scene.node_names.emplace_back(gltf_node.name);
    }

    if (gltf_node.meshIndex.has_value()) {
      auto gltf_mesh_i = gltf_node.meshIndex.value();
      const auto& mesh = gltf.meshes[gltf_mesh_i];
      u32 primitive_i = 0;
      for (const auto& primitive : mesh.primitives) {
        i32 submesh_node = add_node(scene, new_node, level + 1);
        // TODO: name only during editing/string allocation?
        scene.node_to_node_name_idx[submesh_node] = scene.node_names.size();
        scene.node_names.emplace_back(std::string(gltf_node.name) + "_mesh_" +
                                      std::to_string(primitive_i));
        scene.node_mesh_indices[submesh_node] = scene.mesh_datas.size();
        auto& mesh_data = scene.mesh_datas.emplace_back(MeshData{});
        mesh_data.mesh_idx = prim_offsets_of_meshes[gltf_mesh_i] + primitive_i;
        mesh_data.material_id = static_cast<u32>(primitive.materialIndex.value_or(UINT32_MAX));
        mesh_data.pass_flags = mesh_data.material_id != UINT32_MAX
                                   ? materials[mesh_data.material_id].get_pass_flags()
                                   : default_material.get_pass_flags();
        primitive_i++;
      }
    }

    for (const auto& gltf_child_i : gltf_node.children) {
      to_add_node_stack.emplace_back(NodeStackEntry{
          .gltf_node_i = static_cast<int>(gltf_child_i), .parent_i = new_node, .level = level + 1});
    }
  }

  auto& animations = result.animations;
  animations.reserve(gltf.animations.size());
  for (auto& animation : gltf.animations) {
    Animation anim{.name = std::string{animation.name}};
    for (auto& sampler : animation.samplers) {
      AnimSampler new_sampler{};
      auto& input_accessor = gltf.accessors[sampler.inputAccessor];
      auto& inputs = new_sampler.inputs;
      inputs.reserve(input_accessor.count);
      fastgltf::iterateAccessor<float>(gltf, input_accessor,
                                       [&](float t) { inputs.emplace_back(t); });
      // TODO: reserve size?
      auto& outputs_raw = new_sampler.outputs_raw;
      auto& output_accessor = gltf.accessors[sampler.outputAccessor];
      switch (output_accessor.type) {
        case fastgltf::AccessorType::Vec3: {
          fastgltf::iterateAccessor<vec3>(gltf, output_accessor, [&outputs_raw](vec3 v) {
            for (int i = 0; i < 3; i++) {
              outputs_raw.emplace_back(v[i]);
            }
          });
          break;
        }
        case fastgltf::AccessorType::Vec2: {
          fastgltf::iterateAccessor<vec2>(gltf, output_accessor, [&outputs_raw](vec2 v) {
            for (int i = 0; i < 2; i++) {
              outputs_raw.emplace_back(v[i]);
            }
          });
          break;
        }
        case fastgltf::AccessorType::Vec4: {
          fastgltf::iterateAccessor<vec4>(gltf, output_accessor, [&outputs_raw](vec4 v) {
            quat q = glm::normalize(quat{v[3], v[0], v[1], v[2]});
            outputs_raw.emplace_back(q[0]);
            outputs_raw.emplace_back(q[1]);
            outputs_raw.emplace_back(q[2]);
            outputs_raw.emplace_back(q[3]);
          });
          break;
        }
        case fastgltf::AccessorType::Scalar: {
          fastgltf::iterateAccessor<float>(
              gltf, output_accessor, [&outputs_raw](float v) { outputs_raw.emplace_back(v); });
          break;
        }
        default:
          assert(0 && "invalid accessor type for animation");
          break;
      }

      anim.samplers.emplace_back(std::move(new_sampler));
    }

    anim.duration = 0.0f;
    for (const auto& sampler : anim.samplers) {
      if (!sampler.inputs.empty()) {
        float max_time = *std::ranges::max_element(sampler.inputs);
        anim.duration = std::max(anim.duration, max_time);
      }
    }

    for (auto& channel : animation.channels) {
      anim.channels.nodes.push_back(
          channel.nodeIndex.has_value() ? gltf_node_i_to_node_i[channel.nodeIndex.value()] : -1);
      anim.channels.sampler_indices.push_back(channel.samplerIndex);
      assert(anim.channels.nodes.back() != -1);
      AnimationPath anim_path{};
      switch (channel.path) {
        case fastgltf::AnimationPath::Translation:
          anim_path = AnimationPath::Translation;
          break;
        case fastgltf::AnimationPath::Rotation:
          anim_path = AnimationPath::Rotation;
          break;
        case fastgltf::AnimationPath::Scale:
          anim_path = AnimationPath::Scale;
          break;
        case fastgltf::AnimationPath::Weights:
          anim_path = AnimationPath::Weights;
          break;
        default:
          assert(0 && "unsupported animation path");
      }
      anim.channels.anim_paths.emplace_back(anim_path);
    }
    result.animations.emplace_back(std::move(anim));
  }
  auto& out_skins = result.scene_graph_data.skins;
  u32 tot_matrices = 0;
  for (auto& gltf_skin : gltf.skins) {
    auto& new_skin = out_skins.emplace_back(SkinData{
        .name = std::string{gltf_skin.name},
    });

    if (gltf_skin.inverseBindMatrices.has_value()) {
      auto& accessor = gltf.accessors[gltf_skin.inverseBindMatrices.value()];
      assert(accessor.count == gltf_skin.joints.size());
      new_skin.inverse_bind_matrices.reserve(gltf_skin.joints.size());
      fastgltf::iterateAccessor<mat4>(gltf, accessor, [&new_skin](const mat4& m) {
        new_skin.inverse_bind_matrices.emplace_back(m);
      });
    } else {
      new_skin.inverse_bind_matrices.resize(gltf_skin.joints.size());
    }
    new_skin.model_bone_mat_start_i = tot_matrices;
    tot_matrices += gltf_skin.joints.size();
    // tot_matrices += new_skin.inverse_bind_matrices.size();

    new_skin.joint_node_indices.reserve(gltf_skin.joints.size());
    for (const auto& joint_node_index : gltf_skin.joints) {
      int node_i = gltf_node_i_to_node_i[joint_node_index];
      out_skins.back().joint_node_indices.emplace_back(node_i);
      scene.node_flags[node_i] |= Scene2::NodeFlag_IsJointBit;
    }
  }
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
          ImgUploadInfo{.extent = {img.w, img.h, 1},
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
  if (staging_offset > 0) {
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
      mat.pbr_factors.w = gltf_mat.alphaCutoff;

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

  std::vector<int> gltf_node_i_to_node_i(gltf.nodes.size(), -1);
  std::vector<u32> prim_offsets_of_meshes(gltf.meshes.size());
  {
    u32 offset = 0;
    for (u32 mesh_idx = 0; mesh_idx < gltf.meshes.size(); mesh_idx++) {
      prim_offsets_of_meshes[mesh_idx] = offset;
      offset += gltf.meshes[mesh_idx].primitives.size();
    }
  }
  traverse(result->scene_graph_data, gltf, Material{}, result->materials, *result,
           gltf_node_i_to_node_i, prim_offsets_of_meshes);
  mark_changed(result->scene_graph_data, 0);
  recalc_global_transforms(result->scene_graph_data);

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
    u32 num_animated_vertices{};
    {
      u32 primitive_idx{0};
      for (const auto& gltf_mesh : gltf.meshes) {
        for (const auto& gltf_prim : gltf_mesh.primitives) {
          u32 first_index = num_indices;
          u32 first_vertex = num_vertices;
          u32 first_animated_vertex = num_animated_vertices;
          const auto* pos_attrib = gltf_prim.findAttribute("POSITION");
          if (pos_attrib == gltf_prim.attributes.end()) {
            return {};
          }
          const auto* joints_attrib = gltf_prim.findAttribute("JOINTS_0");
          bool animated = joints_attrib != gltf_prim.attributes.end();

          u32 vertex_count = gltf.accessors[pos_attrib->accessorIndex].count;
          num_vertices += vertex_count;
          if (animated) {
            num_animated_vertices += vertex_count;
          }

          const auto& index_accessor = gltf.accessors[gltf_prim.indicesAccessor.value()];
          u32 index_count = index_accessor.count;
          num_indices += index_count;

          result->mesh_draw_infos[primitive_idx++] = {
              .first_index = first_index,
              .index_count = index_count,
              .first_vertex = first_vertex,
              .vertex_count = vertex_count,
              .first_animated_vertex = first_animated_vertex};
        }
      };
    }
    result->indices.resize(num_indices);
    result->vertices.resize(num_vertices);
    result->animated_vertices.resize(num_animated_vertices);

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

    futures.clear();

    std::vector<bool> processed_meshes(gltf.meshes.size());
    for (size_t gltf_node_i = 0; gltf_node_i < gltf.nodes.size(); gltf_node_i++) {
      auto& gltf_node = gltf.nodes[gltf_node_i];
      if (!gltf_node.meshIndex.has_value()) {
        continue;
      }
      auto mesh_idx = gltf_node.meshIndex.value();
      assert(!processed_meshes[mesh_idx]);
      if (processed_meshes[mesh_idx]) {
        continue;
      }
      processed_meshes[mesh_idx] = true;
      for (size_t primitive_idx = 0; primitive_idx < gltf.meshes[mesh_idx].primitives.size();
           primitive_idx++) {
        auto mesh_draw_idx = prim_offsets_of_meshes[mesh_idx] + primitive_idx;
        futures.emplace_back(threads::pool.submit_task([&gltf, mesh_idx, primitive_idx, &result,
                                                        mesh_draw_idx, loaded_tangents_from_disk,
                                                        gltf_node_i]() {
          ZoneScopedN("gltf process primitives");
          const auto& primitive = gltf.meshes[mesh_idx].primitives[primitive_idx];
          const auto& index_accessor = gltf.accessors[primitive.indicesAccessor.value()];
          auto& mesh_draw_info = result->mesh_draw_infos[mesh_draw_idx];
          u32 start_idx = mesh_draw_info.first_index;
          fastgltf::iterateAccessorWithIndex<u32>(gltf, index_accessor, [&](uint32_t index, u32 i) {
            result->indices[start_idx + i] = index;
          });
          const auto* pos_attrib = primitive.findAttribute("POSITION");
          if (pos_attrib == primitive.attributes.end()) {
            assert(0);
          }
          const auto* joints_attrib = primitive.findAttribute("JOINTS_0");
          const auto* weights_attrib = primitive.findAttribute("WEIGHTS_0");
          bool animated = joints_attrib != primitive.attributes.end();
          const auto& pos_accessor = gltf.accessors[pos_attrib->accessorIndex];
          u64 start_i_static = mesh_draw_info.first_vertex;
          u64 start_i_animated = mesh_draw_info.first_animated_vertex;
          u32 o = 0;
          if (gltf.nodes[gltf_node_i].skinIndex.has_value()) {
            o = result->scene_graph_data.skins[gltf.nodes[gltf_node_i].skinIndex.value()]
                    .model_bone_mat_start_i;
          }

          if (animated) {
            fastgltf::iterateAccessorWithIndex<vec3>(
                gltf, pos_accessor, [&result, start_i_animated](const vec3& pos, u32 i) {
                  result->animated_vertices[start_i_animated + i].pos = pos;
                });

            assert(weights_attrib != primitive.attributes.end());

            fastgltf::iterateAccessorWithIndex<uvec4>(
                gltf, gltf.accessors[joints_attrib->accessorIndex],
                [&result, &start_i_animated, &o](const uvec4& joints, size_t i) {
                  for (u32 j = 0; j < 4; j++) {
                    result->animated_vertices[start_i_animated + i].bone_id[j] = joints[j] + o;
                  }
                });

            fastgltf::iterateAccessorWithIndex<vec4>(
                gltf, gltf.accessors[weights_attrib->accessorIndex],
                [&result, start_i_animated](const vec4& weights, size_t i) {
                  for (u32 j = 0; j < 4; j++) {
                    result->animated_vertices[start_i_animated + i].weights[j] = weights[j];
                  }
                });

          } else {
            fastgltf::iterateAccessorWithIndex<vec3>(
                gltf, pos_accessor, [&result, start_i_static](const vec3& pos, u32 i) {
                  result->vertices[start_i_static + i].pos = pos;
                });
          }

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

          const auto* normal_attrib = primitive.findAttribute("NORMAL");
          if (normal_attrib != primitive.attributes.end()) {
            const auto& normal_accessor = gltf.accessors[normal_attrib->accessorIndex];
            auto range = fastgltf::iterateAccessor<glm::vec3>(gltf, normal_accessor);
            assert(normal_accessor.count == pos_accessor.count);
            if (animated) {
              u64 i = start_i_animated;
              for (const glm::vec3& normal : range) {
                result->animated_vertices[i++].normal = vec4{normal, 0.};
              }
            } else {
              u64 i = start_i_static;
              for (const glm::vec3& normal : range) {
                result->vertices[i++].normal = normal;
              }
            }
          }

          const auto* uv_attrib = primitive.findAttribute("TEXCOORD_0");
          if (uv_attrib != primitive.attributes.end()) {
            const auto& accessor = gltf.accessors[uv_attrib->accessorIndex];
            assert(accessor.count == pos_accessor.count);
            auto range = fastgltf::iterateAccessor<glm::vec2>(gltf, accessor);
            if (animated) {
              u64 i = start_i_animated;
              for (const glm::vec2& uv : range) {
                result->animated_vertices[i].uv_x = uv.x;
                result->animated_vertices[i++].uv_y = uv.y;
              }
            } else {
              u64 i = start_i_static;
              for (const glm::vec2& uv : range) {
                result->vertices[i].uv_x = uv.x;
                result->vertices[i++].uv_y = uv.y;
              }
            }
          }
          const auto* tangent_attrib = primitive.findAttribute("TANGENT");
          if (tangent_attrib != primitive.attributes.end()) {
            const auto& accessor = gltf.accessors[tangent_attrib->accessorIndex];
            if (animated) {
              u64 i = start_i_animated;
              assert(accessor.count == pos_accessor.count);
              if (accessor.type == fastgltf::AccessorType::Vec3) {
                auto range = fastgltf::iterateAccessor<glm::vec3>(gltf, accessor);
                for (const glm::vec3& tangent : range) {
                  result->animated_vertices[i++].tangent = vec4(tangent, 0.);
                }
              } else if (accessor.type == fastgltf::AccessorType::Vec4) {
                auto range = fastgltf::iterateAccessor<glm::vec4>(gltf, accessor);
                for (const glm::vec4& tangent : range) {
                  result->animated_vertices[i++].tangent = tangent;
                }
              }
            } else {
              u64 i = start_i_static;
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
            }
          } else if (!loaded_tangents_from_disk) {
            if (animated) {
              auto& verts = result->animated_vertices;
              CalcTangentsVertexInfo info{
                  .pos = {.base = verts.data() + (start_i_animated),
                          .offset = offsetof(AnimatedVertex, pos),
                          .stride = sizeof(AnimatedVertex)},
                  .normal = {.base = verts.data() + (start_i_animated),
                             .offset = offsetof(AnimatedVertex, normal),
                             .stride = sizeof(AnimatedVertex)},
                  .uv_x = {.base = verts.data() + (start_i_animated),
                           .offset = offsetof(AnimatedVertex, uv_x),
                           .stride = sizeof(AnimatedVertex)},
                  .uv_y = {.base = verts.data() + (start_i_animated),
                           .offset = offsetof(AnimatedVertex, uv_y),
                           .stride = sizeof(AnimatedVertex)},
                  .tangent = {.base = verts.data() + (start_i_animated),
                              .offset = offsetof(AnimatedVertex, tangent),
                              .stride = sizeof(AnimatedVertex)},
              };

              calc_tangents<u32>(info, std::span(result->indices.data() + (start_idx),
                                                 mesh_draw_info.index_count));
            } else {
              CalcTangentsVertexInfo info{
                  .pos = {.base = result->vertices.data() + (start_i_static),
                          .offset = offsetof(Vertex, pos),
                          .stride = sizeof(Vertex)},
                  .normal = {.base = result->vertices.data() + (start_i_static),
                             .offset = offsetof(Vertex, normal),
                             .stride = sizeof(Vertex)},
                  .uv_x = {.base = result->vertices.data() + (start_i_static),
                           .offset = offsetof(Vertex, uv_x),
                           .stride = sizeof(Vertex)},
                  .uv_y = {.base = result->vertices.data() + (start_i_static),
                           .offset = offsetof(Vertex, uv_y),
                           .stride = sizeof(Vertex)},
                  .tangent = {.base = result->vertices.data() + (start_i_static),
                              .offset = offsetof(Vertex, tangent),
                              .stride = sizeof(Vertex)},
              };
              calc_tangents<u32>(info, std::span(result->indices.data() + (start_idx),
                                                 mesh_draw_info.index_count));
            }
          }
        }));
      }
    }

    {
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
  return LoadedSceneData{.scene_graph_data = std::move(base_scene_data.scene_graph_data),
                         .materials = std::move(base_scene_data.materials),
                         .textures = std::move(base_scene_data.textures),
                         .mesh_draw_infos = std::move(base_scene_data.mesh_draw_infos),
                         .vertices = std::move(base_scene_data.vertices),
                         .animated_vertices = std::move(base_scene_data.animated_vertices),
                         .indices = std::move(base_scene_data.indices),
                         .animations = std::move(base_scene_data.animations)};
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

uvec2 AnimSampler::get_time_indices(float t) const {
  auto it = std::ranges::lower_bound(inputs, t);
  size_t time_i = 0;
  if (it != inputs.begin()) {
    time_i = std::distance(inputs.begin(), it) - 1;
  }
  size_t next_time_i = inputs.size() == 1 ? 0 : (time_i + 1) % inputs.size();
  return {time_i, next_time_i};
}

}  // namespace gfx
