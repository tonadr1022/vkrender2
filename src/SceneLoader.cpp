#include "SceneLoader.hpp"

#include <volk.h>
#include <vulkan/vulkan_core.h>

#include <fastgltf/core.hpp>
#include <fastgltf/glm_element_traits.hpp>
#include <fastgltf/tools.hpp>
#include <fastgltf/types.hpp>

#include "vk2/StagingBufferPool.hpp"

// #include "ThreadPool.hpp"

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/quaternion.hpp>
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

u32 populate_indices(const fastgltf::Primitive& primitive, const fastgltf::Asset& gltf,
                     std::vector<uint32_t>& indices) {
  if (!primitive.indicesAccessor.has_value()) {
    return 0;
  }
  const auto& index_accessor = gltf.accessors[primitive.indicesAccessor.value()];
  indices.resize(indices.size() + index_accessor.count);
  size_t i = 0;
  fastgltf::iterateAccessor<u32>(gltf, index_accessor,
                                 [&](uint32_t index) { indices[i++] = index; });
  return index_accessor.count;
}

u32 populate_vertices(const fastgltf::Primitive& primitive, const fastgltf::Asset& gltf,
                      std::vector<Vertex>& vertices) {
  const auto* pos_attrib = primitive.findAttribute("POSITION");
  if (pos_attrib == primitive.attributes.end()) {
    return 0;
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
      vertices[i].uv_x = uv.x;
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

void load_scene_graph_data(SceneGraphData& result, fastgltf::Asset& gltf) {
  ZoneScoped;
  // aggregate nodes
  result.node_datas.reserve(gltf.nodes.size());
  for (const fastgltf::Node& gltf_node : gltf.nodes) {
    NodeData new_node;
    new_node.mesh_idx = gltf_node.meshIndex.value_or(NodeData::null_idx);
    if (gltf_node.meshIndex.has_value()) {
      result.mesh_node_indices.emplace_back(new_node.mesh_idx);
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
void load_samplers(const fastgltf::Asset& gltf, std::vector<vk2::Sampler>& samplers) {
  ZoneScoped;
  samplers.reserve(gltf.samplers.size());
  for (const auto& sampler : gltf.samplers) {
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
  load_samplers(gltf, result->samplers);

  // std::array<std::future<void>, 2> scene_load_futures;
  // u32 scene_load_future_idx = 0;
  // scene_load_futures[scene_load_future_idx++] = threads::pool.submit_task([&]() { });
  size_t total_num_gltf_primitives = 0;
  for (const auto& m : gltf.meshes) {
    total_num_gltf_primitives += m.primitives.size();
  }
  result->primitive_draw_infos.resize(total_num_gltf_primitives);

  u32 primitive_idx{0};
  u32 mesh_idx{0};
  for (const auto& gltf_mesh : gltf.meshes) {
    for (const auto& gltf_prim : gltf_mesh.primitives) {
      u32 first_index = result->indices.size();
      u32 index_count = populate_indices(gltf_prim, gltf, result->indices);
      u32 first_vertex = result->vertices.size();
      u32 vertex_count = populate_vertices(gltf_prim, gltf, result->vertices);
      result->primitive_draw_infos[primitive_idx] = {.first_index = first_index,
                                                     .index_count = index_count,
                                                     .first_vertex = first_vertex,
                                                     .vertex_count = vertex_count,
                                                     .mesh_idx = mesh_idx};
      primitive_idx++;
    }
    mesh_idx++;
  };
  // scene_load_futures[scene_load_future_idx++] = threads::pool.submit_task([&result, &gltf]() {});

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
      .vert_idx_staging = staging,
      .vertices_size = vertices_size,
      .indices_size = indices_size,
  };
}

}  // namespace gfx
