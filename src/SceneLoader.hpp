#pragma once

#include <filesystem>
#include <optional>
#include <vector>

#include "Common.hpp"
#include "vk2/Buffer.hpp"
#include "vk2/Texture.hpp"
namespace gfx {

struct Vertex {
  vec3 pos;
  float uv_x;
  vec3 normal;
  float uv_y;
};

struct Box3D {
  vec3 min;
  vec3 max;
};

// TODO: SoA instead of AoS
struct NodeData {
  static constexpr u32 null_idx = UINT32_MAX;
  mat4 local_transform;
  mat4 world_transform{mat4{1}};
  std::vector<u64> children_indices;
  std::string name;
  u32 mesh_idx{null_idx};
  u32 parent_idx{null_idx};
};

struct PrimitiveDrawInfo {
  u32 first_index;
  u32 index_count;
  u32 first_vertex;
  u32 vertex_count;
  u32 mesh_idx;
};

// struct RawMesh {
//   std::vector<Vertex> vertices;
//   std::vector<u32> indices;
//   // Box3D bbox;
// };

struct LoadedSceneData {
  vk2::Buffer vertex_buffer;
  vk2::Buffer index_buffer;
  std::vector<vk2::Sampler> samplers;
};

struct SceneGraphData {
  std::vector<NodeData> node_datas;
  std::vector<u32> mesh_node_indices;
  std::vector<u32> root_node_indices;
  u32 primitive_instance_count;
};

struct LoadedSceneBaseData {
  SceneGraphData scene_graph_data;
  std::vector<PrimitiveDrawInfo> primitive_draw_infos;
  std::vector<Vertex> vertices;
  std::vector<u32> indices;
  std::vector<vk2::Sampler> samplers;
};

std::optional<LoadedSceneBaseData> load_gltf_base(const std::filesystem::path& path);

std::optional<LoadedSceneData> load_gltf(const std::filesystem::path& path);

}  // namespace gfx
