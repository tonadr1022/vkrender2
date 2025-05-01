#pragma once
#include <string>
#include <vector>

#include "Common.hpp"
#include "vk2/Handle.hpp"

struct MeshBounds {
  glm::vec3 origin;
  float radius;
  glm::vec3 extents;
};

using PassFlags = u8;
enum PassFlagBits : PassFlags {
  PassFlags_Opaque = 1 << 0,
  PassFlags_OpaqueAlpha = 1 << 1,
  PassFlags_Transparent = 1 << 2,
};

struct NodeData {
  static constexpr u32 null_idx = UINT32_MAX;
  mat4 local_transform{mat4{1}};
  mat4 world_transform{mat4{1}};
  vec3 translation;
  quat rotation;
  vec3 scale;
  std::vector<u64> children_indices;
  struct MeshData {
    u32 mesh_idx;
    u16 material_id;
    PassFlags pass_flags{};
  };
  std::vector<MeshData> meshes;
  std::string name;
  u32 parent_idx{null_idx};
};

struct SceneLoadData {
  std::vector<NodeData> node_datas;
  // std::vector<MeshBounds> node_mesh_bounds;
  std::vector<u32> mesh_node_indices;
  std::vector<u32> root_node_indices;
};

VK2_DEFINE_HANDLE(Scene);
