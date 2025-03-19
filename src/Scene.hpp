#pragma once
#include <string>
#include <vector>

#include "Common.hpp"

struct NodeData {
  static constexpr u32 null_idx = UINT32_MAX;
  mat4 local_transform;
  mat4 world_transform{mat4{1}};
  std::vector<u64> children_indices;
  struct MeshData {
    u32 material_id;
    u32 mesh_idx;
  };
  std::vector<MeshData> meshes;
  std::string name;
  u32 parent_idx{null_idx};
};

struct SceneLoadData {
  std::vector<NodeData> node_datas;
  std::vector<u32> mesh_node_indices;
  std::vector<u32> root_node_indices;
};
