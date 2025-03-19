#pragma once
#include <string>
#include <vector>

#include "Common.hpp"

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

struct SceneGraphData {
  std::vector<NodeData> node_datas;
  std::vector<u32> mesh_node_indices;
  std::vector<u32> root_node_indices;
  u32 primitive_instance_count;
};
