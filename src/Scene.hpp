#pragma once
#include <string>
#include <vector>

#include "Common.hpp"

namespace gfx {

struct MeshBounds {
  glm::vec3 origin;
  float radius;
  glm::vec3 extents;
};

using PassFlags = u8;
enum PassFlagBits : PassFlags {
  PassFlags_None = 0,
  PassFlags_Opaque = 1 << 0,
  PassFlags_OpaqueAlpha = 1 << 1,
  PassFlags_Transparent = 1 << 2,
};

struct MeshData {
  u32 mesh_idx;
  u32 material_id;
  PassFlags pass_flags{};
};

struct NodeData {
  static constexpr u32 null_idx = UINT32_MAX;
  mat4 local_transform{mat4{1}};
  mat4 world_transform{mat4{1}};
  vec3 translation;
  quat rotation;
  vec3 scale;
  std::vector<u64> children_indices;
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

struct Hierarchy {
  i32 parent{-1};
  i32 first_child{-1};
  i32 next_sibling{-1};
  i32 last_sibling{-1};
  i32 level{0};
};

struct Material {
  vec4 emissive_factors{0.};
  vec4 albedo_factors{1.};
  vec4 pbr_factors;
  uvec4 ids1;  // albedo, normal, metal_rough, emissive
  uvec4 ids2;  // ao
  [[nodiscard]] PassFlags get_pass_flags() const;
  [[nodiscard]] bool is_double_sided() const;
};

struct Scene2 {
  std::vector<mat4> local_transforms;
  std::vector<mat4> global_transforms;
  std::vector<Hierarchy> hierarchies;
  std::vector<std::string> node_names;
  std::vector<u32> node_material_indices;
  std::unordered_map<i32, MeshData> node_to_mesh_data;
  std::unordered_map<i32, i32> node_to_node_name_idx;
  // one mesh contains multiple primitives
  std::vector<u32> node_mesh_indices;
  static constexpr int max_node_depth{10};
  std::vector<u32> changed_this_frame[max_node_depth];
};

void mark_changed(Scene2& scene, int node);
bool recalc_global_transforms(Scene2& scene);

}  // namespace gfx
