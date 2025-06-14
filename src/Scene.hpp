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

struct Hierarchy {
  i32 parent{-1};
  i32 first_child{-1};
  i32 next_sibling{-1};
  i32 last_sibling{-1};
  i32 level{0};
};

struct NodeTransform {
  vec3 translation{0.f};
  quat rotation{glm::identity<glm::quat>()};
  vec3 scale{1.f};
  void to_mat4(mat4& out) const;
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

struct SkinData {
  std::string name;
  std::vector<u32> joint_node_indices;
  std::vector<mat4> inverse_bind_matrices;
  u32 model_bone_mat_start_i{};
  u32 skeleton_i{};
};

struct Scene2 {
  std::vector<mat4> local_transforms;
  std::vector<NodeTransform> node_transforms;
  std::vector<mat4> global_transforms;
  std::vector<Hierarchy> hierarchies;
  std::vector<std::string> node_names;
  // TODO: vector here
  std::unordered_map<i32, i32> node_to_node_name_idx;
  std::vector<i32> node_mesh_indices;

  enum NodeFlags : u8 {
    NodeFlag_IsJointBit = 1 << 0,
  };

  std::vector<u32> node_flags;
  std::vector<MeshData> mesh_datas;
  static constexpr int max_node_depth{25};
  std::vector<u32> changed_this_frame[max_node_depth];
  std::vector<SkinData> skins;
};

void validate_hierarchy(Scene2& scene);
bool decompose_matrix(const glm::mat4& m, glm::vec3& pos, glm::quat& rot, glm::vec3& scale);
void mark_changed(Scene2& scene, int node);
bool recalc_global_transforms(Scene2& scene, std::vector<i32>* changed_nodes = nullptr);

}  // namespace gfx
