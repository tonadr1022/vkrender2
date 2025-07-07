#pragma once

#include <vector>

#include "Common.hpp"
#include "Types.hpp"
#include "core/FixedVector.hpp"

// TODO: no namespace here
namespace gfx {

struct AnimationState {
  u32 anim_id = {UINT32_MAX};
  float curr_t = {0.f};
  bool play_once{false};
  bool active{true};
};

}  // namespace gfx

struct InstanceAnimation;
using AnimationHandle = GenerationalHandle<InstanceAnimation>;

struct BlendTreeNode {
  enum class Type : u8 { Clip, Lerp };
  util::fixed_vector<u32, 8> children;
  u32 weight_idx{UINT32_MAX};
  u32 animation_i{UINT32_MAX};
  Type type;
};

struct BlendTree {
  std::vector<BlendTreeNode> blend_tree_nodes;
  std::unordered_map<std::string, u32> name_to_blend_tree_node;
  std::vector<float> control_vars;
  AnimationHandle animation_id;
  u32 root_node_{invalid_node};
  static constexpr u32 invalid_node = UINT32_MAX;

  void reserve_nodes(u32 node_count) { blend_tree_nodes.reserve(node_count); }

  u32 get_blend_node_idx(const std::string& name);
  [[nodiscard]] BlendTreeNode* get_root_node() { return &blend_tree_nodes[root_node_]; }

  BlendTreeNode* get_blend_node(const std::string& name);
  BlendTreeNode* get_blend_node(u32 idx) {
    assert(idx < blend_tree_nodes.size());
    if (idx >= blend_tree_nodes.size()) return nullptr;
    return &blend_tree_nodes[idx];
  }

  // returns node idx of added node
  u32 add_clip_node(const std::string& name, const std::string& anim_name);

  void add_lerp_node(const std::string& name, const std::string& child_a,
                     const std::string& child_b);
  void set_root_node(const std::string& name);
  void set_control_var(u32 idx, float value) {
    assert(idx < control_vars.size());
    control_vars[idx] = value;
  }
};

struct InstanceAnimation {
  std::vector<gfx::AnimationState> states;
  std::vector<bool> dirty_anim_nodes;
  std::unordered_map<std::string, u32> anim_name_to_idx;
  BlendTree blend_tree;

  gfx::AnimationState* get_state(const std::string& name);

  void set_blend_state(const std::string& name, float weight) {
    auto& tree = blend_tree;
    auto* node = tree.get_blend_node(name);
    assert(node);
    tree.set_control_var(node->weight_idx, weight);
  }

  // void set_target(const std::string& blend_node_name, float weight) {
  //   auto& tree = blend_tree;
  //   u32 node_idx = tree.get_blend_node_idx(blend_node_name);
  //   assert(node_idx != BlendTree::invalid_node);
  //   const auto& node = tree.blend_tree_nodes[node_idx];
  //   assert(node.weight_idx != BlendTree::invalid_node);
  //   tree.set_control_var(node.weight_idx, weight);
  // }

  void resize(int node_count) { dirty_anim_nodes.resize(node_count); }
};
