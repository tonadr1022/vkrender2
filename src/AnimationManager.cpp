#include "AnimationManager.hpp"

#include <cassert>

#include "ResourceManager.hpp"
#include "core/Logger.hpp"

void AnimationManager::init() {
  assert(!g_instance);
  g_instance = new AnimationManager;
}

void AnimationManager::shutdown() {}

void BlendTree::add_lerp_node(const std::string& name, const std::string& child_a,
                              const std::string& child_b) {
  u32 clip_node_1_idx = get_blend_node_idx(child_a);
  u32 clip_node_2_idx = get_blend_node_idx(child_b);
  assert(clip_node_1_idx != invalid_node);
  assert(clip_node_2_idx != invalid_node);

  size_t node_i = blend_tree_nodes.size();
  BlendTreeNode node{};
  node.children = {clip_node_1_idx, clip_node_2_idx};
  node.type = BlendTreeNode::Type::Lerp;
  node.weight_idx = control_vars.size();
  control_vars.emplace_back(0.f);
  name_to_blend_tree_node.emplace(name, node_i);
  blend_tree_nodes.emplace_back(node);
}

u32 BlendTree::add_clip_node(const std::string& name, const std::string& anim_name) {
  auto* animation = AnimationManager::get().get_animation(animation_id);
  assert(animation);
  auto it = animation->anim_name_to_idx.find(anim_name);
  assert(it != animation->anim_name_to_idx.end());
  if (it == animation->anim_name_to_idx.end()) {
    return invalid_node;
  }

  u32 idx = it->second;
  size_t node_i = blend_tree_nodes.size();
  BlendTreeNode node{};
  assert(node.children.empty());
  node.animation_i = idx;
  node.type = BlendTreeNode::Type::Clip;
  name_to_blend_tree_node.emplace(name, node_i);
  blend_tree_nodes.emplace_back(node);
  return node_i;
}

u32 BlendTree::get_blend_node_idx(const std::string& name) {
  auto it = name_to_blend_tree_node.find(name);
  if (it != name_to_blend_tree_node.end()) {
    return it->second;
  }
  return invalid_node;
}

AnimationHandle AnimationManager::add_animation(LoadedInstanceData& instance,
                                                const LoadedModelData& model) {
  AnimationHandle handle = instance_animations_.alloc();
  auto* animation = instance_animations_.get(handle);
  animation->blend_tree.animation_id = handle;
  auto& scene_graph_data = instance.scene_graph_data;
  size_t num_nodes = scene_graph_data.hierarchies.size();
  animation->dirty_anim_nodes.resize(num_nodes);
  animation->states.resize(model.animations.size());
  for (size_t i = 0; i < model.animations.size(); i++) {
    animation->anim_name_to_idx.emplace(model.animations[i].name, i);
    animation->states[i].anim_id = i;
  }

  return handle;
}
void AnimationManager::evaluate_blend_tree(LoadedInstanceData& instance,
                                           const InstanceAnimation& animation,
                                           const std::vector<gfx::Animation>& animations,
                                           std::vector<gfx::NodeTransformAccumulator>& out_accum,
                                           float weight, const BlendTreeNode& node) {
  if (node.type == BlendTreeNode::Type::Clip) {
    assert(node.animation_i < animations.size());
    assert(node.children.empty());
    apply_clip(animations[node.animation_i], animation.states[node.animation_i], weight, out_accum,
               instance.dirty_animation_node_bits);
    return;
  }

  if (node.type == BlendTreeNode::Type::Lerp) {
    assert(node.children.size() == 2);
    const auto& left = animation.blend_tree.blend_tree_nodes[node.children[0]];
    const auto& right = animation.blend_tree.blend_tree_nodes[node.children[1]];

    float left_weight = animation.blend_tree.control_vars[node.weight_idx];
    // LINFO("weight: {} {}", left_weight, node.weight_idx);
    float right_weight = 1.0f - left_weight;
    std::vector<gfx::NodeTransformAccumulator> left_accum(instance.transform_accumulators.size());
    std::vector<gfx::NodeTransformAccumulator> right_accum(instance.transform_accumulators.size());
    evaluate_blend_tree(instance, animation, animations, left_accum, left_weight, left);
    evaluate_blend_tree(instance, animation, animations, right_accum, right_weight, right);

    auto get_translation = [](const gfx::NodeTransformAccumulator& accum,
                              const gfx::NodeTransform& nt) {
      return accum.weights.x > 0.f ? accum.translation / accum.weights.x : nt.translation;
    };
    auto get_rotation = [](const gfx::NodeTransformAccumulator& accum,
                           const gfx::NodeTransform& nt) {
      return accum.weights.y > 0.f ? glm::normalize(accum.rotation) : nt.rotation;
    };
    auto get_scale = [](const gfx::NodeTransformAccumulator& accum, const gfx::NodeTransform& nt) {
      return accum.weights.z > 0.f ? accum.scale / accum.weights.z : nt.scale;
    };
    for (size_t i = 0; i < left_accum.size(); i++) {
      // mix transforms
      const auto& left_t = left_accum[i];
      const auto& right_t = right_accum[i];
      const auto& nt = instance.scene_graph_data.node_transforms[i];
      vec3 translation =
          glm::mix(get_translation(left_t, nt), get_translation(right_t, nt), left_weight);
      quat rot;
      quat left_rot = get_rotation(left_t, nt);
      quat right_rot = get_rotation(right_t, nt);
      if (glm::dot(left_rot, right_rot) < 0.f) {
        rot = glm::slerp(left_rot, -right_rot, left_weight);
      } else {
        rot = glm::slerp(left_rot, right_rot, left_weight);
      }
      vec3 scale = glm::mix(get_scale(left_t, nt), get_scale(right_t, nt), left_weight);
      out_accum[i].translation = translation;
      out_accum[i].rotation = rot;
      out_accum[i].scale = scale;
      out_accum[i].weights = vec3{1.f};
    }
  }
}

namespace {

float get_interpolation_value(float start_anim_t, float end_anim_t, float curr_t) {
  return (curr_t - start_anim_t) / (end_anim_t - start_anim_t);
}

}  // namespace

void AnimationManager::apply_clip(const gfx::Animation& animation, const gfx::AnimationState& state,
                                  float weight,
                                  std::span<gfx::NodeTransformAccumulator> transform_accumulators,
                                  std::vector<bool>& dirty_node_bits) {
  if (!state.active) {
    return;
  }
  assert(transform_accumulators.size() > 0);
  for (size_t channel_i = 0; channel_i < animation.channels.nodes.size(); channel_i++) {
    int channel_node_i = animation.get_channel_node(channel_i);
    assert(channel_node_i >= 0);
    u32 channel_sampler_i = animation.get_channel_sampler_i(channel_i);
    assert(channel_sampler_i < animation.samplers.size());
    gfx::AnimationPath channel_anim_path = animation.get_channel_anim_path(channel_i);
    const gfx::AnimSampler& sampler = animation.samplers[channel_sampler_i];
    uvec2 time_indices = sampler.get_time_indices(state.curr_t);
    u32 time_i = time_indices.x, next_time_i = time_indices.y;
    float interpolation_val = get_interpolation_value(sampler.inputs[time_indices.x],
                                                      sampler.inputs[time_indices.y], state.curr_t);

    assert(channel_node_i < (int)transform_accumulators.size());
    auto& nt = transform_accumulators[channel_node_i];
    dirty_node_bits[channel_node_i] = true;
    assert(sampler.inputs.size() > 1);
    if (channel_anim_path == gfx::AnimationPath::Translation) {
      assert(sampler.outputs_raw.size() == sampler.inputs.size() * 3);
      const vec3* translations = reinterpret_cast<const vec3*>(sampler.outputs_raw.data());
      auto translation =
          sampler.inputs.size() == 1
              ? translations[0]
              : glm::mix(translations[time_i], translations[next_time_i], interpolation_val);
      nt.translation += translation * weight;
      nt.weights.x += weight;
    } else if (channel_anim_path == gfx::AnimationPath::Rotation) {
      assert(sampler.outputs_raw.size() == sampler.inputs.size() * 4);
      assert(time_i < sampler.inputs.size());
      const quat* rotations = reinterpret_cast<const quat*>(sampler.outputs_raw.data());
      quat rotation;
      if (sampler.inputs.size() == 1) {
        rotation = rotations[0];
      } else {
        quat q0 = rotations[time_i];
        quat q1 = rotations[next_time_i];
        if (glm::dot(q0, q1) < 0.0f) q1 = -q1;  // ensure shortest path
        rotation = glm::slerp(q0, q1, interpolation_val);
      }

      if (nt.weights.y == 0.0f) {
        nt.rotation = rotation;
      } else {
        nt.rotation = glm::slerp(nt.rotation, rotation, weight / (nt.weights.y + weight));
      }
      nt.weights.y += weight;
    } else if (channel_anim_path == gfx::AnimationPath::Scale) {
      assert(sampler.outputs_raw.size() == sampler.inputs.size() * 3);
      const vec3* scales = reinterpret_cast<const vec3*>(sampler.outputs_raw.data());
      auto scale = sampler.inputs.size() == 1
                       ? scales[0]
                       : glm::mix(scales[time_i], scales[next_time_i], interpolation_val);
      nt.scale += weight * scale;
      nt.weights.z += weight;
    }
  }
}

void BlendTree::set_root_node(const std::string& name) {
  u32 node = get_blend_node_idx(name);
  assert(node);
  root_node_ = node;
}

BlendTreeNode* BlendTree::get_blend_node(const std::string& name) {
  u32 idx = get_blend_node_idx(name);
  assert(idx != invalid_node);
  if (idx == invalid_node) {
    return nullptr;
  }
  return &blend_tree_nodes[idx];
}
gfx::AnimationState* InstanceAnimation::get_state(const std::string& name) {
  auto it = anim_name_to_idx.find(name);
  if (it != anim_name_to_idx.end()) {
    return &states[it->second];
  }
  return nullptr;
}
