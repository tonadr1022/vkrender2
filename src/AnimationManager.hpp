#pragma once

#include <span>

#include "Animation.hpp"
#include "vk2/Pool.hpp"

struct LoadedInstanceData;
struct LoadedModelData;
namespace gfx {
struct NodeTransformAccumulator;
struct Animation;
}  // namespace gfx

class AnimationManager {
 public:
  static AnimationManager& get() { return *g_instance; }
  static void init();
  static void shutdown();

  AnimationHandle add_animation(LoadedInstanceData& instance, const LoadedModelData& model);
  InstanceAnimation* get_animation(AnimationHandle handle) {
    return instance_animations_.get(handle);
  }

  // TODO: don't actually destroy the object, just "reset it" so
  // new animations don't need to realloc memory
  void remove_animation(AnimationHandle handle) { instance_animations_.destroy(handle); }
  void evaluate_blend_tree(LoadedInstanceData& instance, const InstanceAnimation& animation,
                           const std::vector<gfx::Animation>& animations,
                           std::vector<gfx::NodeTransformAccumulator>& out_accum, float weight,
                           const BlendTreeNode& node);
  void apply_clip(const gfx::Animation& animation, const gfx::AnimationState& state, float weight,
                  std::span<gfx::NodeTransformAccumulator> transform_accumulators,
                  std::vector<bool>& dirty_node_bits);

 private:
  AnimationManager() = default;
  inline static AnimationManager* g_instance{};

  Pool<AnimationHandle, InstanceAnimation> instance_animations_;
};
