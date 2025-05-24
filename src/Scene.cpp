#include "Scene.hpp"

#include "shaders/common.h.glsl"

namespace gfx {

PassFlags Material::get_pass_flags() const {
  PassFlags flags{};
  if (ids2.w & MATERIAL_ALPHA_MODE_MASK_BIT) {
    flags |= PassFlags_OpaqueAlpha;
  } else if (ids2.w & MATERIAL_TRANSPARENT_BIT) {
    flags |= PassFlags_Transparent;
  } else {
    flags |= PassFlags_Opaque;
  }
  return flags;
}

bool Material::is_double_sided() const { return (ids2.w & MATERIAL_DOUBLE_SIDED_BIT); }

void mark_changed(Scene2& scene, int node) {
  scene.changed_this_frame[scene.hierarchies[node].level].emplace_back(node);
  for (int sibling = scene.hierarchies[node].first_child; sibling != -1;
       sibling = scene.hierarchies[sibling].next_sibling) {
    mark_changed(scene, sibling);
  }
}

void recalc_global_transforms(Scene2& scene) {
  // root node
  if (scene.changed_this_frame[0].size() > 0) {
    int changed_node = scene.changed_this_frame[0][0];
    scene.global_transforms[changed_node] = scene.local_transforms[changed_node];
    scene.changed_this_frame[0].clear();
  }

  for (int level = 1; level < Scene2::max_node_depth && scene.changed_this_frame[level].size() > 0;
       level++) {
    for (auto changed_node : scene.changed_this_frame[level]) {
      int parent = scene.hierarchies[changed_node].parent;
      scene.global_transforms[changed_node] =
          scene.global_transforms[parent] * scene.local_transforms[changed_node];
    }
    scene.changed_this_frame[level].clear();
  }
}
}  // namespace gfx
