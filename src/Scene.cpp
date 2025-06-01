#include "Scene.hpp"

#include <tracy/Tracy.hpp>

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

bool recalc_global_transforms(Scene2& scene, std::vector<i32>* changed_nodes) {
  ZoneScoped;
  // root node
  bool dirty = false;
  if (scene.changed_this_frame[0].size() > 0) {
    int changed_node = scene.changed_this_frame[0][0];
    scene.global_transforms[changed_node] = scene.local_transforms[changed_node];
    scene.changed_this_frame[0].clear();
    if (changed_nodes) {
      changed_nodes->emplace_back(changed_node);
    }
    dirty = true;
  }

  for (int level = 1; level < Scene2::max_node_depth; level++) {
    if (scene.changed_this_frame[level].empty()) continue;
    for (auto changed_node : scene.changed_this_frame[level]) {
      if (changed_nodes) {
        changed_nodes->emplace_back(changed_node);
      }
      int parent = scene.hierarchies[changed_node].parent;
      scene.global_transforms[changed_node] =
          scene.global_transforms[parent] * scene.local_transforms[changed_node];
      dirty = true;
    }
    scene.changed_this_frame[level].clear();
  }
  return dirty;
}
}  // namespace gfx
