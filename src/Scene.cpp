#include "Scene.hpp"

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/matrix_decompose.hpp>
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
  const int level = scene.hierarchies[node].level;
  assert(level >= 0 && level < Scene2::max_node_depth);
  assert(node >= 0 && (size_t)node < scene.hierarchies.size());
  scene.changed_this_frame[level].push_back(node);
  for (int s = scene.hierarchies[node].first_child; s != -1;
       s = scene.hierarchies[s].next_sibling) {
    mark_changed(scene, s);
  }
}

void validate_hierarchy(Scene2& scene) {
  for (size_t i = 0; i < scene.hierarchies.size(); i++) {
    const auto& hier = scene.hierarchies[i];
    if (hier.parent != -1) {
      // Check that parent's level is one less than child's level
      assert(scene.hierarchies[hier.parent].level == hier.level - 1);
    }
    // Check children point back to correct parent
    for (int child = hier.first_child; child != -1; child = scene.hierarchies[child].next_sibling) {
      assert(scene.hierarchies[child].parent == (int)i);
    }
  }
}

bool recalc_global_transforms(Scene2& scene, std::vector<i32>* changed_nodes) {
  ZoneScoped;
  // root node
  bool dirty = false;
  if (scene.changed_this_frame[0].size() > 0) {
    const int changed_node = scene.changed_this_frame[0][0];
    scene.global_transforms[changed_node] = scene.local_transforms[changed_node];
    scene.changed_this_frame[0].clear();
    if (changed_nodes) {
      changed_nodes->emplace_back(changed_node);
    }
    dirty = true;
  }

  for (int level = 1; level < Scene2::max_node_depth; level++) {
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

bool decompose_matrix(const glm::mat4& m, glm::vec3& pos, glm::quat& rot, glm::vec3& scale) {
  glm::vec3 skew;
  glm::vec4 perspective;
  return glm::decompose(m, scale, rot, pos, skew, perspective);
}
}  // namespace gfx
