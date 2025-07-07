#pragma once

#include "Common.hpp"
#include "glm/gtc/epsilon.hpp"

namespace util::math {

inline bool is_identity(const mat4& mat, float epsilon = 0.0001f) {
  glm::mat4 identity(1.0f);
  for (int i = 0; i < 4; ++i) {
    if (!glm::all(glm::epsilonEqual(mat[i], identity[i], epsilon))) {
      return false;
    }
  }
  return true;
}

}  // namespace util::math
