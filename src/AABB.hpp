#pragma once

#include <span>

#include "Common.hpp"

struct AABB {
  vec3 min;
  vec3 max;

  void get_corners(std::span<vec3> corners) const {
    corners[0] = vec3{min.x, min.y, min.z};
    corners[1] = vec3{min.x, min.y, max.z};
    corners[2] = vec3{min.x, max.y, min.z};
    corners[3] = vec3{min.x, max.y, max.z};
    corners[4] = vec3{max.x, min.y, min.z};
    corners[5] = vec3{max.x, min.y, max.z};
    corners[6] = vec3{max.x, max.y, max.z};
    corners[7] = vec3{max.x, max.y, min.z};
  }
  [[nodiscard]] vec3 get_min(std::span<vec3> corners) const {
    vec3 m{std::numeric_limits<float>::max()};
    for (auto c : corners) {
      m = glm::min(c, m);
    }
    return m;
  }
  [[nodiscard]] vec3 get_max(std::span<vec3> corners) const {
    vec3 m{std::numeric_limits<float>::lowest()};
    for (auto c : corners) {
      m = glm::max(c, m);
    }
    return m;
  }
};
