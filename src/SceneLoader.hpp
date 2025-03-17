#pragma once

#include <filesystem>
#include <optional>
#include <vector>

#include "Common.hpp"
namespace gfx {

struct Vertex {
  vec3 pos;
  float uv_x;
  vec3 normal;
  float uv_y;
};

struct LoadedSceneData {
  std::vector<Vertex> vertices;
  std::vector<u32> indices;
};

std::optional<LoadedSceneData> load_gltf(const std::filesystem::path& path);

}  // namespace gfx
