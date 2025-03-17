#pragma once

#include <filesystem>
#include <optional>
#include <vector>

#include "Common.hpp"
#include "vk2/Buffer.hpp"
#include "vk2/Texture.hpp"
namespace gfx {

struct Vertex {
  vec3 pos;
  float uv_x;
  vec3 normal;
  float uv_y;
};

struct LoadedSceneData {
  vk2::Buffer vertex_buffer;
  vk2::Buffer index_buffer;
  std::vector<vk2::Sampler> samplers;
};

struct LoadedSceneBaseData {
  std::vector<Vertex> vertices;
  std::vector<u32> indices;
  std::vector<vk2::Sampler> samplers;
};

std::optional<LoadedSceneBaseData> load_gltf_base(const std::filesystem::path& path);

std::optional<LoadedSceneData> load_gltf(const std::filesystem::path& path);

}  // namespace gfx
