#pragma once

#include <filesystem>
#include <optional>
#include <vector>

#include "Common.hpp"
#include "Scene.hpp"
#include "vk2/Buffer.hpp"
#include "vk2/Texture.hpp"
namespace gfx {

struct Vertex {
  vec3 pos;
  float uv_x;
  vec3 normal;
  float uv_y;
};

struct Box3D {
  vec3 min;
  vec3 max;
};

struct PrimitiveDrawInfo {
  u32 first_index;
  u32 index_count;
  u32 first_vertex;
  u32 vertex_count;
  u32 mesh_idx;
};

// struct RawMesh {
//   std::vector<Vertex> vertices;
//   std::vector<u32> indices;
//   // Box3D bbox;
// };

struct LoadedSceneData {
  SceneGraphData scene_graph_data;
  vk2::Buffer vertex_staging;
  vk2::Buffer index_staging;
  std::vector<vk2::Sampler> samplers;
};

struct LoadedSceneBaseData {
  SceneGraphData scene_graph_data;
  std::vector<PrimitiveDrawInfo> primitive_draw_infos;
  std::vector<Vertex> vertices;
  std::vector<u32> indices;
  std::vector<vk2::Sampler> samplers;
};

std::optional<LoadedSceneBaseData> load_gltf_base(const std::filesystem::path& path);

std::optional<LoadedSceneData> load_gltf(const std::filesystem::path& path);

}  // namespace gfx
