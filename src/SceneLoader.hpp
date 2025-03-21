#pragma once

#include <filesystem>
#include <optional>
#include <vector>

#include "Common.hpp"
#include "Scene.hpp"
#include "vk2/Buffer.hpp"
#include "vk2/SamplerCache.hpp"
#include "vk2/Texture.hpp"
namespace gfx {

struct Vertex {
  vec3 pos;
  float uv_x;
  vec3 normal;
  float uv_y;
  // vec2 uv;
};

struct Box3D {
  vec3 min;
  vec3 max;
};

struct Material {
  u32 albedo_idx;
  u32 normal_idx;
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
  SceneLoadData scene_graph_data;
  std::vector<vk2::Sampler> samplers;
  std::vector<Material> materials;
  std::vector<vk2::Texture> textures;
  std::vector<PrimitiveDrawInfo> mesh_draw_infos;
  vk2::Buffer* vert_idx_staging;
  u64 vertices_size;
  u64 indices_size;
};

struct LoadedSceneBaseData {
  SceneLoadData scene_graph_data;
  std::vector<PrimitiveDrawInfo> mesh_draw_infos;
  std::vector<Vertex> vertices;
  std::vector<u32> indices;
  std::vector<vk2::Texture> textures;
  std::vector<Material> materials;
  std::vector<vk2::Sampler> samplers;
};

std::optional<LoadedSceneBaseData> load_gltf_base(const std::filesystem::path& path);

std::optional<LoadedSceneData> load_gltf(const std::filesystem::path& path);

}  // namespace gfx
