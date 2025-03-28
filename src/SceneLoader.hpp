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
};

struct Box3D {
  vec3 min;
  vec3 max;
};

struct Material {
  vec4 emissive_factors;
  uvec4 ids1;  // albedo, normal, metal_rough, emissive
  uvec4 ids2;  // ao
};

struct PrimitiveDrawInfo {
  u32 first_index;
  u32 index_count;
  u32 first_vertex;
  u32 vertex_count;
  u32 mesh_idx;
};

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
struct DefaultMaterialData {
  u32 white_img_handle;
};

std::optional<LoadedSceneBaseData> load_gltf_base(const std::filesystem::path& path,
                                                  const DefaultMaterialData& default_mat);

std::optional<LoadedSceneData> load_gltf(const std::filesystem::path& path,
                                         const DefaultMaterialData& default_mat);

}  // namespace gfx
