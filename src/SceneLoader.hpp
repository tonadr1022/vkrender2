#pragma once

#include <filesystem>
#include <optional>
#include <vector>

#include "AABB.hpp"
#include "Common.hpp"
#include "Scene.hpp"
#include "Types.hpp"
#include "vk2/Pool.hpp"

namespace gfx {

struct ObjectData {
  mat4 model;
  // TODO: better padding
  vec4 aabb_min;
  vec4 aabb_max;
};

struct Vertex {
  vec3 pos;
  float uv_x;
  vec3 normal;
  float uv_y;
  vec4 tangent;
};

struct PrimitiveDrawInfo {
  AABB aabb;
  u32 first_index;
  u32 index_count;
  u32 first_vertex;
  u32 vertex_count;
  u32 mesh_idx;
};

struct LoadedSceneData {
  Scene2 scene_graph_data;
  std::vector<Material> materials;
  std::vector<Holder<ImageHandle>> textures;
  std::vector<PrimitiveDrawInfo> mesh_draw_infos;
  std::vector<Vertex> vertices;
  std::vector<u32> indices;
};

struct LoadedSceneBaseData {
  Scene2 scene_graph_data;
  // SceneLoadData scene_graph_data;
  std::vector<PrimitiveDrawInfo> mesh_draw_infos;
  std::vector<Vertex> vertices;
  std::vector<u32> indices;
  std::vector<Holder<ImageHandle>> textures;
  std::vector<Material> materials;
};

struct DefaultMaterialData {
  u32 white_img_handle;
};

std::optional<LoadedSceneBaseData> load_gltf_base(const std::filesystem::path& path,
                                                  const DefaultMaterialData& default_mat);

std::optional<LoadedSceneData> load_gltf(const std::filesystem::path& path,
                                         const DefaultMaterialData& default_mat);
struct CPUHDRImageData {
  u32 w, h, channels;
  float* data{};
};
namespace loader {

std::optional<CPUHDRImageData> load_hdr(const std::filesystem::path& path, int num_components = 4,
                                        bool flip = false);
void free_hdr(CPUHDRImageData& img_data);

}  // namespace loader

}  // namespace gfx
