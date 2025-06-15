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
  bool operator==(const ObjectData& other) const {
    return model == other.model && aabb_min == other.aabb_min && aabb_max == other.aabb_max;
  }
};

struct Vertex {
  vec3 pos;
  float uv_x;
  vec3 normal;
  float uv_y;
  vec4 tangent;
};

enum class AnimationPath : u8 {
  Translation = 1,
  Rotation = 2,
  Scale = 3,
  Weights = 4,
};

struct AnimSampler {
  std::vector<float> inputs;
  std::vector<float> outputs_raw;
};

struct Channels {
  std::vector<int> nodes;
  std::vector<u32> sampler_indices;
  std::vector<AnimationPath> anim_paths;
};

struct Animation {
  Channels channels;
  [[nodiscard]] int get_channel_node(u32 channel_i) const { return channels.nodes[channel_i]; }
  [[nodiscard]] u32 get_channel_sampler_i(u32 channel_i) const {
    return channels.sampler_indices[channel_i];
  }
  [[nodiscard]] AnimationPath get_channel_anim_path(u32 channel_i) const {
    return channels.anim_paths[channel_i];
  }
  std::vector<AnimSampler> samplers;
  std::string name{"Animation"};
  float ticks_per_second{1.f};
  float duration{0.};
};

struct AnimationState {
  u32 anim_id = {UINT32_MAX};
  float curr_t = {0.f};
  bool play_once{false};
  bool active{true};
};

inline constexpr u32 max_bones_per_vertex{4};

struct AnimatedVertex {
  vec3 pos;
  u32 instance_i;
  vec4 normal;
  vec4 tangent;
  u32 bone_id[max_bones_per_vertex]{~0u, ~0u, ~0u, ~0u};
  float weights[max_bones_per_vertex]{};
};

struct PrimitiveDrawInfo {
  AABB aabb;
  u32 first_index;
  u32 index_count;
  u32 first_vertex;
  u32 vertex_count;
  u32 mesh_idx;
  u32 first_animated_vertex;
};

struct LoadedSceneData {
  Scene2 scene_graph_data;
  std::vector<Material> materials;
  std::vector<Holder<ImageHandle>> textures;
  std::vector<PrimitiveDrawInfo> mesh_draw_infos;
  std::vector<Vertex> vertices;
  std::vector<AnimatedVertex> animated_vertices;
  std::vector<u32> indices;
  std::vector<Animation> animations;
};

struct LoadedSceneBaseData {
  Scene2 scene_graph_data;
  std::vector<PrimitiveDrawInfo> mesh_draw_infos;
  std::vector<Vertex> vertices;
  std::vector<AnimatedVertex> animated_vertices;
  std::vector<u32> indices;
  std::vector<Holder<ImageHandle>> textures;
  std::vector<Material> materials;
  std::vector<Animation> animations;
  bool has_bones{};
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
