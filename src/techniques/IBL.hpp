#pragma once

#include <filesystem>
#include <optional>
#include <vector>

#include "Types.hpp"
#include "vk2/Texture.hpp"

namespace gfx {

class PipelineLoader;
class Device;
struct CmdEncoder;

class IBL {
 public:
  explicit IBL(Device* device, BufferHandle cube_vertex_buf);
  void load_pipelines(PipelineLoader& loader);
  void load_env_map(CmdEncoder& ctx, const std::filesystem::path& path);
  void init_post_pipeline_load();

 private:
  void make_cubemap_views_all_mips(const Image& texture,
                                   std::vector<std::optional<ImageView>>& views);
  void equirect_to_cube(CmdEncoder& ctx);
  void convolute_cube(CmdEncoder& ctx);
  void prefilter_env_map(CmdEncoder& ctx);

 public:
  std::optional<Image> env_equirect_tex_;
  std::optional<Image> env_cubemap_tex_;
  std::optional<Image> irradiance_cubemap_tex_;
  std::optional<TextureCubeAndViews> prefiltered_env_map_tex_;
  std::optional<Image> brdf_lut_;
  std::vector<std::optional<ImageView>> prefiltered_env_tex_views_;

 private:
  Device* device_{};
  std::array<std::optional<ImageView>, 6> cubemap_tex_views_;
  std::array<std::optional<ImageView>, 6> convoluted_cubemap_tex_views_;
  PipelineHandle equirect_to_cube_pipeline2_;
  PipelineHandle convolute_cube_pipeline_;
  PipelineHandle integrate_brdf_pipeline_;
  PipelineHandle convolute_cube_raster_pipeline_;
  PipelineHandle prefilter_env_map_pipeline_;
  BufferHandle cube_vertex_buf_;
  SamplerHandle linear_sampler_;
};

}  // namespace gfx
