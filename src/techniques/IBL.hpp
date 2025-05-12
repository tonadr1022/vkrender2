#pragma once

#include <filesystem>
#include <vector>

#include "Types.hpp"
#include "vk2/Pool.hpp"
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
  void make_cubemap_views_all_mips(ImageHandle handle, std::vector<i32>& views);
  void equirect_to_cube(CmdEncoder& ctx);
  void convolute_cube(CmdEncoder& cmd);
  void prefilter_env_map(CmdEncoder& ctx);

 public:
  Holder<ImageHandle> env_equirect_tex_;
  Holder<ImageHandle> env_cubemap_tex_;
  Holder<ImageHandle> irradiance_cubemap_tex_;
  Holder<ImageHandle> prefiltered_env_map_tex_;
  std::array<Holder<ImageViewHandle>, 6> prefiltered_env_tex_views_mips_;
  std::vector<i32> prefiltered_env_map_tex_views_;
  Holder<ImageHandle> brdf_lut_;

 private:
  Device* device_{};
  std::array<Holder<ImageViewHandle>, 6> cubemap_tex_views_;
  std::array<Holder<ImageViewHandle>, 6> convoluted_cubemap_tex_views_;
  PipelineHandle equirect_to_cube_pipeline2_;
  PipelineHandle convolute_cube_pipeline_;
  PipelineHandle integrate_brdf_pipeline_;
  PipelineHandle convolute_cube_raster_pipeline_;
  PipelineHandle prefilter_env_map_pipeline_;
  BufferHandle cube_vertex_buf_;
  SamplerHandle linear_sampler_;
};

}  // namespace gfx
