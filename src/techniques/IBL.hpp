#pragma once

#include <filesystem>
#include <optional>
#include <vector>

#include "Types.hpp"
#include "vk2/Texture.hpp"

namespace gfx {
struct CmdEncoder;
class IBL {
 public:
  explicit IBL(BufferHandle cube_vertex_buf);
  void load_env_map(CmdEncoder& ctx, const std::filesystem::path& path);

 private:
  void make_cubemap_views_all_mips(const vk2::Image& texture,
                                   std::vector<std::optional<vk2::ImageView>>& views);
  void make_brdf_lut();
  void equirect_to_cube(CmdEncoder& ctx);
  void convolute_cube(CmdEncoder& ctx);
  void prefilter_env_map(CmdEncoder& ctx);

 public:
  std::optional<vk2::Image> env_equirect_tex_;
  std::optional<vk2::Image> env_cubemap_tex_;
  std::optional<vk2::Image> irradiance_cubemap_tex_;
  std::optional<vk2::TextureCubeAndViews> prefiltered_env_map_tex_;
  std::optional<vk2::Image> brdf_lut_;
  std::vector<std::optional<vk2::ImageView>> prefiltered_env_tex_views_;

 private:
  std::array<std::optional<vk2::ImageView>, 6> cubemap_tex_views_;
  std::array<std::optional<vk2::ImageView>, 6> convoluted_cubemap_tex_views_;
  // vk2::PipelineHandle equirect_to_cube_pipeline_;
  PipelineHandle equirect_to_cube_pipeline2_;
  PipelineHandle convolute_cube_pipeline_;
  PipelineHandle integrate_brdf_pipeline_;
  PipelineHandle convolute_cube_raster_pipeline_;
  PipelineHandle prefilter_env_map_pipeline_;
  BufferHandle cube_vertex_buf_;
};

}  // namespace gfx
