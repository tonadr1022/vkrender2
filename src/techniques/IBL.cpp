#include "IBL.hpp"

#include <volk.h>
#include <vulkan/vulkan_core.h>

#include "CommandEncoder.hpp"
#include "StateTracker.hpp"
#include "Types.hpp"
#include "VkRender2.hpp"
#include "shaders/ibl/eq_to_cube_comp_common.h.glsl"
#include "vk2/Device.hpp"
#include "vk2/Initializers.hpp"
#include "vk2/ShaderCompiler.hpp"

namespace {

const mat4 PROJ = glm::perspective(glm::radians(90.f), 1.f, .1f, 512.f);
const std::array<glm::mat4, 6> VIEW_MATRICES = {
    glm::rotate(glm::rotate(glm::mat4(1.0f), glm::radians(90.0f), glm::vec3(0.0f, 1.0f, 0.0f)),
                glm::radians(180.0f), glm::vec3(1.0f, 0.0f, 0.0f)),
    glm::rotate(glm::rotate(glm::mat4(1.0f), glm::radians(-90.0f), glm::vec3(0.0f, 1.0f, 0.0f)),
                glm::radians(180.0f), glm::vec3(1.0f, 0.0f, 0.0f)),
    glm::rotate(glm::mat4(1.0f), glm::radians(-90.0f), glm::vec3(1.0f, 0.0f, 0.0f)),
    glm::rotate(glm::mat4(1.0f), glm::radians(90.0f), glm::vec3(1.0f, 0.0f, 0.0f)),
    glm::rotate(glm::mat4(1.0f), glm::radians(180.0f), glm::vec3(1.0f, 0.0f, 0.0f)),
    glm::rotate(glm::mat4(1.0f), glm::radians(180.0f), glm::vec3(0.0f, 0.0f, 1.0f)),
};

}  // namespace

namespace gfx {

void IBL::load_env_map(CmdEncoder& ctx, const std::filesystem::path& path) {
  env_equirect_tex_ = VkRender2::get().load_hdr_img(ctx, path);
  equirect_to_cube(ctx);
  convolute_cube(ctx);
  prefilter_env_map(ctx);
}

IBL::IBL(Device* device, BufferHandle cube_vertex_buf)
    : device_(device), cube_vertex_buf_(cube_vertex_buf) {
  u32 skybox_res = 1024;
  u32 convoluted_res = 32;
  irradiance_cubemap_tex_ = Image{ImageCreateInfo{.view_type = VK_IMAGE_VIEW_TYPE_CUBE,
                                                  .format = VK_FORMAT_R16G16B16A16_SFLOAT,
                                                  .extent = {convoluted_res, convoluted_res, 1},
                                                  .mip_levels = 1,
                                                  .array_layers = 6,
                                                  .usage = ImageUsage::General}};
  env_cubemap_tex_ = Image{ImageCreateInfo{.view_type = VK_IMAGE_VIEW_TYPE_CUBE,
                                           .format = VK_FORMAT_R16G16B16A16_SFLOAT,
                                           .extent = {skybox_res, skybox_res, 1},
                                           .mip_levels = get_mip_levels(uvec2{skybox_res}),
                                           .array_layers = 6,
                                           .usage = ImageUsage::General}};
  u32 prefiltered_env_map_res = 256;
  prefiltered_env_map_tex_ = TextureCubeAndViews{
      ImageCreateInfo{.view_type = VK_IMAGE_VIEW_TYPE_CUBE,
                      .format = VK_FORMAT_R16G16B16A16_SFLOAT,
                      .extent = {prefiltered_env_map_res, prefiltered_env_map_res, 1},
                      .mip_levels = get_mip_levels(uvec2{prefiltered_env_map_res}),
                      .array_layers = 6,
                      .usage = ImageUsage::General}};
  make_cubemap_views_all_mips(prefiltered_env_map_tex_->texture.value(),
                              prefiltered_env_tex_views_);

  auto make_cubemap_views = [](const Image& tex) -> std::array<std::optional<ImageView>, 6> {
    std::array<std::optional<ImageView>, 6> result;
    for (u32 i = 0; i < 6; i++) {
      result[i] = ImageView{tex, ImageViewCreateInfo{
                                     .format = tex.format(),
                                     .range =
                                         {
                                             .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                                             .baseMipLevel = 0,
                                             .levelCount = tex.create_info().mip_levels,
                                             .baseArrayLayer = i,
                                             .layerCount = 1,
                                         },
                                     .view_type = VK_IMAGE_VIEW_TYPE_2D,
                                 }};
    }
    return result;
  };

  cubemap_tex_views_ = make_cubemap_views(env_cubemap_tex_.value());
  convoluted_cubemap_tex_views_ = make_cubemap_views(irradiance_cubemap_tex_.value());
  brdf_lut_ = Image{ImageCreateInfo{.view_type = VK_IMAGE_VIEW_TYPE_2D,
                                    .format = VK_FORMAT_R16G16_SFLOAT,
                                    .extent = {512, 512, 1},
                                    .mip_levels = 1,
                                    .array_layers = 1,
                                    .usage = ImageUsage::General}};
}

void IBL::make_cubemap_views_all_mips(const Image& texture,
                                      std::vector<std::optional<ImageView>>& views) {
  for (u32 mip = 0; mip < texture.create_info().mip_levels; mip++) {
    views.emplace_back(ImageView{texture, ImageViewCreateInfo{
                                              .format = texture.format(),
                                              .range =
                                                  {
                                                      .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                                                      .baseMipLevel = mip,
                                                      .levelCount = 1,
                                                      .baseArrayLayer = 0,
                                                      .layerCount = VK_REMAINING_ARRAY_LAYERS,
                                                  },
                                              .view_type = VK_IMAGE_VIEW_TYPE_CUBE,
                                          }});
  }
}

void IBL::init_post_pipeline_load() {
  if (!linear_sampler_.is_valid()) {
    linear_sampler_ = device_->get_or_create_sampler(SamplerCreateInfo{
        .min_filter = FilterMode::Linear,
        .mag_filter = FilterMode::Linear,
        .mipmap_mode = FilterMode::Linear,
        .address_mode = AddressMode::Repeat,
    });
  }
  VkRender2::get().immediate_submit([this](CmdEncoder& ctx) {
    VkCommandBuffer cmd = ctx.cmd();
    VkRender2::get().bind_bindless_descriptors(ctx);
    StateTracker state;
    state.reset(cmd);
    state
        .transition(*brdf_lut_, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                    VK_ACCESS_2_SHADER_WRITE_BIT, VK_IMAGE_LAYOUT_GENERAL)
        .flush_barriers();
    PipelineManager::get().bind_compute(cmd, integrate_brdf_pipeline_);
    struct {
      u32 tex_idx, sampler_idx;
    } pc{brdf_lut_->view().storage_img_resource().handle,
         device_->get_bindless_idx(device_->get_or_create_sampler(SamplerCreateInfo{
             .min_filter = FilterMode::Linear,
             .mag_filter = FilterMode::Linear,
             .mipmap_mode = FilterMode::Linear,
             .address_mode = AddressMode::Repeat,
         }))};

    ctx.push_constants(sizeof(pc), &pc);
    vkCmdDispatch(cmd, brdf_lut_->extent_2d().width / 16, brdf_lut_->extent_2d().height / 16, 1);
    state
        .transition(*brdf_lut_, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                    VK_ACCESS_2_SHADER_READ_BIT, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
        .flush_barriers();
  });
}

void IBL::equirect_to_cube(CmdEncoder& ctx) {
  VkCommandBuffer cmd = ctx.cmd();
  VkRender2::get().bind_bindless_descriptors(ctx);
  transition_image(cmd, *env_cubemap_tex_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
  PipelineManager::get().bind_compute(cmd, equirect_to_cube_pipeline2_);
  // TODO: define linear sampler idx in the shader

  EquirectToCubeComputePushConstants pc{
      .sampler_idx = device_->get_bindless_idx(
          device_->get_or_create_sampler(SamplerCreateInfo{.min_filter = FilterMode::Linear,
                                                           .mag_filter = FilterMode::Linear,
                                                           .mipmap_mode = FilterMode::Linear,
                                                           .address_mode = AddressMode::Repeat})),
      .tex_idx = env_equirect_tex_->view().sampled_img_resource().handle,
      .out_tex_idx = env_cubemap_tex_->view().storage_img_resource().handle};
  ctx.push_constants(sizeof(pc), &pc);
  ctx.dispatch(env_cubemap_tex_->extent_2d().width / 16, env_cubemap_tex_->extent_2d().height / 16,
               6);
  transition_image(cmd, env_cubemap_tex_.value(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
  VkRender2::get().generate_mipmaps(ctx, *env_cubemap_tex_);
}

void IBL::convolute_cube(CmdEncoder& cmd) {
  transition_image(cmd.cmd(), *env_cubemap_tex_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
  transition_image(cmd.cmd(), *irradiance_cubemap_tex_, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
  for (u32 i = 0; i < 6; i++) {
    uvec2 extent{irradiance_cubemap_tex_->extent_2d().width,
                 irradiance_cubemap_tex_->extent_2d().height};
    cmd.begin_rendering({.extent = extent},
                        {RenderingAttachmentInfo{&convoluted_cubemap_tex_views_[i].value(),
                                                 RenderingAttachmentInfo::Type::Color,
                                                 RenderingAttachmentInfo::LoadOp::Load}});
    cmd.set_viewport_and_scissor(irradiance_cubemap_tex_->extent_2d().width,
                                 irradiance_cubemap_tex_->extent_2d().height);
    PipelineManager::get().bind_graphics(cmd.cmd(), convolute_cube_raster_pipeline_);
    struct {
      mat4 vp;
      u32 in_tex_idx, sampler_idx, vertex_buffer_idx;
    } pc{PROJ * VIEW_MATRICES[i], env_cubemap_tex_->view().sampled_img_resource().handle,
         device_->get_bindless_idx(linear_sampler_),
         get_device().get_buffer(cube_vertex_buf_)->resource_info_->handle};
    cmd.push_constants(sizeof(pc), &pc);
    cmd.set_cull_mode(CullMode::None);
    cmd.draw(36);
    cmd.end_rendering();
  }
}

void IBL::prefilter_env_map(CmdEncoder& ctx) {
  auto* cmd = ctx.cmd();
  transition_image(cmd, *prefiltered_env_map_tex_->texture,
                   VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);

  // make image views
  std::vector<std::array<std::optional<ImageView>, 6>> cube_mip_views;
  u32 mip_levels = prefiltered_env_map_tex_->texture->create_info().mip_levels;
  for (u32 mip = 0; mip < mip_levels; mip++) {
    auto& texture = prefiltered_env_map_tex_->texture.value();
    std::array<std::optional<ImageView>, 6> mip_views;
    cube_mip_views.emplace_back(std::array<std::optional<ImageView>, 6>{});
    for (u32 layer = 0; layer < 6; layer++) {
      cube_mip_views.back()[layer] =
          ImageView{texture, ImageViewCreateInfo{
                                 .format = texture.format(),
                                 .range =
                                     {
                                         .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                                         .baseMipLevel = mip,
                                         .levelCount = 1,
                                         .baseArrayLayer = layer,
                                         .layerCount = 1,
                                     },
                                 .view_type = VK_IMAGE_VIEW_TYPE_2D,
                             }};
    }
  }

  {
    for (u32 mip = 0; mip < mip_levels; mip++) {
      float roughness = (float)mip / (float)(mip_levels - 1);
      for (u32 i = 0; i < 6; i++) {
        u32 size = prefiltered_env_map_tex_->texture->extent_2d().width;
        unsigned int mip_width = size * std::pow(0.5, mip);
        unsigned int mip_height = size * std::pow(0.5, mip);
        uvec2 extent{mip_width, mip_height};
        ctx.begin_rendering({.extent = extent},
                            {RenderingAttachmentInfo{&cube_mip_views[mip][i].value(),
                                                     RenderingAttachmentInfo::Type::Color,
                                                     RenderingAttachmentInfo::LoadOp::Load}});
        ctx.set_viewport_and_scissor(mip_width, mip_height);
        PipelineManager::get().bind_graphics(cmd, prefilter_env_map_pipeline_);

        struct {
          mat4 vp;
          float roughness;
          u32 in_tex_idx, sampler_idx, vertex_buffer_idx;
          float cubemap_res;
        } pc{PROJ * VIEW_MATRICES[i],
             roughness,
             env_cubemap_tex_->view().sampled_img_resource().handle,
             device_->get_bindless_idx(device_->get_or_create_sampler(SamplerCreateInfo{
                 .min_filter = FilterMode::Linear,
                 .mag_filter = FilterMode::Linear,
                 .mipmap_mode = FilterMode::Linear,
                 .address_mode = AddressMode::Repeat,
             })),
             get_device().get_buffer(cube_vertex_buf_)->resource_info_->handle,
             static_cast<float>(env_cubemap_tex_->extent_2d().width)};
        ctx.push_constants(sizeof(pc), &pc);
        ctx.set_cull_mode(CullMode::None);
        ctx.draw(36);
        ctx.end_rendering();
      }
    }
  }

  transition_image(cmd, *irradiance_cubemap_tex_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
}
void IBL::load_pipelines(PipelineLoader& loader) {
  loader.add_compute("ibl/integrate_brdf.comp", &integrate_brdf_pipeline_);
  loader.add_compute("ibl/eq_to_cube.comp", &equirect_to_cube_pipeline2_);
  loader.add_compute("ibl/cube_convolute.comp", &convolute_cube_pipeline_);
  loader.add_graphics(
      GraphicsPipelineCreateInfo{
          .shaders = {{"ibl/prefilter_env_map.vert", ShaderType::Vertex},
                      {"ibl/prefilter_env_map.frag", ShaderType::Fragment}},
          .rendering =
              {
                  .color_formats = {prefiltered_env_map_tex_->texture->format()},
              },
          .rasterization = {.cull_mode = CullMode::None},
          .name = "prefilter env map",
      },
      &prefilter_env_map_pipeline_);
  loader.add_graphics(
      GraphicsPipelineCreateInfo{.shaders = {{"ibl/cube_convolute.vert", ShaderType::Vertex},
                                             {"ibl/cube_convolute.frag", ShaderType::Fragment}},
                                 .rendering =
                                     {
                                         .color_formats = {irradiance_cubemap_tex_->format()},
                                     },
                                 .rasterization = {.cull_mode = CullMode::None},
                                 .name = "cube convolute raster"},
      &convolute_cube_raster_pipeline_);
}
}  // namespace gfx
