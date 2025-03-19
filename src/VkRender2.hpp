#pragma once

#include <vulkan/vulkan_core.h>

#include <filesystem>
#include <optional>

#include "App.hpp"
#include "Scene.hpp"
#include "StateTracker.hpp"
#include "vk2/Buffer.hpp"
#include "vk2/DeletionQueue.hpp"
#include "vk2/PipelineManager.hpp"
#include "vk2/Texture.hpp"

struct CmdEncoder {
  explicit CmdEncoder(VkCommandBuffer cmd) : cmd_(cmd) {}
  void dispatch(u32 work_groups_x, u32 work_groups_y, u32 work_groups_z);
  void bind_compute_pipeline(VkPipeline pipeline);
  void bind_descriptor_set(VkPipelineBindPoint bind_point, VkPipelineLayout layout,
                           VkDescriptorSet* set, u32 idx);
  void push_constants(VkPipelineLayout layout, u32 size, void* data);

  [[nodiscard]] VkCommandBuffer cmd() const { return cmd_; }

 private:
  VkCommandBuffer cmd_;
};
template <typename T>
struct InFlightResource {
  T data;
  VkFence fence;
};

// yes everything is public, this is a wrapper for a main.cpp
struct VkRender2 final : public BaseRenderer {
  explicit VkRender2(const InitInfo& info);
  ~VkRender2() override;
  void on_update() override;
  void on_draw() override;
  void on_gui() override;
  void on_resize() override;
  void create_attachment_imgs();
  void set_viewport_and_scissor(VkCommandBuffer cmd, VkExtent2D extent);

  // TODO: refactor
  struct LoadedScene {
    SceneGraphData scene_graph_data;
    vk2::Buffer vertex_buffer;
    vk2::Buffer index_buffer;
    std::vector<vk2::Sampler> samplers;
  };
  std::optional<LoadedScene> cube;

  StateTracker state;
  std::optional<vk2::Texture> depth_img;
  std::optional<vk2::Texture> img;

  vk2::DeletionQueue main_del_q;
  std::filesystem::path resource_dir;
  std::filesystem::path shader_dir;
  [[nodiscard]] std::string get_shader_path(const std::string& path) const;
  vk2::PipelineHandle img_pipeline;
  vk2::PipelineHandle draw_pipeline;
  VkPipelineLayout default_pipeline_layout{};
  std::queue<InFlightResource<std::pair<vk2::Buffer, vk2::Buffer>>>
      in_flight_vertex_index_staging_buffers_;

  std::vector<vk2::Buffer> free_staging_buffers_;

  // non owning
  VkDescriptorSet main_set{};
  VmaAllocator allocator_;
};
