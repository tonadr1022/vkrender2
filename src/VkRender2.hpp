#pragma once

#include <vulkan/vulkan_core.h>

#include <filesystem>

#include "App.hpp"
#include "StateTracker.hpp"
#include "vk2/DeletionQueue.hpp"
#include "vk2/PipelineManager.hpp"
#include "vk2/Texture.hpp"

struct CmdEncoder {
  explicit CmdEncoder(VkCommandBuffer cmd) : cmd_(cmd) {}
  void reset_and_begin();
  void dispatch(u32 work_groups_x, u32 work_groups_y, u32 work_groups_z);
  void bind_compute_pipeline(VkPipeline pipeline);
  void bind_descriptor_set(VkPipelineBindPoint bind_point, VkPipelineLayout layout,
                           VkDescriptorSet* set, u32 idx);
  void push_constants(VkPipelineLayout layout, u32 size, void* data);

  [[nodiscard]] VkCommandBuffer cmd() const { return cmd_; }

 private:
  VkCommandBuffer cmd_;
};

// yes everything is public, this is a wrapper for a main.cpp
struct VkRender2 final : public BaseRenderer {
  explicit VkRender2(const InitInfo& info);
  ~VkRender2() override;
  void on_update() override;
  void on_draw() override;
  void on_gui() override;
  StateTracker state;
  std::optional<vk2::Texture> img;
  vk2::DeletionQueue main_del_q;
  std::filesystem::path resource_dir;
  std::filesystem::path shader_dir;
  [[nodiscard]] std::string get_shader_path(const std::string& path) const;
  vk2::PipelineHandle img_pipeline;
  vk2::PipelineHandle draw_pipeline;
  VkPipelineLayout default_pipeline_layout{};

  // non owning
  VkDescriptorSet main_set{};
  VmaAllocator allocator_;
};
