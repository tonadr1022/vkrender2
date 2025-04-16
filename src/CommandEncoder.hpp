#pragma once

#include <vulkan/vulkan_core.h>

#include "Common.hpp"

namespace gfx {

struct CmdEncoder {
  explicit CmdEncoder(VkCommandBuffer cmd, VkPipelineLayout default_pipeline_layout)
      : default_pipeline_layout_(default_pipeline_layout), cmd_(cmd) {}
  void dispatch(u32 work_groups_x, u32 work_groups_y, u32 work_groups_z);
  void bind_compute_pipeline(VkPipeline pipeline);
  void bind_descriptor_set(VkPipelineBindPoint bind_point, VkPipelineLayout layout,
                           VkDescriptorSet* set, u32 idx);
  void push_constants(VkPipelineLayout layout, u32 size, void* data);
  void push_constants(u32 size, void* data);

  [[nodiscard]] VkCommandBuffer cmd() const { return cmd_; }

 private:
  VkPipelineLayout default_pipeline_layout_;
  VkCommandBuffer cmd_;
};
}  // namespace gfx
