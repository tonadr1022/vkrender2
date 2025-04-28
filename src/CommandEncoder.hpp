#pragma once

#include <vulkan/vulkan_core.h>

#include "Common.hpp"

namespace tracy {
struct VkCtx;
}
namespace gfx {

enum class StoreOp : uint8_t { Store, Discard };

struct RenderPassInfo {};
struct CmdEncoder {
  explicit CmdEncoder(VkCommandBuffer cmd, VkPipelineLayout default_pipeline_layout,
                      tracy::VkCtx* tracy_ctx = nullptr)
      : tracy_ctx_(tracy_ctx), default_pipeline_layout_(default_pipeline_layout), cmd_(cmd) {}
  [[nodiscard]] tracy::VkCtx* get_tracy_ctx() const { return tracy_ctx_; }
  void dispatch(u32 work_groups_x, u32 work_groups_y, u32 work_groups_z);
  void bind_compute_pipeline(VkPipeline pipeline);
  void bind_descriptor_set(VkPipelineBindPoint bind_point, VkPipelineLayout layout,
                           VkDescriptorSet* set, u32 idx);
  void barrier(VkPipelineStageFlags2 src_stage, VkAccessFlags2 src_access,
               VkPipelineStageFlags2 dst_stage, VkAccessFlags2 dst_access);
  void push_constants(VkPipelineLayout layout, u32 size, void* data);
  void push_constants(u32 size, void* data);

  [[nodiscard]] VkCommandBuffer cmd() const { return cmd_; }

 private:
  tracy::VkCtx* tracy_ctx_{};
  VkPipelineLayout default_pipeline_layout_;
  VkCommandBuffer cmd_;
};
}  // namespace gfx
