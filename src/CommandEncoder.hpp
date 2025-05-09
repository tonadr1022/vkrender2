#pragma once

#include <vulkan/vulkan_core.h>

#include "Common.hpp"
#include "Types.hpp"

namespace tracy {
struct VkCtx;
}

namespace gfx {
class Buffer;
class Device;

struct CmdEncoder {
  explicit CmdEncoder(Device* device, VkCommandBuffer cmd, VkPipelineLayout default_pipeline_layout,
                      tracy::VkCtx* tracy_ctx = nullptr)
      : tracy_ctx_(tracy_ctx),
        device_(device),
        default_pipeline_layout_(default_pipeline_layout),
        cmd_(cmd) {}
  [[nodiscard]] tracy::VkCtx* get_tracy_ctx() const { return tracy_ctx_; }
  void dispatch(u32 work_groups_x, u32 work_groups_y, u32 work_groups_z);
  void bind_compute_pipeline(VkPipeline pipeline);
  void bind_descriptor_set(VkPipelineBindPoint bind_point, VkPipelineLayout layout,
                           VkDescriptorSet* set, u32 idx);
  void barrier(VkPipelineStageFlags2 src_stage, VkAccessFlags2 src_access,
               VkPipelineStageFlags2 dst_stage, VkAccessFlags2 dst_access);
  void push_constants(VkPipelineLayout layout, u32 size, void* data);
  void push_constants(u32 size, void* data);
  void set_viewport_and_scissor(u32 width, u32 height);
  void set_cull_mode(CullMode mode);
  void set_depth_bias(float constant_factor, float bias, float slope_factor);
  void bind_pipeline(PipelineBindPoint bind_point, PipelineHandle pipeline);
  void end_rendering();
  void draw(u32 vertex_count, u32 instance_count = 1, u32 first_vertex = 0, u32 first_instance = 0);
  struct RenderArea {
    uvec2 extent{};
    ivec2 offset{};
  };

  void begin_rendering(const RenderArea& render_area,
                       std::initializer_list<RenderingAttachmentInfo> attachment_descs);

  void copy_buffer(const Buffer& src, const Buffer& dst, u64 src_offset, u64 dst_offset,
                   u64 size) const;

  [[nodiscard]] VkCommandBuffer cmd() const { return cmd_; }

 private:
  tracy::VkCtx* tracy_ctx_{};
  Device* device_{};
  VkPipelineLayout default_pipeline_layout_;
  VkCommandBuffer cmd_;
};
}  // namespace gfx
