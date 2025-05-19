#pragma once

#include "Common.hpp"
#include "Types.hpp"
#include "vk2/Swapchain.hpp"

namespace tracy {
struct VkCtx;
}

namespace gfx {
class Buffer;
class Device;

struct CmdEncoder {
  // TODO: constructor can't take in command buffer
  explicit CmdEncoder(Device* device, VkPipelineLayout default_pipeline_layout)
      : device_(device), default_pipeline_layout_(default_pipeline_layout) {}

  void reset(u32 frame_in_flight);

  void dispatch(u32 work_groups_x, u32 work_groups_y, u32 work_groups_z) const;
  void bind_descriptor_set(VkPipelineBindPoint bind_point, VkPipelineLayout layout,
                           VkDescriptorSet* set, u32 idx) const;
  void barrier(VkPipelineStageFlags2 src_stage, VkAccessFlags2 src_access,
               VkPipelineStageFlags2 dst_stage, VkAccessFlags2 dst_access) const;

  void transition_image(ImageHandle image, VkImageLayout old_layout, VkImageLayout new_layout,
                        VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT);
  void transition_image(ImageHandle image, VkImageLayout new_layout,
                        VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT);
  void push_constants(VkPipelineLayout layout, u32 size, void* data) const;
  void push_constants(u32 size, void* data) const;
  void set_viewport_and_scissor(u32 width, u32 height) const;
  void set_viewport_and_scissor(vec2 extent, vec2 offset = vec2{0}) const;
  void set_cull_mode(CullMode mode) const;
  void set_depth_bias(float constant_factor, float bias, float slope_factor) const;
  void bind_pipeline(PipelineBindPoint bind_point, PipelineHandle pipeline) const;
  void end_rendering() const;
  void draw(u32 vertex_count, u32 instance_count = 1, u32 first_vertex = 0,
            u32 first_instance = 0) const;
  struct RenderArea {
    uvec2 extent{};
    ivec2 offset{};
  };

  void begin_rendering(const RenderArea& render_area,
                       std::initializer_list<RenderingAttachmentInfo> attachment_descs);

  void fill_buffer(BufferHandle buffer, u64 offset, u64 size = constants::whole_size, u32 data = 0);
  void bind_index_buffer(BufferHandle buffer, u64 offset = 0, IndexType type = IndexType::uint32);
  void update_buffer(BufferHandle buffer, u64 offset, u64 size, void* data);
  void begin_region(const char* name) const;
  void end_region() const;
  void draw_indexed_indirect(BufferHandle buffer, u64 offset, u32 draw_count, u32 stride);
  void draw_indexed_indirect_count(BufferHandle draw_cmd_buf, u64 draw_cmd_offset,
                                   BufferHandle draw_count_buf, u64 draw_count_offset,
                                   u32 draw_count, u32 stride);
  void copy_buffer(const Buffer& src, const Buffer& dst, u64 src_offset, u64 dst_offset,
                   u64 size) const;

  [[nodiscard]] VkCommandBuffer cmd() const { return get_cmd_buf(); }

  [[nodiscard]] VkCommandBuffer get_cmd_buf() const {
    return command_bufs_[frame_in_flight_][(u32)queue_];
  }
  [[nodiscard]] VkCommandPool get_cmd_pool() const {
    return command_pools_[frame_in_flight_][(u32)queue_];
  }

  void begin_swapchain_blit();
  void blit_img(ImageHandle src, ImageHandle dst, uvec3 extent, VkImageAspectFlags aspect);

 private:
  // TODO: fix
  friend class Device;
  Device* device_{};
  QueueType queue_{};
  u32 id_{UINT32_MAX};
  u32 frame_in_flight_{};

  VkCommandPool command_pools_[frames_in_flight][(u32)QueueType::Count] = {};
  VkCommandBuffer command_bufs_[frames_in_flight][(u32)QueueType::Count] = {};
  std::vector<vk2::Swapchain> submit_swapchains_;
  std::vector<VkSemaphore> wait_semaphores_;
  std::vector<VkSemaphore> signal_semaphores_;

  VkPipelineLayout default_pipeline_layout_;
};

}  // namespace gfx
