#pragma once

#include <vulkan/vulkan_core.h>

#include <array>
#include <cassert>
#include <vector>

#include "Common.hpp"

namespace gfx {
class Buffer;
class Image;
struct CmdEncoder;
}  // namespace gfx

namespace gfx {

struct BufferBarrier {
  VkPipelineStageFlags2 src_stage{};
  VkAccessFlags2 src_access{};
  VkPipelineStageFlags2 dst_stage{};
  VkAccessFlags2 dst_access{};
  u32 src_queue{VK_QUEUE_FAMILY_IGNORED};
  u32 dst_queue{VK_QUEUE_FAMILY_IGNORED};
  VkBuffer buffer;
  u64 offset{0};
  u64 size{VK_WHOLE_SIZE};
  BufferBarrier(Buffer& buffer, u32 src_queue, u32 dst_queue, u64 offset = 0,
                u64 size = VK_WHOLE_SIZE);
  BufferBarrier() = default;
};

VkBufferMemoryBarrier2 buffer_memory_barrier(const BufferBarrier& t);

constexpr VkImageSubresourceRange default_image_subresource_range{
    .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
    .baseMipLevel = 0,
    .levelCount = VK_REMAINING_MIP_LEVELS,
    .baseArrayLayer = 0,
    .layerCount = VK_REMAINING_ARRAY_LAYERS};

class StateTracker {
 public:
  explicit StateTracker() {
    tracked_imgs_.reserve(10);
    img_barriers_.reserve(10);
    buffer_barriers_.reserve(10);
  }

  void flush_barriers();

  VkImageSubresourceRange default_image_subresource_range(
      VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT);

  StateTracker& transition(VkImage image, VkPipelineStageFlags2 dst_stage,
                           VkAccessFlags2 dst_access, VkImageLayout new_layout,
                           VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT);
  StateTracker& transition(Image& image, VkPipelineStageFlags2 dst_stage, VkAccessFlags2 dst_access,
                           VkImageLayout new_layout,
                           VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT);
  StateTracker& transition_img_to_copy_dst(Image& image,
                                           VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT);
  StateTracker& transition_buffer_to_transfer_dst(VkBuffer buffer);

  StateTracker& transition(VkImage image, VkPipelineStageFlags2 dst_stage,
                           VkAccessFlags2 dst_access, VkImageLayout new_layout,
                           const VkImageSubresourceRange& range);
  StateTracker& transition(Image& image, VkPipelineStageFlags2 dst_stage, VkAccessFlags2 dst_access,
                           VkImageLayout new_layout, const VkImageSubresourceRange& range);
  StateTracker& buffer_barrier(VkBuffer buffer, VkPipelineStageFlags2 dst_stage,
                               VkAccessFlags2 dst_access);
  StateTracker& buffer_barrier(const Buffer& buffer, VkPipelineStageFlags2 dst_stage,
                               VkAccessFlags2 dst_access);
  void barrier();

  StateTracker& queue_transfer_buffer(StateTracker& dst_tracker, VkPipelineStageFlags2 dst_stage,
                                      VkAccessFlags2 dst_access, VkBuffer buffer, u32 src_queue,
                                      u32 dst_queue, u64 offset = 0, u64 size = VK_WHOLE_SIZE);

  StateTracker& reset(CmdEncoder& cmd);
  StateTracker& flush_transfers(u32 queue_idx);
  struct ImageState {
    VkImage image;
    VkAccessFlags2 curr_access;
    VkPipelineStageFlags2 curr_stage;
    VkImageLayout curr_layout;
  };

  struct BufferState {
    VkBuffer buffer;
    VkAccessFlags2 curr_access;
    VkPipelineStageFlags2 curr_stage;
  };

  [[nodiscard]] ImageState* get_img_state(VkImage image) {
    for (auto& img : tracked_imgs_) {
      if (img.image == image) {
        return &img;
      }
    }
    return nullptr;
  }

 private:
  std::vector<ImageState> tracked_imgs_;
  std::vector<BufferState> tracked_buffers_;
  std::vector<VkImageMemoryBarrier2> img_barriers_;
  std::vector<VkBufferMemoryBarrier2> buffer_barriers_;
  // TODO: initialize a vector in constructor instead, using the precise number of queues in play
  static constexpr int max_queue_idx = 5;
  std::array<std::vector<VkBufferMemoryBarrier2>, max_queue_idx> buffer_transfer_barriers_;

  VkCommandBuffer cmd_{};
  CmdEncoder* cmd2_{};

  decltype(tracked_imgs_.end()) get_img(VkImage image) {
    for (auto it = tracked_imgs_.begin(); it != tracked_imgs_.end(); it++) {
      if (it->image == image) {
        return it;
      }
    }
    return tracked_imgs_.end();
  }
};
}  // namespace gfx
