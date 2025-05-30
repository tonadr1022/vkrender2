#include "StateTracker.hpp"

#include <volk.h>
#include <vulkan/vk_enum_string_helper.h>
#include <vulkan/vulkan_core.h>

#include <algorithm>

#include "CommandEncoder.hpp"
#include "Common.hpp"
#include "vk2/Buffer.hpp"
#include "vk2/Initializers.hpp"
#include "vk2/Texture.hpp"

namespace gfx {

void StateTracker::flush_barriers() {
  if (buffer_barriers_.empty() && img_barriers_.empty()) return;
  VkDependencyInfo info = vk2::init::dependency_info(buffer_barriers_, img_barriers_);
  vkCmdPipelineBarrier2KHR(cmd_, &info);
  buffer_barriers_.clear();
  img_barriers_.clear();
}

VkImageSubresourceRange StateTracker::default_image_subresource_range(VkImageAspectFlags aspect) {
  return {.aspectMask = aspect,
          .baseMipLevel = 0,
          .levelCount = VK_REMAINING_MIP_LEVELS,
          .baseArrayLayer = 0,
          .layerCount = VK_REMAINING_ARRAY_LAYERS};
}

StateTracker& StateTracker::transition(Image& image, VkPipelineStageFlags2 dst_stage,
                                       VkAccessFlags2 dst_access, VkImageLayout new_layout,
                                       VkImageAspectFlags aspect) {
  transition(image, dst_stage, dst_access, new_layout, default_image_subresource_range(aspect));
  return *this;
}
StateTracker& StateTracker::transition(VkImage image, VkPipelineStageFlags2 dst_stage,
                                       VkAccessFlags2 dst_access, VkImageLayout new_layout,
                                       VkImageAspectFlags aspect) {
  transition(image, dst_stage, dst_access, new_layout, default_image_subresource_range(aspect));
  return *this;
}

StateTracker& StateTracker::transition(Image& image, VkPipelineStageFlags2 dst_stage,
                                       VkAccessFlags2 dst_access, VkImageLayout new_layout,
                                       const VkImageSubresourceRange& range) {
  image.curr_layout = new_layout;
  return transition(image.image(), dst_stage, dst_access, new_layout, range);
}

StateTracker& StateTracker::transition(VkImage image, VkPipelineStageFlags2 dst_stage,
                                       VkAccessFlags2 dst_access, VkImageLayout new_layout,
                                       const VkImageSubresourceRange& range) {
  auto it = std::ranges::find_if(
      tracked_imgs_, [image = image](const ImageState& img) { return img.image == image; });
  if (it == tracked_imgs_.end()) {
    tracked_imgs_.push_back(
        ImageState{image, 0, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_IMAGE_LAYOUT_UNDEFINED});
    it = std::prev(tracked_imgs_.end());
  }
  VkImageMemoryBarrier2 barrier{.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                                .srcStageMask = it->curr_stage,
                                .srcAccessMask = it->curr_access,
                                .dstStageMask = dst_stage,
                                .dstAccessMask = dst_access,
                                .oldLayout = it->curr_layout,
                                .newLayout = new_layout,
                                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                .image = it->image,
                                .subresourceRange = range

  };
  it->curr_access = dst_access;
  it->curr_stage = dst_stage;
  it->curr_layout = new_layout;
  img_barriers_.push_back(barrier);
  return *this;
}

StateTracker& StateTracker::buffer_barrier(VkBuffer buffer, VkPipelineStageFlags2 dst_stage,
                                           VkAccessFlags2 dst_access) {
  auto it = std::ranges::find_if(tracked_buffers_,
                                 [buffer](const BufferState& buf) { return buf.buffer == buffer; });
  if (it == tracked_buffers_.end()) {
    tracked_buffers_.push_back(
        BufferState{buffer, VK_ACCESS_2_NONE, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT});
    it = std::prev(tracked_buffers_.end());
  }
  buffer_barriers_.emplace_back(VkBufferMemoryBarrier2{
      .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
      .srcStageMask = it->curr_stage,
      .srcAccessMask = it->curr_access,
      .dstStageMask = dst_stage,
      .dstAccessMask = dst_access,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .buffer = buffer,
      .offset = 0,
      .size = VK_WHOLE_SIZE,
  });
  it->curr_access = dst_access;
  it->curr_stage = dst_stage;
  return *this;
}

// TODO: might need to change src stage/access
BufferBarrier::BufferBarrier(Buffer& buffer, u32 src_queue, u32 dst_queue, u64 offset, u64 size)
    : src_stage(VK_PIPELINE_STAGE_2_TRANSFER_BIT),
      src_access(VK_ACCESS_2_TRANSFER_WRITE_BIT),
      src_queue(src_queue),
      dst_queue(dst_queue),
      buffer(buffer.buffer()),
      offset(offset),
      size(size) {}
VkBufferMemoryBarrier2 buffer_memory_barrier(const BufferBarrier& t) {
  return {.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
          .srcStageMask = t.src_stage,
          .srcAccessMask = t.src_access,
          .dstStageMask = t.dst_stage,
          .dstAccessMask = t.dst_access,
          .srcQueueFamilyIndex = t.src_queue,
          .dstQueueFamilyIndex = t.dst_queue,
          .buffer = t.buffer,
          .offset = t.offset,
          .size = t.size};
}

StateTracker& StateTracker::flush_transfers(u32 queue_idx) {
  auto info = vk2::init::dependency_info(buffer_transfer_barriers_[queue_idx], img_barriers_);
  vkCmdPipelineBarrier2KHR(cmd_, &info);
  buffer_transfer_barriers_[queue_idx].clear();
  return *this;
}

StateTracker& StateTracker::reset(VkCommandBuffer cmd) {
  assert(img_barriers_.empty());
  assert(buffer_barriers_.empty());
  cmd_ = cmd;
  tracked_buffers_.clear();
  tracked_imgs_.clear();
  img_barriers_.clear();
  buffer_barriers_.clear();
  return *this;
}

StateTracker& StateTracker::reset(CmdEncoder& cmd) { return reset(cmd.cmd()); }

void StateTracker::barrier() {
  VkMemoryBarrier2 barrier{
      .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
      .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
      .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
      .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
      .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
  };
  VkDependencyInfo info{
      .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
      .memoryBarrierCount = 1,
      .pMemoryBarriers = &barrier,
  };
  vkCmdPipelineBarrier2KHR(cmd_, &info);
}
StateTracker& StateTracker::buffer_barrier(const Buffer& buffer, VkPipelineStageFlags2 dst_stage,
                                           VkAccessFlags2 dst_access) {
  return buffer_barrier(buffer.buffer(), dst_stage, dst_access);
}

StateTracker& StateTracker::transition_img_to_copy_dst(Image& image, VkImageAspectFlags aspect) {
  return transition(image, VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, aspect);
}

StateTracker& StateTracker::transition_buffer_to_transfer_dst(VkBuffer buffer) {
  return buffer_barrier(buffer, VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
}

}  // namespace gfx
