#include "StateTracker.hpp"

#include <volk.h>

#include "Common.hpp"

void StateTracker::barrier() {
  VkDependencyInfo info{
      .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
      .bufferMemoryBarrierCount = static_cast<u32>(buffer_barriers_.size()),
      .pBufferMemoryBarriers = buffer_barriers_.size() ? buffer_barriers_.data() : nullptr,
      .imageMemoryBarrierCount = static_cast<u32>(img_barriers_.size()),
      .pImageMemoryBarriers = img_barriers_.size() ? img_barriers_.data() : nullptr};
  vkCmdPipelineBarrier2(cmd_, &info);
  buffer_barriers_.clear();
  img_barriers_.clear();
}

void StateTracker::add_image(VkImage image, VkAccessFlags2 access, VkPipelineStageFlags2 stage) {
  tracked_imgs_.push_back(ImageState{image, access, stage});
}

VkImageSubresourceRange StateTracker::default_image_subresource_range(VkImageAspectFlags aspect) {
  return {.aspectMask = aspect,
          .baseMipLevel = 0,
          .levelCount = VK_REMAINING_MIP_LEVELS,
          .baseArrayLayer = 0,
          .layerCount = VK_REMAINING_ARRAY_LAYERS};
}

void StateTracker::queue_transition(VkImage image, VkPipelineStageFlags2 dst_stage,
                                    VkAccessFlags2 dst_access, VkImageLayout new_layout,
                                    VkImageAspectFlags aspect) {
  queue_transition(image, dst_stage, dst_access, new_layout,
                   default_image_subresource_range(aspect));
}

void StateTracker::queue_transition(VkImage image, VkPipelineStageFlags2 dst_stage,
                                    VkAccessFlags2 dst_access, VkImageLayout new_layout,
                                    const VkImageSubresourceRange& range) {
  auto it = std::ranges::find_if(tracked_imgs_,
                                 [image](const ImageState& img) { return img.image == image; });
  if (it == tracked_imgs_.end()) {
    tracked_imgs_.push_back(
        ImageState{image, 0, VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT, VK_IMAGE_LAYOUT_UNDEFINED});
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
  img_barriers_.push_back(barrier);
}
