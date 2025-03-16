#pragma once

#include <vulkan/vulkan_core.h>

#include <vector>
struct StateTracker {
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
  StateTracker() {
    tracked_imgs.reserve(10);
    img_barriers.reserve(10);
    buffer_barriers.reserve(10);
  }
  std::vector<ImageState> tracked_imgs;
  std::vector<BufferState> tracked_buffers;
  std::vector<VkImageMemoryBarrier2> img_barriers;
  std::vector<VkBufferMemoryBarrier2> buffer_barriers;

  void flush(VkCommandBuffer cmd);

  void add_image(VkImage image, VkAccessFlags2 access, VkPipelineStageFlags2 stage);

  VkImageSubresourceRange default_image_subresource_range(
      VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT);

  void queue_transition(VkImage image, VkPipelineStageFlags2 dst_stage, VkAccessFlags2 dst_access,
                        VkImageLayout new_layout,
                        VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT);

  void queue_transition(VkImage image, VkPipelineStageFlags2 dst_stage, VkAccessFlags2 dst_access,
                        VkImageLayout new_layout, const VkImageSubresourceRange& range);

 private:
  decltype(tracked_imgs.end()) get_img(VkImage image) {
    for (auto it = tracked_imgs.begin(); it != tracked_imgs.end(); it++) {
      if (it->image == image) {
        return it;
      }
    }
    return tracked_imgs.end();
  }
};
