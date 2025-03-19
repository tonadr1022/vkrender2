#pragma once

#include <vulkan/vulkan_core.h>

#include <cassert>
#include <vector>
class StateTracker {
 public:
  explicit StateTracker() {
    tracked_imgs_.reserve(10);
    img_barriers_.reserve(10);
    buffer_barriers_.reserve(10);
  }

  void barrier();

  void add_image(VkImage image, VkAccessFlags2 access, VkPipelineStageFlags2 stage);

  VkImageSubresourceRange default_image_subresource_range(
      VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT);

  void transition(VkImage image, VkPipelineStageFlags2 dst_stage, VkAccessFlags2 dst_access,
                  VkImageLayout new_layout, VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT);

  void transition(VkImage image, VkPipelineStageFlags2 dst_stage, VkAccessFlags2 dst_access,
                  VkImageLayout new_layout, const VkImageSubresourceRange& range);

  void reset(VkCommandBuffer cmd) {
    assert(img_barriers_.empty());
    assert(buffer_barriers_.empty());
    cmd_ = cmd;
    tracked_buffers_.clear();
    tracked_imgs_.clear();
    img_barriers_.clear();
    buffer_barriers_.clear();
  }

 private:
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

  std::vector<ImageState> tracked_imgs_;
  std::vector<BufferState> tracked_buffers_;
  std::vector<VkImageMemoryBarrier2> img_barriers_;
  std::vector<VkBufferMemoryBarrier2> buffer_barriers_;

  VkCommandBuffer cmd_{};

  decltype(tracked_imgs_.end()) get_img(VkImage image) {
    for (auto it = tracked_imgs_.begin(); it != tracked_imgs_.end(); it++) {
      if (it->image == image) {
        return it;
      }
    }
    return tracked_imgs_.end();
  }
};
