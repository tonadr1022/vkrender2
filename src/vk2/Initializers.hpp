#pragma once

#include <vulkan/vulkan_core.h>

#include <span>

#include "Common.hpp"

#define SPAN1(x) std::span(std::addressof(x), 1)

namespace vk2 {

struct ImageCreateInfo {
  VkImageCreateFlags img_flags{};
  VkImageType img_type{VK_IMAGE_TYPE_2D};
  VkFormat format{};
  uvec3 dims{};
  u32 mip_levels{1};
  u32 array_layers{1};
  VkSampleCountFlagBits samples{VK_SAMPLE_COUNT_1_BIT};
  VkImageUsageFlags usage{};
  VkImageLayout initial_layout{VK_IMAGE_LAYOUT_UNDEFINED};
  bool mapped{false};
};

}  // namespace vk2

namespace vk2::init {

VkCommandBufferBeginInfo command_buffer_begin_info(
    VkCommandBufferUsageFlags flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
VkSubmitInfo2 queue_submit_info();
VkSubmitInfo2 queue_submit_info(std::span<VkCommandBufferSubmitInfo> cmds,
                                std::span<VkSemaphoreSubmitInfo> wait_semaphores,
                                std::span<VkSemaphoreSubmitInfo> submit_semaphores);

VkCommandBufferSubmitInfo command_buffer_submit_info(VkCommandBuffer buffer);
VkSemaphoreSubmitInfo semaphore_submit_info(VkSemaphore semaphore, VkPipelineStageFlags2 stage_mask,
                                            u32 value = 1);

VkImageSubresourceRange image_subresource_range(VkImageAspectFlags aspectMask);

VkImageCreateInfo img_create_info(const ImageCreateInfo& info);
VkImageCreateInfo img_create_info_2d(VkFormat format, uvec2 dims, bool mipmap,
                                     VkImageUsageFlags usage, bool mapped = false);

void transition_image(VkCommandBuffer cmd, VkImage image, VkImageLayout currentLayout,
                      VkImageLayout newLayout);

VkImageSubresourceRange subresource_range_whole(VkImageAspectFlags aspect);
}  // namespace vk2::init
