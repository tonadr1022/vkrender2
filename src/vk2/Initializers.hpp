#pragma once

#include <vulkan/vulkan_core.h>

#include <span>

#include "Common.hpp"

#define SPAN1(x) std::span(&x, 1)

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
void transition_image(VkCommandBuffer cmd, VkImage image, VkImageLayout currentLayout,
                      VkImageLayout newLayout);

}  // namespace vk2::init
