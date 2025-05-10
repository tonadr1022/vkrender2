#pragma once

#include <vulkan/vulkan_core.h>

#include <span>

#include "Common.hpp"

#define SPAN1(x) std::span(std::addressof(x), 1)
#define ARR_SPAN(x) std::span(x, COUNTOF(x))

namespace gfx {
class ImageView;
}  // namespace gfx
namespace gfx::vk2::init {
VkDependencyInfo dependency_info(std::span<VkBufferMemoryBarrier2> buffer_barriers,
                                 std::span<VkImageMemoryBarrier2> img_barriers);

VkBufferCopy2KHR buffer_copy(VkDeviceSize src_offset, VkDeviceSize dst_offset, VkDeviceSize size);

VkCommandBufferBeginInfo command_buffer_begin_info(
    VkCommandBufferUsageFlags flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
VkSubmitInfo2 queue_submit_info();
VkSubmitInfo2 queue_submit_info(std::span<VkCommandBufferSubmitInfo> cmds,
                                std::span<VkSemaphoreSubmitInfo> wait_semaphores,
                                std::span<VkSemaphoreSubmitInfo> submit_semaphores);

VkCommandBufferSubmitInfo command_buffer_submit_info(VkCommandBuffer buffer);
VkSemaphoreSubmitInfo semaphore_submit_info(VkSemaphore semaphore, VkPipelineStageFlags2 stage_mask,
                                            u64 value = 1);

VkImageSubresourceRange image_subresource_range(VkImageAspectFlags aspectMask);

void transition_image(VkCommandBuffer cmd, VkImage image, VkImageLayout currentLayout,
                      VkImageLayout newLayout);

void begin_debug_utils_label(VkCommandBuffer cmd, const char* name);
void end_debug_utils_label(VkCommandBuffer cmd);

VkImageSubresourceRange subresource_range_whole(VkImageAspectFlags aspect);
}  // namespace gfx::vk2::init
