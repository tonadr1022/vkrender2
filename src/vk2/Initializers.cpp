#include "Initializers.hpp"

#include <volk.h>
#include <vulkan/vulkan_core.h>

#include "Common.hpp"

namespace gfx::vk2::init {

VkCommandBufferBeginInfo command_buffer_begin_info(VkCommandBufferUsageFlags flags) {
  return {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, .flags = flags};
}

VkSubmitInfo2 queue_submit_info() {
  return VkSubmitInfo2{.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
}

VkSubmitInfo2 queue_submit_info(std::span<VkCommandBufferSubmitInfo> cmds,
                                std::span<VkSemaphoreSubmitInfo> wait_semaphores,
                                std::span<VkSemaphoreSubmitInfo> submit_semaphores) {
  return VkSubmitInfo2{.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
                       .waitSemaphoreInfoCount = static_cast<u32>(wait_semaphores.size()),
                       .pWaitSemaphoreInfos = wait_semaphores.data(),
                       .commandBufferInfoCount = static_cast<u32>(cmds.size()),
                       .pCommandBufferInfos = cmds.data(),
                       .signalSemaphoreInfoCount = static_cast<u32>(submit_semaphores.size()),
                       .pSignalSemaphoreInfos = submit_semaphores.data()};
}

VkCommandBufferSubmitInfo command_buffer_submit_info(VkCommandBuffer buffer) {
  return {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO, .commandBuffer = buffer};
}

VkSemaphoreSubmitInfo semaphore_submit_info(VkSemaphore semaphore, VkPipelineStageFlags2 stage_mask,
                                            u64 value) {
  return {
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
      .semaphore = semaphore,
      .value = value,
      .stageMask = stage_mask,
      .deviceIndex = 0,
  };
}

VkImageSubresourceRange image_subresource_range(VkImageAspectFlags aspectMask) {
  VkImageSubresourceRange sub_image{};
  sub_image.aspectMask = aspectMask;
  sub_image.baseMipLevel = 0;
  sub_image.levelCount = VK_REMAINING_MIP_LEVELS;
  sub_image.baseArrayLayer = 0;
  sub_image.layerCount = VK_REMAINING_ARRAY_LAYERS;
  return sub_image;
}
void transition_image(VkCommandBuffer cmd, VkImage image, VkImageLayout currentLayout,
                      VkImageLayout newLayout) {
  VkImageMemoryBarrier2 image_barrier{.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
  image_barrier.pNext = nullptr;

  image_barrier.srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
  image_barrier.srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT;
  image_barrier.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
  image_barrier.dstAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT | VK_ACCESS_2_MEMORY_READ_BIT;

  image_barrier.oldLayout = currentLayout;
  image_barrier.newLayout = newLayout;

  VkImageAspectFlags aspect_mask = (newLayout == VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL)
                                       ? VK_IMAGE_ASPECT_DEPTH_BIT
                                       : VK_IMAGE_ASPECT_COLOR_BIT;
  image_barrier.subresourceRange = image_subresource_range(aspect_mask);
  image_barrier.image = image;

  VkDependencyInfo dep_info{};
  dep_info.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
  dep_info.pNext = nullptr;

  dep_info.imageMemoryBarrierCount = 1;
  dep_info.pImageMemoryBarriers = &image_barrier;

  vkCmdPipelineBarrier2(cmd, &dep_info);
}

VkImageSubresourceRange subresource_range_whole(VkImageAspectFlags aspect) {
  return {.aspectMask = aspect,
          .baseMipLevel = 0,
          .levelCount = VK_REMAINING_MIP_LEVELS,
          .baseArrayLayer = 0,
          .layerCount = VK_REMAINING_ARRAY_LAYERS};
}

VkDependencyInfo dependency_info(std::span<VkBufferMemoryBarrier2> buffer_barriers,
                                 std::span<VkImageMemoryBarrier2> img_barriers) {
  return {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
          .bufferMemoryBarrierCount = static_cast<u32>(buffer_barriers.size()),
          .pBufferMemoryBarriers = buffer_barriers.size() ? buffer_barriers.data() : nullptr,
          .imageMemoryBarrierCount = static_cast<u32>(img_barriers.size()),
          .pImageMemoryBarriers = img_barriers.size() ? img_barriers.data() : nullptr};
}
VkBufferCopy2KHR buffer_copy(VkDeviceSize src_offset, VkDeviceSize dst_offset, VkDeviceSize size) {
  return VkBufferCopy2KHR{.sType = VK_STRUCTURE_TYPE_BUFFER_COPY_2_KHR,
                          .srcOffset = src_offset,
                          .dstOffset = dst_offset,
                          .size = size};
}

}  // namespace gfx::vk2::init
