#pragma once

#include <vulkan/vulkan_core.h>

#include <span>

#include "Common.hpp"

#define SPAN1(x) std::span(std::addressof(x), 1)
#define ARR_SPAN(x) std::span(x, COUNTOF(x))

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

namespace vk2 {
class Texture;
class TextureView;
}  // namespace vk2
namespace vk2::init {
VkDependencyInfo dependency_info(std::span<VkBufferMemoryBarrier2> buffer_barriers,
                                 std::span<VkImageMemoryBarrier2> img_barriers);

VkRenderingAttachmentInfo rendering_attachment_info(VkImageView texture, VkImageLayout layout,
                                                    VkClearValue* clear_value = nullptr);
VkRenderingAttachmentInfo rendering_attachment_info(vk2::TextureView& texture, VkImageLayout layout,
                                                    VkClearValue* clear_value = nullptr);
VkRenderingInfo rendering_info(VkExtent2D render_extent,
                               VkRenderingAttachmentInfo* color_attachment,
                               VkRenderingAttachmentInfo* depth_attachment = nullptr,
                               VkRenderingAttachmentInfo* stencil_attachment = nullptr);
VkRenderingInfo rendering_info(VkExtent2D render_extent,
                               std::span<VkRenderingAttachmentInfo> color_attachment,
                               VkRenderingAttachmentInfo* depth_attachment = nullptr,
                               VkRenderingAttachmentInfo* stencil_attachment = nullptr);
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
