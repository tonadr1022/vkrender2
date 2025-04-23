#include "Initializers.hpp"

#include <volk.h>
#include <vulkan/vulkan_core.h>

#include "Common.hpp"
#include "vk2/Texture.hpp"

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

// VkImageCreateInfo img_create_info(const ImageCreateInfo& info) {
//   return {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
//           .flags = info.img_flags,
//           .imageType = info.img_type,
//           .format = info.format,
//           .extent = VkExtent3D{info.dims.x, info.dims.y, info.dims.z},
//           .mipLevels = info.mip_levels,
//           .arrayLayers = info.array_layers,
//           .samples = info.samples,
//           .tiling = VK_IMAGE_TILING_OPTIMAL,
//           .usage = info.usage,
//           .initialLayout = info.initial_layout};
// }
namespace {

// uint32_t get_mip_levels(VkExtent2D size) {
//   return static_cast<uint32_t>(std::floor(std::log2(glm::max(size.width, size.height)))) + 1;
// }

}  // namespace

// VkImageCreateInfo img_create_info_2d(VkFormat format, uvec2 dims, bool mipmap,
//                                      VkImageUsageFlags usage, bool mapped) {
//   return img_create_info(
//       {.format = format,
//        .dims = {dims, 1},
//        .mip_levels = mipmap ? get_mip_levels({.width = dims.x, .height = dims.y}) : 1,
//        .usage = usage,
//        .mapped = mapped}
//
//   );
// }

VkImageSubresourceRange subresource_range_whole(VkImageAspectFlags aspect) {
  return {.aspectMask = aspect,
          .baseMipLevel = 0,
          .levelCount = VK_REMAINING_MIP_LEVELS,
          .baseArrayLayer = 0,
          .layerCount = VK_REMAINING_ARRAY_LAYERS};
}

VkRenderingAttachmentInfo rendering_attachment_info(VkImageView texture, VkImageLayout layout,
                                                    VkClearValue* clear_value) {
  return {.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
          .imageView = texture,
          .imageLayout = layout,
          .loadOp = clear_value ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_LOAD,
          .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
          .clearValue = clear_value != nullptr ? *clear_value : VkClearValue{}};
}
VkRenderingAttachmentInfo rendering_attachment_info(vk2::ImageView& texture, VkImageLayout layout,
                                                    VkClearValue* clear_value) {
  return rendering_attachment_info(texture.view(), layout, clear_value);
}

VkRenderingInfo rendering_info(VkExtent2D render_extent,
                               VkRenderingAttachmentInfo* color_attachment,
                               VkRenderingAttachmentInfo* depth_attachment,
                               VkRenderingAttachmentInfo* stencil_attachment) {
  return {.sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
          .renderArea = VkRect2D{{0, 0}, render_extent},
          .layerCount = 1,
          .colorAttachmentCount = color_attachment != nullptr ? 1u : 0u,
          .pColorAttachments = color_attachment,
          .pDepthAttachment = depth_attachment,
          .pStencilAttachment = stencil_attachment};
}

VkRenderingInfo rendering_info(VkExtent2D render_extent,
                               VkRenderingAttachmentInfo* color_attachments, u32 color_att_count,
                               VkRenderingAttachmentInfo* depth_attachment,
                               VkRenderingAttachmentInfo* stencil_attachment) {
  return {.sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
          .renderArea = VkRect2D{{0, 0}, render_extent},
          .layerCount = 1,
          .colorAttachmentCount = color_att_count,
          .pColorAttachments = color_attachments,
          .pDepthAttachment = depth_attachment,
          .pStencilAttachment = stencil_attachment};
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
void begin_debug_utils_label([[maybe_unused]] VkCommandBuffer cmd,
                             [[maybe_unused]] const char* name) {
#ifndef NDEBUG
  VkDebugUtilsLabelEXT debug_label_info{.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT,
                                        .pLabelName = name};
  vkCmdBeginDebugUtilsLabelEXT(cmd, &debug_label_info);
#endif
}

void end_debug_utils_label([[maybe_unused]] VkCommandBuffer cmd) {
#ifndef NDEBUG
  vkCmdEndDebugUtilsLabelEXT(cmd);
#endif
}

}  // namespace gfx::vk2::init
