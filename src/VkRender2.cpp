#include "VkRender2.hpp"

#include <volk.h>
#include <vulkan/vulkan_core.h>

#include <algorithm>

#include "vk2/Device.hpp"
#include "vk2/Initializers.hpp"
#include "vk2/VkCommon.hpp"

// void new_compute_pipeline() {
//   VkComputePipelineCreateInfo info{.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
// }
using namespace vk2;
VkRender2::VkRender2(const InitInfo& info)
    : BaseRenderer(info, BaseRenderer::BaseInitInfo{.frames_in_flight = 2}) {
  {
    auto dims = window_dims();
    img = vk2::device().alloc_img_with_view(
        vk2::init::img_create_info_2d(
            VK_FORMAT_R8G8B8A8_UNORM, dims, false,
            VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT),
        init::subresource_range_whole(VK_IMAGE_ASPECT_COLOR_BIT), VK_IMAGE_VIEW_TYPE_2D);

    main_del_q_.push([this]() { device().destroy_img(img); });
  }
}

void VkRender2::on_update() {}

void VkRender2::on_draw() {
  VkCommandBuffer cmd = curr_frame().main_cmd_buffer;
  auto info = vk2::init::command_buffer_begin_info();
  VK_CHECK(vkResetCommandBuffer(cmd, 0));
  VK_CHECK(vkBeginCommandBuffer(cmd, &info));
  auto& swapchain_img = swapchain_.imgs[curr_swapchain_img_idx()];
  state.add_image(swapchain_img, 0, VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT_KHR);

  state.queue_transition(swapchain_.imgs[curr_swapchain_img_idx()],
                         VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                         VK_ACCESS_2_MEMORY_WRITE_BIT, VK_IMAGE_LAYOUT_GENERAL);

  state.flush(cmd);

  VkClearColorValue clear_value;
  float flash = std::abs(std::sin(curr_frame_num() / 120.f));
  clear_value = {{0.0f, 0.0f, flash, 1.0f}};

  VkImageSubresourceRange clear_range =
      vk2::init::image_subresource_range(VK_IMAGE_ASPECT_COLOR_BIT);
  vkCmdClearColorImage(cmd, swapchain_.imgs[curr_swapchain_img_idx()], VK_IMAGE_LAYOUT_GENERAL,
                       &clear_value, 1, &clear_range);
  state.queue_transition(swapchain_img, 0, VK_ACCESS_2_MEMORY_READ_BIT,
                         VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);
  state.flush(cmd);
  VK_CHECK(vkEndCommandBuffer(cmd));

  submit_single_command_buf_to_graphics(cmd);
}

void VkRender2::on_gui() {}
void StateTracker::flush(VkCommandBuffer cmd) {
  VkDependencyInfo info{
      .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
      .bufferMemoryBarrierCount = static_cast<u32>(buffer_barriers.size()),
      .pBufferMemoryBarriers = buffer_barriers.size() ? buffer_barriers.data() : nullptr,

      .imageMemoryBarrierCount = static_cast<u32>(img_barriers.size()),
      .pImageMemoryBarriers = img_barriers.size() ? img_barriers.data() : nullptr};
  vkCmdPipelineBarrier2(cmd, &info);
  buffer_barriers.clear();
  img_barriers.clear();
}

void StateTracker::add_image(VkImage image, VkAccessFlags2 access, VkPipelineStageFlags2 stage) {
  tracked_imgs.push_back(ImageState{image, access, stage});
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
  auto it = std::ranges::find_if(tracked_imgs,
                                 [image](const ImageState& img) { return img.image == image; });
  if (it == tracked_imgs.end()) {
    tracked_imgs.push_back(ImageState{image, 0, 0, new_layout});
    it = std::prev(tracked_imgs.end());
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
  img_barriers.push_back(barrier);
}
