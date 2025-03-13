#include "VkRender2.hpp"

#include <volk.h>
#include <vulkan/vulkan_core.h>

#include "vk2/Initializers.hpp"
#include "vk2/VkCommon.hpp"

VkRender2::VkRender2(const InitInfo& info)
    : BaseRenderer(info, BaseRenderer::BaseInitInfo{.frames_in_flight = 2}) {}

void VkRender2::on_update() {}

void VkRender2::on_draw() {
  VkCommandBuffer cmd = curr_frame().main_cmd_buffer;
  auto info = vk2::init::command_buffer_begin_info();
  VK_CHECK(vkResetCommandBuffer(cmd, 0));
  VK_CHECK(vkBeginCommandBuffer(cmd, &info));

  vk2::init::transition_image(cmd, swapchain_.imgs[curr_swapchain_img_idx()],
                              VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
  VkClearColorValue clear_value;
  float flash = std::abs(std::sin(curr_frame_num() / 120.f));
  clear_value = {{0.0f, 0.0f, flash, 1.0f}};

  VkImageSubresourceRange clear_range =
      vk2::init::image_subresource_range(VK_IMAGE_ASPECT_COLOR_BIT);
  vkCmdClearColorImage(cmd, swapchain_.imgs[curr_swapchain_img_idx()], VK_IMAGE_LAYOUT_GENERAL,
                       &clear_value, 1, &clear_range);
  vk2::init::transition_image(cmd, swapchain_.imgs[curr_swapchain_img_idx()],
                              VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);

  VK_CHECK(vkEndCommandBuffer(cmd));

  submit_single_command_buf_to_graphics(cmd);
}

void VkRender2::on_gui() {}
