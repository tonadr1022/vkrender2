#include "VkRender2.hpp"

#include <volk.h>
#include <vulkan/vulkan_core.h>

#include <cassert>
#include <filesystem>
#include <tracy/Tracy.hpp>

#include "Logger.hpp"
#include "vk2/BindlessResourceAllocator.hpp"
#include "vk2/Device.hpp"
#include "vk2/Initializers.hpp"
#include "vk2/PipelineManager.hpp"
#include "vk2/VkCommon.hpp"

namespace {

std::optional<std::filesystem::path> get_resource_dir() {
  auto curr_path = std::filesystem::current_path();
  while (curr_path.has_parent_path()) {
    auto resource_path = curr_path / "resources";
    if (std::filesystem::exists(resource_path)) {
      return resource_path;
    }
    curr_path = curr_path.parent_path();
  }
  return std::nullopt;
}

}  // namespace

using namespace vk2;

VkRender2::VkRender2(const InitInfo& info)
    : BaseRenderer(info, BaseRenderer::BaseInitInfo{.frames_in_flight = 2}) {
  auto resource_dir_result = get_resource_dir();
  if (!resource_dir_result.has_value()) {
    LCRITICAL("unable to locate 'resources' directory from current path: {}",
              std::filesystem::current_path().string());
    exit(1);
  }
  resource_dir = resource_dir_result.value();
  shader_dir = resource_dir / "shaders";

  vk2::BindlessResourceAllocator::init(device_, vk2::device().allocator());

  main_del_q.push([]() {
    vk2::PipelineManager::shutdown();
    vk2::BindlessResourceAllocator::shutdown();
  });

  vk2::PipelineManager::init(device_);

  VkPushConstantRange default_range{.stageFlags = VK_SHADER_STAGE_ALL, .offset = 0, .size = 128};
  // TODO: refactor
  VkDescriptorSetLayout main_set_layout = vk2::BindlessResourceAllocator::get().main_set_layout();
  main_set = vk2::BindlessResourceAllocator::get().main_set();

  VkPipelineLayoutCreateInfo pipeline_info{.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                                           .setLayoutCount = 1,
                                           .pSetLayouts = &main_set_layout,
                                           .pushConstantRangeCount = 1,
                                           .pPushConstantRanges = &default_range};
  VK_CHECK(vkCreatePipelineLayout(device_, &pipeline_info, nullptr, &default_pipeline_layout));
  main_del_q.push([this]() { vkDestroyPipelineLayout(device_, default_pipeline_layout, nullptr); });

  img_pipeline = PipelineManager::get().load_compute_pipeline(
      {get_shader_path("debug/clear_img.comp"), default_pipeline_layout});

  auto dims = window_dims();
  img = vk2::BindlessResourceAllocator::get().alloc_img_with_view(
      vk2::init::img_create_info_2d(
          VK_FORMAT_R8G8B8A8_UNORM, dims, false,
          VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT),
      init::subresource_range_whole(VK_IMAGE_ASPECT_COLOR_BIT), VK_IMAGE_VIEW_TYPE_2D);
}

void VkRender2::on_update() {}

void VkRender2::on_draw() {
  VkCommandBuffer cmd_buf = curr_frame().main_cmd_buffer;

  CmdEncoder cmd{cmd_buf};
  cmd.reset_and_begin();
  state.queue_transition(img->image(), VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                         VK_ACCESS_2_MEMORY_WRITE_BIT, VK_IMAGE_LAYOUT_GENERAL);

  cmd.bind_descriptor_set(VK_PIPELINE_BIND_POINT_COMPUTE, default_pipeline_layout, &main_set, 0);

  cmd.bind_compute(PipelineManager::get().get(img_pipeline)->pipeline);
  cmd.dispatch_compute(1, 1, 1);

  auto& swapchain_img = swapchain_.imgs[curr_swapchain_img_idx()];
  state.queue_transition(swapchain_.imgs[curr_swapchain_img_idx()],
                         VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                         VK_ACCESS_2_MEMORY_WRITE_BIT, VK_IMAGE_LAYOUT_GENERAL);
  state.flush(cmd_buf);

  VkClearColorValue clear_value;
  float flash = std::abs(std::sin(curr_frame_num() / 120.f));
  clear_value = {{0.0f, 0.0f, flash, 1.0f}};

  VkImageSubresourceRange clear_range =
      vk2::init::image_subresource_range(VK_IMAGE_ASPECT_COLOR_BIT);
  vkCmdClearColorImage(cmd_buf, swapchain_.imgs[curr_swapchain_img_idx()], VK_IMAGE_LAYOUT_GENERAL,
                       &clear_value, 1, &clear_range);
  state.queue_transition(swapchain_img, 0, VK_ACCESS_2_MEMORY_READ_BIT,
                         VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);
  state.flush(cmd_buf);
  VK_CHECK(vkEndCommandBuffer(cmd_buf));

  submit_single_command_buf_to_graphics(cmd_buf);
}

void VkRender2::on_gui() {}

std::string VkRender2::get_shader_path(const std::string& path) const { return shader_dir / path; }

VkRender2::~VkRender2() { vkDeviceWaitIdle(device_); }

void CmdEncoder::reset_and_begin() {
  auto info = vk2::init::command_buffer_begin_info();
  VK_CHECK(vkResetCommandBuffer(cmd_, 0));
  VK_CHECK(vkBeginCommandBuffer(cmd_, &info));
}

void CmdEncoder::dispatch_compute(u32 work_groups_x, u32 work_groups_y, u32 work_groups_z) {
  vkCmdDispatch(cmd_, work_groups_x, work_groups_y, work_groups_z);
}

void CmdEncoder::bind_compute(VkPipeline pipeline) {
  vkCmdBindPipeline(cmd_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
}
void CmdEncoder::bind_descriptor_set(VkPipelineBindPoint bind_point, VkPipelineLayout layout,
                                     VkDescriptorSet* set, u32 idx) {
  vkCmdBindDescriptorSets(cmd_, bind_point, layout, idx, 1, set, 0, nullptr);
}
