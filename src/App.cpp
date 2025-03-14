#include "App.hpp"

#include <volk.h>
#include <vulkan/vk_enum_string_helper.h>
#include <vulkan/vulkan_core.h>

#include "GLFW/glfw3.h"
#include "Logger.hpp"
#include "VkBootstrap.h"
#include "tracy/Tracy.hpp"
#include "vk2/Device.hpp"
#include "vk2/Initializers.hpp"
#include "vk2/Swapchain.hpp"
#include "vk2/VkCommon.hpp"

namespace {

#ifdef DEBUG_CALLBACK_ENABLED
VKAPI_ATTR VkBool32 VKAPI_CALL
debug_callback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
               VkDebugUtilsMessageTypeFlagsEXT messageType,
               const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData, void*) {
  const auto* ms = vkb::to_string_message_severity(messageSeverity);
  const auto* mt = vkb::to_string_message_type(messageType);
  if (messageType & VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT) {
    LERROR("[{}: {}] - {}\n{}\n", ms, mt, pCallbackData->pMessageIdName, pCallbackData->pMessage);
  } else {
    LERROR("[{}: {}]\n{}\n", ms, mt, pCallbackData->pMessage);
  }

  return VK_FALSE;
}

#endif

}  // namespace

BaseRenderer::BaseRenderer(const InitInfo& info, const BaseInitInfo& base_info) {
  ZoneScoped;
  frames_in_flight_ = std::min(base_info.frames_in_flight, max_frames_in_flight);
  {
    ZoneScopedN("volk init");
    VK_CHECK(volkInitialize());
  }
  {
    ZoneScopedN("instance init");
    vkb::InstanceBuilder instance_builder;
    instance_builder
        .set_minimum_instance_version(vk2::min_api_version_major, vk2::min_api_version_minor, 0)
        .set_app_name(info.name)
        .require_api_version(1, 3, 0);

#ifdef DEBUG_CALLBACK_ENABLED
    instance_builder.set_debug_callback(debug_callback);
#endif
#ifdef VALIDATION_LAYERS_ENABLED
    instance_builder.request_validation_layers(true);
#endif

    auto instance_ret = instance_builder.build();
    if (!instance_ret) {
      LCRITICAL("Failed to acquire Vulkan Instance: {}", instance_ret.error().message());
      exit(1);
    }
    instance_ = instance_ret.value();
    app_del_queue_.push([this]() { vkb::destroy_instance(instance_); });
  }

  {
    ZoneScopedN("volk");
    volkLoadInstance(instance_.instance);
    app_del_queue_.push([]() { volkFinalize(); });
  }

  {
    ZoneScopedN("glfw init");
    if (!glfwInit()) {
      LCRITICAL("Failed to initialize GLFW");
      exit(1);
    }
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);
    glfwWindowHint(GLFW_DECORATED, info.decorate);
    glfwWindowHint(GLFW_MAXIMIZED, info.maximize);
    glfwWindowHint(GLFW_FOCUSED, GLFW_FALSE);
    window_ = glfwCreateWindow(info.width, info.height, info.name, nullptr, nullptr);
    if (!window_) {
      LCRITICAL("Failed to create window");
      exit(1);
    }
    glfwCreateWindowSurface(instance_.instance, window_, nullptr, &surface_);
    if (!surface_) {
      LCRITICAL("Failed to create surface");
      exit(1);
    }
  }

  app_del_queue_.push([this]() {
    vkb::destroy_surface(instance_, surface_);
    glfwDestroyWindow(window_);
    glfwTerminate();
  });

  vk2::Device::init({.instance = instance_, .surface = surface_});
  app_del_queue_.push([]() { vk2::Device::destroy(); });
  device_ = vk2::device().device();

  auto graphics_queue_result = vk2::device().vkb_device().get_queue(vkb::QueueType::graphics);
  if (!graphics_queue_result.value()) {
    LCRITICAL("graphics queue unavailable");
    exit(1);
  }
  queues_.graphics_queue = graphics_queue_result.value();

  auto graphics_queue_idx_res =
      vk2::device().vkb_device().get_queue_index(vkb::QueueType::graphics);
  if (!graphics_queue_idx_res) {
    LCRITICAL("graphics queue idx unavailable");
    exit(1);
  }
  queues_.graphics_queue_idx = graphics_queue_idx_res.value();

  {
    ZoneScopedN("init swapchain");
    swapchain_.init({.phys_device = vk2::device().phys_device(),
                     .device = vk2::device().device(),
                     .surface = surface_,
                     .present_mode = info.present_mode,
                     .queue_idx = queues_.graphics_queue_idx,
                     .requested_resize = false},
                    vk2::device().get_swapchain_format());
    swapchain_.recreate_img_views(vk2::device().device());
  }
  app_del_queue_.push([this]() { swapchain_.destroy(vk2::device().device()); });

  {
    ZoneScopedN("init per frame");
    per_frame_data_.resize(frames_in_flight_);
    for (auto& frame : per_frame_data_) {
      frame.cmd_pool = vk2::device().create_command_pool(queues_.graphics_queue_idx);
      frame.main_cmd_buffer = vk2::device().create_command_buffer(frame.cmd_pool);
      frame.render_fence = vk2::device().create_fence();
      frame.swapchain_semaphore = vk2::device().create_semaphore();
      frame.render_semaphore = vk2::device().create_semaphore();
    }
  }

  app_del_queue_.push([this]() {
    for (const auto& frame : per_frame_data_) {
      auto& d = vk2::device();
      d.destroy_fence(frame.render_fence);
      d.destroy_semaphore(frame.render_semaphore);
      d.destroy_semaphore(frame.swapchain_semaphore);
      d.destroy_command_pool(frame.cmd_pool);
    }
  });

  initialized_ = true;
}

void BaseRenderer::run() {
  while (!glfwWindowShouldClose(window_)) {
    glfwPollEvents();
    draw();
  }
}

BaseRenderer::~BaseRenderer() {
  vkDeviceWaitIdle(device_);
  app_del_queue_.flush();
}
void BaseRenderer::on_draw() {}
void BaseRenderer::on_gui() {}
void BaseRenderer::on_update() {}

void BaseRenderer::draw() {
  curr_frame_swapchain_status_ = swapchain_.update({.phys_device = vk2::device().phys_device(),
                                                    .device = vk2::device().device(),
                                                    .surface = surface_,
                                                    .present_mode = swapchain_.present_mode,
                                                    .queue_idx = queues_.graphics_queue_idx,
                                                    .requested_resize = resize_swapchain_req_});
  resize_swapchain_req_ = false;
  if (curr_frame_swapchain_status_ == vk2::Swapchain::Status::NotReady) {
    return;
  }
  if (curr_frame_swapchain_status_ == vk2::Swapchain::Status::Resized) {
    swapchain_.recreate_img_views(device_);
  }

  // wait for fence
  u64 timeout = 1000000000;
  vkWaitForFences(device_, 1, &curr_frame().render_fence, true, timeout);

  // acquire next image
  VkResult get_swapchain_img_res =
      vkAcquireNextImageKHR(device_, swapchain_.swapchain, timeout,
                            curr_frame().swapchain_semaphore, nullptr, &curr_swapchain_img_idx_);
  if (get_swapchain_img_res == VK_ERROR_OUT_OF_DATE_KHR) {
    return;
  }

  // reset fence once drawing, else return
  VK_CHECK(vkResetFences(device_, 1, &curr_frame().render_fence));
  on_draw();

  {
    ZoneScopedN("presentation");
    VkResult res;
    VkPresentInfoKHR info{.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
                          .waitSemaphoreCount = 1,
                          .pWaitSemaphores = &curr_frame().render_semaphore,
                          .swapchainCount = 1,
                          .pSwapchains = &swapchain_.swapchain,
                          .pImageIndices = &curr_swapchain_img_idx_,
                          .pResults = &res};
    VkResult present_result = vkQueuePresentKHR(queues_.graphics_queue, &info);
    if (present_result == VK_ERROR_OUT_OF_DATE_KHR || present_result == VK_SUBOPTIMAL_KHR) {
    } else {
      VK_CHECK(present_result);
    }

    curr_frame_num_++;
  }
}

void BaseRenderer::quit() { glfwSetWindowShouldClose(window_, true); }

vk2::Swapchain::Status BaseRenderer::curr_frame_swapchain_status() const {
  return curr_frame_swapchain_status_;
}
PerFrameData& BaseRenderer::curr_frame() { return per_frame_data_[curr_frame_num_ % 2]; }

void BaseRenderer::submit_single_command_buf_to_graphics(VkCommandBuffer cmd) {
  ZoneScoped;
  {
    ZoneScopedN("queue submit");
    // wait for swapchain to be ready
    auto wait_info = vk2::init::semaphore_submit_info(
        curr_frame().swapchain_semaphore, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT);
    // signal the render semaphore so presentation can wait on it
    auto signal_info = vk2::init::semaphore_submit_info(curr_frame().render_semaphore,
                                                        VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT);
    auto cmd_buf_submit_info = vk2::init::command_buffer_submit_info(cmd);
    auto submit = vk2::init::queue_submit_info(SPAN1(cmd_buf_submit_info), SPAN1(wait_info),
                                               SPAN1(signal_info));
    VK_CHECK(vkQueueSubmit2(queues_.graphics_queue, 1, &submit, curr_frame().render_fence));
  }
}
