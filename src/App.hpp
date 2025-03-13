#pragma once

#include <vulkan/vulkan_core.h>

#include "Common.hpp"
#include "VkBootstrap.h"
#include "vk2/DeletionQueue.hpp"
#include "vk2/Swapchain.hpp"

struct GLFWwindow;

struct QueueFamilies {
  VkQueue graphics_queue{};
  VkQueue compute_queue{};
  VkQueue transfer_queue{};
  u32 graphics_queue_idx{};
  u32 compute_queue_idx{};
  u32 transfer_queue_idx{};
};

struct PerFrameData {
  VkCommandPool cmd_pool;
  VkCommandBuffer main_cmd_buffer;
  VkSemaphore swapchain_semaphore, render_semaphore;
  VkFence render_fence;
};

class BaseRenderer {
 public:
  struct InitInfo {
    const char* name = "App";
    u32 width{1600};
    u32 height{900};
    bool maximize{false};
    bool decorate{true};
    bool vsync{true};
    VkPresentModeKHR present_mode{VK_PRESENT_MODE_FIFO_KHR};
  };

  BaseRenderer(const BaseRenderer&) = delete;
  BaseRenderer(BaseRenderer&& other) = delete;
  BaseRenderer operator=(BaseRenderer&&) = delete;
  BaseRenderer operator=(const BaseRenderer&) = delete;

  virtual ~BaseRenderer();
  void run();

  static constexpr u32 max_frames_in_flight{3};

 protected:
  struct BaseInitInfo {
    u32 frames_in_flight{2};
  };
  explicit BaseRenderer(const InitInfo& info, const BaseInitInfo& base_info);
  virtual void on_update();
  virtual void on_draw();
  virtual void on_gui();

  VkSurfaceCapabilitiesKHR surface_caps_{};
  QueueFamilies queues_;
  vkb::Instance instance_;
  VkSurfaceKHR surface_;
  GLFWwindow* window_;
  vk2::Swapchain swapchain_;
  [[nodiscard]] vk2::Swapchain::Status curr_frame_swapchain_status() const;
  u32 frames_in_flight_{2};
  std::vector<PerFrameData> per_frame_data_;
  PerFrameData& curr_frame();
  void quit();

  // begin non owning
  VkDevice device_;

  // end non-owning

  void submit_single_command_buf_to_graphics(VkCommandBuffer cmd);

  [[nodiscard]] u32 curr_swapchain_img_idx() const { return curr_swapchain_img_idx_; }
  [[nodiscard]] u64 curr_frame_num() const { return curr_frame_num_; }

 private:
  vk2::DeletionQueue app_del_queue_;
  VkDebugUtilsMessengerEXT debug_messenger_;
  vk2::Swapchain::Status curr_frame_swapchain_status_;
  u32 curr_swapchain_img_idx_;
  bool initialized_{false};
  u64 curr_frame_num_{};
  void draw();
};
