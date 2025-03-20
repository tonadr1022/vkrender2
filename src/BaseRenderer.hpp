#pragma once

#include <vulkan/vulkan_core.h>

#include <filesystem>
#include <memory>

#include "Common.hpp"
#include "StateTracker.hpp"
#include "VkBootstrap.h"
#include "vk2/DeletionQueue.hpp"
#include "vk2/Swapchain.hpp"

struct GLFWwindow;
namespace tracy {
struct VkCtx;
}

#ifndef NDEBUG
#define VALIDATION_LAYERS_ENABLED 1
#define DEBUG_CALLBACK_ENABLED 1
#endif

struct QueueFamilies {
  VkQueue graphics_queue{};
  VkQueue compute_queue{};
  VkQueue transfer_queue{};
  u32 graphics_queue_idx{UINT32_MAX};
  u32 compute_queue_idx{UINT32_MAX};
  u32 transfer_queue_idx{UINT32_MAX};
  bool is_unified_graphics_transfer{};
};

struct CmdPool {
  explicit CmdPool(VkCommandPool pool) : pool_(pool) {}
  ~CmdPool();

  CmdPool& operator=(const CmdPool&) = delete;
  CmdPool(const CmdPool&) = delete;
  CmdPool(CmdPool&& old) noexcept;
  CmdPool& operator=(CmdPool&& old) noexcept;

  [[nodiscard]] VkCommandPool pool() const { return pool_; }

 private:
  VkCommandPool pool_;
};

struct PerFrameData {
  VkCommandPool cmd_pool;
  VkCommandBuffer main_cmd_buffer;
  VkSemaphore swapchain_semaphore, render_semaphore;
  VkFence render_fence;
};

struct QueueManager {
  explicit QueueManager(u32 queue_idx, u32 cmd_buffer_cnt = 1);
  QueueManager() = delete;
  ~QueueManager();

  QueueManager(QueueManager&&) = delete;
  QueueManager& operator=(QueueManager&&) = delete;
  QueueManager(const QueueManager&) = delete;
  QueueManager& operator=(const QueueManager&) = delete;

  VkCommandBuffer get_cmd_buffer();

  // TODO: bad here
  VkSemaphore submit_semaphore_;
  bool submit_signaled_{};

 private:
  std::vector<VkCommandBuffer> active_cmd_buffers_;
  std::vector<VkCommandBuffer> free_cmd_buffers_;
  StateTracker state_tracker_;
  CmdPool cmd_pool_;
};

struct SceneDrawInfo {
  mat4 view;
  mat4 proj;
};

class BaseRenderer {
 public:
  struct InitInfo {
    GLFWwindow* window;
    std::filesystem::path resource_dir;
    const char* name = "App";
    bool vsync{true};
    VkPresentModeKHR present_mode{VK_PRESENT_MODE_FIFO_KHR};
    std::function<void()> on_gui_callback;
  };

  BaseRenderer(const BaseRenderer&) = delete;
  BaseRenderer(BaseRenderer&& other) = delete;
  BaseRenderer operator=(BaseRenderer&&) = delete;
  BaseRenderer operator=(const BaseRenderer&) = delete;

  virtual ~BaseRenderer();

  static constexpr u32 max_frames_in_flight{3};
  void draw(const SceneDrawInfo& info);

  bool draw_imgui{true};

 protected:
  struct BaseInitInfo {
    u32 frames_in_flight{2};
  };
  explicit BaseRenderer(const InitInfo& info, const BaseInitInfo& base_info);
  virtual void on_update();
  virtual void on_draw(const SceneDrawInfo& info);
  virtual void on_gui();
  virtual void on_resize();
  void render_imgui(VkCommandBuffer cmd, uvec2 draw_extent, VkImageView target_img_view);

  QueueFamilies queues_;
  vkb::Instance instance_;
  VkSurfaceKHR surface_;
  GLFWwindow* window_;
  vk2::Swapchain swapchain_;
  [[nodiscard]] vk2::Swapchain::Status curr_frame_swapchain_status() const;
  u32 frames_in_flight_{2};
  std::vector<PerFrameData> per_frame_data_;
  PerFrameData& curr_frame();
  std::unique_ptr<QueueManager> transfer_queue_manager_;
  std::filesystem::path resource_dir_;

  // begin non owning
  VkDevice device_;

  // end non-owning

  void submit_single_command_buf_to_graphics(VkCommandBuffer cmd);

  [[nodiscard]] u32 curr_swapchain_img_idx() const { return curr_swapchain_img_idx_; }
  [[nodiscard]] u64 curr_frame_num() const { return curr_frame_num_; }

  uvec2 window_dims();

 private:
#ifdef TRACY_ENABLE
  tracy::VkCtx* tracy_vk_ctx_{};
#endif
  std::function<void()> on_gui_callback_;
  vk2::DeletionQueue app_del_queue_;
  vk2::Swapchain::Status curr_frame_swapchain_status_;
  bool resize_swapchain_req_{};
  u32 curr_swapchain_img_idx_{};
  bool initialized_{false};
  u64 curr_frame_num_{};
};
