#pragma once

#include <vulkan/vulkan_core.h>

#include <filesystem>
#include <memory>

#include "Common.hpp"
#include "StateTracker.hpp"
#include "vk2/DeletionQueue.hpp"
#include "vk2/Swapchain.hpp"

struct GLFWwindow;
namespace tracy {
struct VkCtx;
}

namespace gfx {

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
  tracy::VkCtx* tracy_vk_ctx{};
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
  u64 semaphore_value_{0};
  bool submit_signaled_{};

 private:
  std::vector<VkCommandBuffer> active_cmd_buffers_;
  std::vector<VkCommandBuffer> free_cmd_buffers_;
  StateTracker state_tracker_;
  CmdPool cmd_pool_;
};

struct SceneDrawInfo {
  mat4 view;
  float fov_degees;
  vec3 view_pos;
  vec3 light_dir;
  vec3 light_color;
  float ambient_intensity{0.1};
  float fov_degrees{70.f};
};

class BaseRenderer {
 public:
  struct InitInfo {
    GLFWwindow* window;
    std::filesystem::path resource_dir;
    const char* name = "App";
    bool vsync{true};
    // VkPresentModeKHR present_mode{VK_PRESENT_MODE_FIFO_KHR};
    std::function<void()> on_gui_callback;
  };

  BaseRenderer(const BaseRenderer&) = delete;
  BaseRenderer(BaseRenderer&& other) = delete;
  BaseRenderer operator=(BaseRenderer&&) = delete;
  BaseRenderer operator=(const BaseRenderer&) = delete;

  virtual ~BaseRenderer();

  void draw(const SceneDrawInfo& info);
  void new_frame();

  bool draw_imgui{true};

  [[nodiscard]] u64 curr_frame_num() const { return curr_frame_num_; }
  [[nodiscard]] u64 curr_frame_in_flight_num() const;

 protected:
  explicit BaseRenderer(const InitInfo& info);
  virtual void on_update();
  virtual void on_draw(const SceneDrawInfo& info);
  virtual void on_imgui();
  virtual void on_resize();
  void render_imgui(VkCommandBuffer cmd, uvec2 draw_extent, VkImageView target_img_view);

  // QueueFamilies queues_;
  GLFWwindow* window_;
  [[nodiscard]] vk2::Swapchain::Status curr_frame_swapchain_status() const;
  std::vector<PerFrameData> per_frame_data_;
  PerFrameData& curr_frame();
  std::unique_ptr<QueueManager> transfer_queue_manager_;
  std::filesystem::path resource_dir_;

  // begin non owning
  VkDevice device_;

  // end non-owning

  [[nodiscard]] u32 curr_swapchain_img_idx() const { return curr_swapchain_img_idx_; }

  [[nodiscard]] uvec2 window_dims() const;
  [[nodiscard]] float aspect_ratio() const;

 private:
  std::function<void()> on_gui_callback_;
  vk2::DeletionQueue app_del_queue_;
  vk2::Swapchain::Status curr_frame_swapchain_status_;
  bool resize_swapchain_req_{};
  u32 curr_swapchain_img_idx_{};
  bool initialized_{false};
  u64 curr_frame_num_{};
};

}  // namespace gfx
