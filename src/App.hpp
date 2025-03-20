#pragma once

#include <filesystem>

#include "Camera.hpp"
#include "Common.hpp"
struct GLFWwindow;

struct App {
  struct InitInfo {
    const char* name = "App";
    u32 width{800};
    u32 height{800};
    bool maximize{false};
    bool decorate{true};
    bool vsync{true};
  };
  explicit App(const InitInfo& info);
  void run();
  void quit() const;
  void on_key_event([[maybe_unused]] int key, [[maybe_unused]] int scancode,
                    [[maybe_unused]] int action, [[maybe_unused]] int mods);
  void on_hide_mouse_change(bool new_hide_mouse);
  void on_cursor_event(vec2 pos);
  void on_imgui();

  Camera cam_data;
  CameraController cam;
  GLFWwindow* window{};
  bool hide_mouse{false};
  std::filesystem::path resource_dir;

 private:
  void shutdown() const;
  void update(float dt);
  [[nodiscard]] float aspect_ratio() const;
  [[nodiscard]] uvec2 window_dims() const;
};
