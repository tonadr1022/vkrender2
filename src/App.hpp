#pragma once

#include <filesystem>

#include "Camera.hpp"
#include "Common.hpp"
#include "Scene.hpp"
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
  std::filesystem::path local_models_dir;
  std::vector<SceneHandle> scenes_;
  struct SceneData {
    vec3 light_dir{0.5, 0.5, 0.5};
    vec3 light_color{1.0, 1.0, 1.0};
  } scene_data;
  float fov_degrees{70.f};

 private:
  void shutdown() const;
  void update(float dt);
  [[nodiscard]] float aspect_ratio() const;
  [[nodiscard]] uvec2 window_dims() const;
};
