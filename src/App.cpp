#include "App.hpp"

#include <fstream>

#include "Camera.hpp"
#include "GLFW/glfw3.h"
#include "Input.hpp"
#include "Logger.hpp"
#include "VkRender2.hpp"
#include "imgui.h"
#include "util/CVar.hpp"
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

App::App(const InitInfo& info) : cam(cam_data, .1) {
  auto resource_dir_ret = get_resource_dir();
  if (!resource_dir_ret.has_value()) {
    LCRITICAL("failed to find resource directory");
    exit(1);
  }
  resource_dir = resource_dir_ret.value();

  if (!glfwInit()) {
    LCRITICAL("glfwInit failed");
    exit(1);
  }
  glfwSetErrorCallback([](int error_code, const char* description) {
    LERROR("glfw error: {}, {}", error_code, description);
  });
  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
  glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);
  glfwWindowHint(GLFW_DECORATED, info.decorate);
  glfwWindowHint(GLFW_MAXIMIZED, info.maximize);
  window = glfwCreateWindow(info.width, info.height, info.name, nullptr, nullptr);
  if (!window) {
    LCRITICAL("Failed to create window");
    exit(1);
  }

  glfwSetWindowUserPointer(window, this);
  glfwSetKeyCallback(window, []([[maybe_unused]] GLFWwindow* win, [[maybe_unused]] int key,
                                [[maybe_unused]] int scancode, [[maybe_unused]] int action,
                                [[maybe_unused]] int mods) {
    if (ImGui::GetIO().WantCaptureKeyboard) return;
    Input::set_key_down(key, action == GLFW_PRESS || action == GLFW_REPEAT);
    ((App*)glfwGetWindowUserPointer(win))->on_key_event(key, scancode, action, mods);
  });

  glfwSetCursorPosCallback(window, [](GLFWwindow* win, double xpos, double ypos) {
    ((App*)glfwGetWindowUserPointer(win))->on_cursor_event({xpos, ypos});
  });

  VkRender2::init(VkRender2::InitInfo{.window = window,
                                      .resource_dir = resource_dir,
                                      .name = info.name,
                                      .vsync = info.vsync,
                                      .on_gui_callback = [this]() { this->on_imgui(); }});
  local_models_dir = resource_dir / "local_models/";
}

namespace {

std::filesystem::path cache_dir{"./.cache"};
std::filesystem::path cam_data_path{cache_dir / "camera.bin"};

void save_cam(const Camera& cam) {
  if (!std::filesystem::exists(cache_dir)) {
    std::filesystem::create_directory(cache_dir);
  }
  std::ofstream file(cam_data_path, std::ios::binary);
  if (!file.is_open()) {
    LERROR("failed to save camera");
    return;
  }
  file.write((const char*)&cam, sizeof(Camera));
}

void load_cam(Camera& cam) {
  if (std::filesystem::exists(cam_data_path)) {
    std::ifstream file(cam_data_path, std::ios::binary);
    if (!file.is_open()) {
      LERROR("failed to load camera data");
      return;
    }
    file.read((char*)&cam, sizeof(Camera));
  }
}

}  // namespace

void App::run() {
  load_cam(cam_data);
  float last_time{};
  // VkRender2::get().load_scene(local_models_dir / "sponza.glb", false);
  // VkRender2::get().load_scene("/Users/tony/models/Models/ABeautifulGame/glTF/ABeautifulGame.gltf");
  VkRender2::get().load_scene(local_models_dir / "ABeautifulGame.glb", false);
  VkRender2::get().load_scene(local_models_dir / "DamagedHelmet.glb", false);

  while (!glfwWindowShouldClose(window)) {
    glfwPollEvents();
    float curr_t = glfwGetTime();
    float dt = curr_t - last_time;
    last_time = curr_t;
    update(dt);

    mat4 proj = glm::perspective(glm::radians(70.f), aspect_ratio(), 0.1f, 1000.f);
    VkRender2::get().draw({.view = cam_data.get_view(), .proj = proj, .view_pos = cam_data.pos});
  }

  save_cam(cam_data);
  shutdown();
}

void App::quit() const { glfwSetWindowShouldClose(window, true); }

void App::shutdown() const {
  VkRender2::shutdown();
  glfwDestroyWindow(window);
  glfwTerminate();
}

void App::update(float dt) {
  cam.update_pos(window, dt);
  // static glm::quat rot = glm::quat(1, 0, 0, 0);
  // glm::quat delta_rot = glm::angleAxis(dt, glm::vec3(0., 1., 0.));
  // rot = glm::normalize(delta_rot * rot);  // Accumulate rotation
  // cam_data.set_rotation(rot);
}

void App::on_key_event([[maybe_unused]] int key, [[maybe_unused]] int scancode,
                       [[maybe_unused]] int action, [[maybe_unused]] int mods) {
  if (action == GLFW_PRESS) {
    if (key == GLFW_KEY_ESCAPE) {
      on_hide_mouse_change(!hide_mouse);
    }
    if (key == GLFW_KEY_G && mods & GLFW_MOD_ALT) {
      VkRender2::get().draw_imgui = !VkRender2::get().draw_imgui;
    }
  }
}

void App::on_hide_mouse_change(bool new_hide_mouse) {
  hide_mouse = new_hide_mouse;
  glfwSetInputMode(window, GLFW_CURSOR, hide_mouse ? GLFW_CURSOR_DISABLED : GLFW_CURSOR_NORMAL);
}

// TODO: move
namespace {
bool first_mouse{true};
vec2 last_pos{};
}  // namespace
void App::on_cursor_event(vec2 pos) {
  if (first_mouse) {
    first_mouse = false;
    last_pos = pos;
    return;
  }
  vec2 offset = {pos.x - last_pos.x, last_pos.y - pos.y};
  last_pos = pos;
  if (hide_mouse) {
    cam.process_mouse(offset);
  }
}

float App::aspect_ratio() const {
  auto dims = window_dims();
  return (float)dims.x / (float)dims.y;
}

uvec2 App::window_dims() const {
  int x, y;
  glfwGetWindowSize(window, &x, &y);
  return {x, y};
}

void App::on_imgui() {
  ImGui::Begin("hello");
  CVarSystem::get().draw_imgui_editor();
  for (auto scene_handle : scenes_) {
    auto& scene = VkRender2::get().loaded_dynamic_scenes_[(int)scene_handle.get()];
    for (auto& c : scene.scene_graph_data.cameras) {
      if (ImGui::Button("set cam")) {
        cam_data = c;
      }
    }
  }
  ImGui::End();
}
