#include "App.hpp"

#include "GLFW/glfw3.h"
#include "Logger.hpp"
#include "VkRender2.hpp"

App::App(const InitInfo& info) : cam(cam_data, .1) {
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
  // glfwWindowHint(GLFW_FOCUSED, GLFW_FALSE);
  window = glfwCreateWindow(info.width, info.height, info.name, nullptr, nullptr);
  if (!window) {
    LCRITICAL("Failed to create window");
    exit(1);
  }
  glfwSetWindowUserPointer(window, this);
  glfwSetKeyCallback(window, []([[maybe_unused]] GLFWwindow* win, [[maybe_unused]] int key,
                                [[maybe_unused]] int scancode, [[maybe_unused]] int action,
                                [[maybe_unused]] int mods) {
    ((App*)glfwGetWindowUserPointer(win))->on_key_event(key, scancode, action, mods);
  });
  glfwSetCursorPosCallback(window, [](GLFWwindow* win, double xpos, double ypos) {
    ((App*)glfwGetWindowUserPointer(win))->on_cursor_event({xpos, ypos});
  });

  VkRender2::init(VkRender2::InitInfo{.window = window, .name = info.name, .vsync = info.vsync});
}
void App::run() {
  float last_time{};
  while (!glfwWindowShouldClose(window)) {
    glfwPollEvents();
    float curr_t = glfwGetTime();
    float dt = curr_t - last_time;
    last_time = curr_t;
    update(dt);

    mat4 proj = glm::perspective(glm::radians(70.f), aspect_ratio(), 0.1f, 1000.f);
    VkRender2::get().draw({.view = cam_data.get_view(), .proj = proj});
  }

  shutdown();
}

void App::quit() const { glfwSetWindowShouldClose(window, true); }

void App::shutdown() const {
  VkRender2::shutdown();
  glfwDestroyWindow(window);
  glfwTerminate();
}
void App::update(float dt) { cam.update_pos(window, dt); }

void App::on_key_event([[maybe_unused]] int key, [[maybe_unused]] int scancode,
                       [[maybe_unused]] int action, [[maybe_unused]] int mods) {
  if (action == GLFW_PRESS) {
    if (key == GLFW_KEY_ESCAPE) {
      on_hide_mouse_change(!hide_mouse);
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
