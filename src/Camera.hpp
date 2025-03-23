#pragma once

#include <glm/gtc/quaternion.hpp>

#include "Common.hpp"
#include "Input.hpp"

struct Camera {
  vec3 pos{0, 0, 5};
  vec3 front{0, 0, -1};
  vec3 right{1, 0, 0};
  vec3 up{0, 1, 0};
  float pitch{}, yaw{-90.f};
  [[nodiscard]] mat4 get_view() const { return glm::lookAt(pos, pos + front, {0, 1, 0}); }

  void set_rotation(quat rot) {
    glm::vec3 forward = rot * glm::vec3(0, 0, -1);
    pitch = glm::degrees(glm::asin(glm::clamp(forward.y, -1.f, 1.f)));
    yaw = glm::degrees(glm::atan(-forward.z, forward.x));
    front = glm::normalize(forward);
    right = glm::normalize(glm::cross(front, {0, 1, 0}));
    up = glm::normalize(glm::cross(right, front));
  }

  void update_vectors() {
    glm::vec3 dir;
    dir.x = glm::cos(glm::radians(yaw)) * glm::cos(glm::radians(pitch));
    dir.y = glm::sin(glm::radians(pitch));
    dir.z = glm::sin(glm::radians(yaw)) * glm::cos(glm::radians(pitch));
    front = glm::normalize(dir);
    right = glm::normalize(glm::cross(front, {0, 1, 0}));
    up = glm::normalize(glm::cross(right, front));
  }
};

struct CameraController {
  explicit CameraController(Camera& cam, float sensitivity = .1)
      : cam(cam), sensivity(sensitivity) {}

  void process_mouse(vec2 offset) {
    offset *= sensivity;
    cam.yaw += offset.x;
    cam.pitch += offset.y;
    cam.pitch = glm::clamp(cam.pitch, -89.f, 89.f);
    cam.update_vectors();
  }

  void update_pos(GLFWwindow*, float dt) {
    auto get_key = [&](int key) { return Input::key_down(key); };
    vec3 offset{};
    if (get_key(GLFW_KEY_W) || get_key(GLFW_KEY_I)) {
      offset += cam.front;
    }
    if (get_key(GLFW_KEY_S) || get_key(GLFW_KEY_K)) {
      offset -= cam.front;
    }
    if (get_key(GLFW_KEY_A) || get_key(GLFW_KEY_J)) {
      offset -= cam.right;
    }
    if (get_key(GLFW_KEY_D) || get_key(GLFW_KEY_L)) {
      offset += cam.right;
    }
    float vert{};
    if (get_key(GLFW_KEY_Y) || get_key(GLFW_KEY_R)) {
      vert += 1;
    }
    if (get_key(GLFW_KEY_H) || get_key(GLFW_KEY_F)) {
      vert -= 1;
    }
    cam.pos += offset * move_speed * dt;
    cam.pos.y += vert * move_speed * dt;
  }

  Camera& cam;
  float sensivity{.1};
  float move_speed{10.};
};
