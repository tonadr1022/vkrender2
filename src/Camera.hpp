#pragma once

#include <glm/gtc/quaternion.hpp>

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/io.hpp>

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
  [[nodiscard]] quat get_rotation_quat() const { return glm::quatLookAt(front, up); }

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
  void on_imgui() const;
  CameraController() = default;
  explicit CameraController(Camera* cam, float sensitivity = .1)
      : cam(cam), mouse_sensivity(sensitivity) {
    cam->update_vectors();
  }

  bool process_mouse(vec2 offset) const {
    assert(cam);
    offset *= mouse_sensivity;
    cam->yaw += offset.x;
    cam->pitch += offset.y;
    cam->pitch = glm::clamp(cam->pitch, -89.f, 89.f);
    cam->update_vectors();
    return !glm::all(glm::equal(offset, vec2{0}, glm::epsilon<float>()));
  }

  bool update_pos(float dt) {
    assert(cam);

    cam->update_vectors();
    auto get_key = [&](int key) { return Input::key_down(key); };
    vec3 acceleration{};
    bool accelerating{};

    if (get_key(GLFW_KEY_W) || get_key(GLFW_KEY_I)) {
      acceleration += cam->front;
      accelerating = true;
    }
    if (get_key(GLFW_KEY_S) || get_key(GLFW_KEY_K)) {
      acceleration -= cam->front;
      accelerating = true;
    }
    if (get_key(GLFW_KEY_A) || get_key(GLFW_KEY_J)) {
      acceleration -= cam->right;
      accelerating = true;
    }
    if (get_key(GLFW_KEY_D) || get_key(GLFW_KEY_L)) {
      acceleration += cam->right;
      accelerating = true;
    }

    if (get_key(GLFW_KEY_Y) || get_key(GLFW_KEY_R)) {
      acceleration += vec3(0, 1, 0);
      accelerating = true;
    }
    if (get_key(GLFW_KEY_H) || get_key(GLFW_KEY_F)) {
      acceleration += vec3(0, -1, 0);
      accelerating = true;
    }

    if (get_key(GLFW_KEY_B)) {
      acceleration_strength *= 1.1f;
      max_velocity *= 1.1f;
    }
    if (get_key(GLFW_KEY_V)) {
      acceleration_strength /= 1.1f;
      max_velocity /= 1.1f;
    }

    if (accelerating) {
      acceleration = glm::normalize(acceleration) * acceleration_strength;
    }

    velocity += acceleration * dt;
    velocity *= damping;

    velocity = glm::clamp(velocity, -max_velocity, max_velocity);

    cam->pos += velocity * dt;

    return accelerating || !glm::all(glm::equal(velocity, vec3{0}, glm::epsilon<float>()));
  }

  Camera* cam{};
  vec3 velocity{};
  vec3 max_velocity{10.f};
  float acceleration_strength{100.0f};
  float damping{0.9f};

  float mouse_sensivity{.1};
  float move_speed{10.};
};
