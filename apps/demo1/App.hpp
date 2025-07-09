#pragma once

#include <filesystem>

#include "Animation.hpp"
#include "Camera.hpp"
#include "Common.hpp"
#include "VkRender2.hpp"
struct GLFWwindow;

struct CharacterFSM {
  enum class State : u8 { Idle, Walk, Jump };
  const char* state_to_string(State state) {
    switch (state) {
      case State::Idle:
        return "idle";
      case State::Walk:
        return "walk";
      case State::Jump:
        return "jump";
      default:
        assert(0);
        return "";
    }
  }
  State curr_state{State::Idle};
  State prev_state{State::Idle};
  AnimationHandle animation_id;
  float blend_weight{};
  float jump_time_remaining{};
  void update([[maybe_unused]] float dt, float speed);

 private:
  State determine_state(float speed) {
    if (jump_time_remaining < 0.f && curr_state == State::Jump) {
      jump_time_remaining = 0.f;
      return prev_state;
    }
    if (jump_time_remaining > 0.f) {
      return State::Jump;
    }
    if (speed < .3f) {
      return State::Idle;
    }
    return State::Walk;
  }
};

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
  void on_file_drop(int count, const char** paths);
  void on_hide_mouse_change(bool new_hide_mouse);
  void on_cursor_event(vec2 pos) const;
  void on_imgui();
  std::vector<InstanceHandle> instances_;

  Camera cam_data;
  CameraController cam;
  GLFWwindow* window{};
  bool hide_mouse{false};
  std::filesystem::path resource_dir;
  std::filesystem::path local_models_dir;
  std::vector<InstanceHandle> scenes_;
  gfx::SceneDrawInfo info_{.light_color = {1., 1., 1.}, .fov_degrees = 70.f};
  vec3 light_dir_{2., -3.5, -2.};
  bool spin_light_{};
  float light_angle_{};
  float light_speed_{.002f};

  float dt{};

 private:
  bool running_{};
  void shutdown() const;
  void update(float dt);
  [[nodiscard]] float aspect_ratio() const;
  [[nodiscard]] uvec2 window_dims() const;
  void scene_node_imgui(gfx::Scene2& scene, int node, u32 obj_id);

  u32 add_instance(const std::filesystem::path& model, const mat4& transform = mat4{1});
  Camera character_cam_;
  CameraController character_cam_controller_{&character_cam_};
  void update_character(float dt);
  CharacterFSM character_fsm_;
  vec3 last_look_dir_{};
  static constexpr float min_speed_thresh{.001f};

  u32 character_instance_{};
  LoadedInstanceData* get_instance(u32 instance);
  int selected_node_{-1};
  int selected_obj_{-1};
};
