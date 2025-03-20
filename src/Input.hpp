#pragma once

#include <array>

#include "Common.hpp"
#include "GLFW/glfw3.h"

struct Input {
  static void update() { keys_pressed.fill(false); }
  static void set_key_down(int key, bool down) {
    keys_down[key] = down;
    keys_pressed[key] = down;
  }
  static bool key_down(int key) { return keys_down[key]; }
  static bool key_pressed(int key) { return keys_pressed[key]; }

  static bool mod_down(int mod) { return mod_state & mod; }
  static inline std::array<bool, GLFW_KEY_LAST> keys_pressed{};
  static inline std::array<bool, GLFW_KEY_LAST> keys_down{};
  static inline u64 mod_state{};
};
