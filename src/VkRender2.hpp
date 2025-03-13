#pragma once

#include "App.hpp"

class VkRender2 : public BaseRenderer {
 public:
  explicit VkRender2(const InitInfo& info);
  void on_update() override;
  void on_draw() override;
  void on_gui() override;
};
