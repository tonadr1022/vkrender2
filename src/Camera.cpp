#include "Camera.hpp"

#include "imgui.h"

void CameraController::on_imgui() {
  ImGui::DragFloat3("Position", &cam.pos.x, .1f);
  if (ImGui::Button("Reset")) {
    cam = {};
  }
}
