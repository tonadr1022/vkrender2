#include "Camera.hpp"

#include "imgui.h"

void CameraController::on_imgui() const {
  ImGui::Text("Pos %f %f %f", cam->pos.x, cam->pos.y, cam->pos.z);
  ImGui::Text("Front %f %f %f", cam->front.x, cam->front.y, cam->front.z);
  ImGui::Text("Velocity %f %f %f", velocity.x, velocity.y, velocity.z);
  if (ImGui::Button("Reset")) {
    *cam = {};
  }
}
