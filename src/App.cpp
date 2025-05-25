#include "App.hpp"

#include <nfd.h>

#include <fstream>
#include <tracy/Tracy.hpp>

#include "Camera.hpp"
#include "GLFW/glfw3.h"
#include "ResourceManager.hpp"
#include "Scene.hpp"
#include "VkRender2.hpp"
#include "core/Logger.hpp"
#include "imgui.h"
#include "util/CVar.hpp"
#include "vk2/Device.hpp"
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

using namespace gfx;
App::App(const InitInfo& info) : cam(cam_data, .1) {
  NFD_Init();

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

  glfwSetDropCallback(window, [](GLFWwindow* win, int count, const char** paths) {
    ((App*)glfwGetWindowUserPointer(win))->on_file_drop(count, paths);
  });

  glfwSetCursorPosCallback(window, [](GLFWwindow* win, double xpos, double ypos) {
    ((App*)glfwGetWindowUserPointer(win))->on_cursor_event({xpos, ypos});
  });

  Device::init({info.name, window, info.vsync});
  bool success;
  VkRender2::init(VkRender2::InitInfo{.window = window,
                                      .device = &Device::get(),
                                      .resource_dir = resource_dir,
                                      .name = info.name,
                                      .vsync = info.vsync},
                  success);
  if (!success) {
    return;
  }
  ResourceManager::init();
  local_models_dir = resource_dir / "local_models/";
  running_ = true;
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
  if (!running_) {
    return;
  }
  load_cam(cam_data);
  float last_time{};
  auto& renderer = VkRender2::get();
  // VkRender2::get().load_scene(local_models_dir / "ABeautifulGame.glb", false,
  //                             glm::scale(mat4{1}, vec3{10}));

  instances_.emplace_back(
      ResourceManager::get().load_model(local_models_dir / "Bistro_Godot_opt.glb"));
  // VkRender2::get().load_model(local_models_dir / "Bistro_Godot.glb", false);
  // std::filesystem::path env_tex = local_models_dir / "quarry_04_puresky_4k.hdr";
  // std::filesystem::path env_tex = local_models_dir / "immenstadter_horn_2k.hdr";

  // VkRender2::get().load_model(local_models_dir / "sponza.glb", false);

  instances_.emplace_back(
      ResourceManager::get().load_model(resource_dir / "models/Cube/glTF/Cube.gltf"));
  std::filesystem::path env_tex = local_models_dir / "newport_loft.hdr";
  // std::filesystem::path env_tex = "/home/tony/Downloads/quarry_04_puresky_4k.hdr";
  // std::filesystem::path env_tex = "/home/tony/Downloads/golden_gate_hills_4k.hdr";

  VkRender2::get().set_env_map(env_tex);
  while (running_ && !glfwWindowShouldClose(window)) {
    {
      ZoneScopedN("poll events");
      glfwPollEvents();
    }

    float curr_t = glfwGetTime();
    dt = curr_t - last_time;
    last_time = curr_t;

    renderer.new_frame();
    update(dt);
    on_imgui();
    for (auto& instance_handle : instances_) {
      auto* instance = ResourceManager::get().get_instance(instance_handle);
      if (recalc_global_transforms(instance->scene_graph_data)) {
        VkRender2::get().update_transforms(*instance);
      }
    }
    renderer.draw(info_);
  }

  save_cam(cam_data);
  shutdown();
}

void App::quit() const { glfwSetWindowShouldClose(window, true); }

void App::shutdown() const {
  ZoneScoped;
  // NOTE: destroying window first doesn't break the renderer for now. it's nice
  // since the window of the app closes faster
  glfwDestroyWindow(window);
  ResourceManager::shutdown();
  VkRender2::shutdown();
  Device::destroy();
  glfwTerminate();
  NFD_Quit();
}

void App::update(float dt) {
  ZoneScoped;
  cam.update_pos(window, dt);
  info_.view = cam_data.get_view();
  info_.view_pos = cam_data.pos;
  info_.light_dir = glm::normalize(light_dir_);
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
      VkRender2::get().set_imgui_enabled(!VkRender2::get().get_imgui_enabled());
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
  ZoneScoped;
  if (ImGui::Begin("app")) {
    static char filename[100];
    static std::string err_filename;
    static bool enter_clicked = false;
    static bool no_file_err = false;
    if (ImGui::InputText("Upload Model", filename, 100, ImGuiInputTextFlags_EnterReturnsTrue)) {
      enter_clicked = true;
      if (!std::filesystem::exists(filename)) {
        no_file_err = true;
        err_filename = filename;
      } else {
        ResourceManager::get().load_model(filename);
        no_file_err = false;
        enter_clicked = false;
      }
    }
    if (enter_clicked && no_file_err) {
      ImGui::Text("File not found: %s", err_filename.c_str());
    }
    if (ImGui::Button("Load glTF Model")) {
      nfdu8filteritem_t filters[] = {{"glTF", "glb,glTF"}};
      nfdopendialogu8args_t args = {};
      args.filterList = filters;
      args.filterCount = glm::countof(filters);
      nfdu8char_t* outpath{};
      if (NFD_OpenDialogU8_With(&outpath, &args) == NFD_OKAY) {
        ResourceManager::get().load_model(outpath);
        NFD_FreePathU8(outpath);
      }
    }

    if (ImGui::Button("Set IBL HDR Map")) {
      nfdu8filteritem_t filters[] = {{"HDR Map", "hdr"}};
      nfdopendialogu8args_t args = {};
      args.filterList = filters;
      args.filterCount = glm::countof(filters);
      nfdu8char_t* outpath{};
      if (NFD_OpenDialogU8_With(&outpath, &args) == NFD_OKAY) {
        VkRender2::get().set_env_map(outpath);
        NFD_FreePathU8(outpath);
      }
    }

    // TODO: frame time graph
    static std::vector<float> frame_times;
    frame_times.emplace_back(dt);
    if (frame_times.size() > 30) {
      frame_times.erase(frame_times.begin());
    }
    float tot = 0;
    for (auto t : frame_times) {
      tot += t;
    }
    float frame_time = tot / frame_times.size();
    ImGui::Text("Frame Time: %f ms/frame, FPS: %f", frame_time * 1000.f, 1.f / frame_time);
    if (ImGui::TreeNodeEx("Camera", ImGuiTreeNodeFlags_DefaultOpen)) {
      cam.on_imgui();
      ImGui::TreePop();
    }

    ImGui::DragFloat3("Sunlight Direction", &light_dir_.x, 0.01, -10.f, 10.f);
    ImGui::DragFloat("Light Speed", &light_speed_, .01);
    ImGui::Checkbox("Light Spin", &spin_light_);
    if (spin_light_) {
      light_angle_ += light_speed_;
      light_angle_ = glm::clamp(light_angle_, .0f, 360.f);

      light_dir_.x = std::sin(light_angle_);
      light_dir_.z = std::cos(light_angle_);
      // scene_data.light_dir =
    }

    ImGui::ColorEdit3("Sunlight Color", &info_.light_color.x, ImGuiColorEditFlags_Float);
    ImGui::DragFloat("Ambient Intensity", &info_.ambient_intensity);

    if (ImGui::Button("add sponza")) {
      static int offset = 1;
      ResourceManager::get().load_model(local_models_dir / "sponza.glb",
                                        glm::translate(mat4{1}, vec3{0, 0, offset * 40}));
      offset++;
    }

    if (ImGui::TreeNodeEx("Scene", ImGuiTreeNodeFlags_DefaultOpen)) {
      util::fixed_vector<u32, 8> to_delete;
      size_t i = 0;
      for (auto& instance_handle : instances_) {
        auto* instance = ResourceManager::get().get_instance(instance_handle);
        if (!instance || !instance->is_valid()) {
          continue;
        }
        auto* model = ResourceManager::get().get_model(instance->model_handle);
        ImGui::PushID(&instance_handle);

        if (ImGui::Button("X")) {
          if (!to_delete.full()) {
            to_delete.emplace_back(i);
          }
        }
        ImGui::SameLine();
        if (ImGui::TreeNodeEx("%s", ImGuiTreeNodeFlags_DefaultOpen, "%s",
                              model->path.string().c_str())) {
          scene_node_imgui(instance->scene_graph_data, 0);
          ImGui::TreePop();
        }
        ImGui::PopID();
        i++;
      }

      for (auto d : to_delete) {
        ResourceManager::get().remove_model(instances_[d]);
        if (instances_.size() > 1) {
          instances_[d] = instances_.back();
        }
        instances_.pop_back();
      }
      ImGui::TreePop();
    }

    CVarSystem::get().draw_imgui_editor();
  }
  ImGui::End();
}

void App::on_file_drop(int count, const char** paths) {
  for (int i = 0; i < count; i++) {
    LINFO("dropped file: {}", paths[i]);
    if (std::filesystem::exists(paths[i])) {
      ResourceManager::get().load_model(paths[i]);
    }
  }
}

void App::scene_node_imgui(gfx::Scene2& scene, int node) {
  auto it = scene.node_to_node_name_idx.find(node);
  ImGui::PushID(node);
  if (ImGui::TreeNode("%s", "%s",
                      it == scene.node_to_node_name_idx.end()
                          ? "Node"
                          : scene.node_names[it->second].c_str())) {
    auto& local_transform = scene.local_transforms[node];
    if (ImGui::DragFloat("z", &local_transform[3][2])) {
      mark_changed(scene, node);
    }
    for (int c = scene.hierarchies[node].first_child; c != -1;
         c = scene.hierarchies[c].next_sibling) {
      scene_node_imgui(scene, c);
    }
    ImGui::TreePop();
  }
  ImGui::PopID();
}
