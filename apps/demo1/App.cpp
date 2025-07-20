#include "App.hpp"

#include <nfd.h>

#include <fstream>
#include <iostream>
#include <tracy/Tracy.hpp>

#include "AnimationManager.hpp"
#include "Camera.hpp"
#include "GLFW/glfw3.h"
#include "ResourceManager.hpp"
#include "Scene.hpp"
#include "VkRender2.hpp"
#include "core/Logger.hpp"

// clang-format off
#include "glm/gtc/quaternion.hpp"
#include "imgui.h"
#include "ImGuizmo.h"
// clang-format on

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
App::App(const InitInfo& info) : cam(&cam_data, .1) {
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

  Device::init({info.name, window, info.vsync, info.enable_validation_layers});
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
  AnimationManager::init();
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

template <typename T>
int compare_vec(std::span<T> a, std::span<T> b) {
  for (size_t i = 0; i < a.size(); i++) {
    if (a[i] != b[i]) {
      return i;
    }
  }
  return -1;
};
}  // namespace

void App::run() {
  if (!running_) {
    return;
  }
  load_cam(cam_data);
  float last_time{};
  auto& renderer = VkRender2::get();

  // instances_.emplace_back(
  //     ResourceManager::get().load_model("/Users/tony/Downloads/bistro/Exterior/exterior.glb"));
  // instances_.emplace_back(ResourceManager::get().load_model(
  //     "/Users/tony/Downloads/secret_of_the_mimic_-_mimic/scene.gltf"));

  glm::vec3 v{};
  int w = 1;
  float spacing = 3.f;
  for (v.z = -w; v.z < w; v.z++) {
    for (v.x = -w; v.x < w; v.x++) {
      glm::mat4 transform = glm::translate(glm::mat4{1}, v * spacing);
      // glm::mat4 transform = glm::scale(glm::translate(glm::mat4{1}, v * spacing), vec3{.1});
      // instances_.emplace_back(ResourceManager::get().load_model(
      //     "/Users/tony/models/Models/Fox/glTF/Fox.gltf", transform));
      // instances_.emplace_back(ResourceManager::get().load_model(
      //     "/Users/tony/Downloads/killer_clown_balatro_style/scene.gltf", transform));
      // instances_.emplace_back(ResourceManager::get().load_model(
      //     "/Users/tony/Downloads/wally_walrus_leoncio/scene.gltf", transform));
      // instances_.emplace_back(ResourceManager::get().load_model(
      //     "/Users/tony/models/Models/Cube/glTF/Cube.gltf", transform));
      // "/Users/tony/models/Models/AnimatedCube/glTF/AnimatedCube.gltf", transform));
    }
  }
  // instances_.emplace_back(ResourceManager::get().load_model(
  //     "/Users/tony/models/Models/AnimatedCube/glTF/AnimatedCube.gltf",
  //     glm::scale(glm::translate(glm::mat4{1}, vec3{0, -5, 0}), vec3{10000, 1, 10000})));

  // for (int i = 0; i < 10; i++) {
  //   LINFO("loading model");
  //   instances_.emplace_back(
  //       ResourceManager::get().load_model("/Users/tony/models/Models/Cube/glTF/Cube.gltf"));
  // }
  // instances_.emplace_back(
  //     ResourceManager::get().load_model("/Users/tony/models/Models/Fox/glTF/Fox.gltf"));
  // instances_.emplace_back(ResourceManager::get().load_model("/Users/tony/theboss.glb"));
  character_instance_ = add_instance("/Users/tony/Downloads/theboss.glb");
  // instances_.emplace_back(
  //     ResourceManager::get().load_model("/Users/tony/models/Models/Cube/glTF/Cube.gltf"));
  // instances_.emplace_back(ResourceManager::get().load_model(
  //     "/Users/tony/Downloads/killer_clown_balatro_style/scene.gltf"));
  // instances_.emplace_back(ResourceManager::get().load_model("/Users/tony/theboss/theboss.gltf"));
  // instances_.emplace_back(ResourceManager::get().load_model(
  //     "/Users/tony/models/Models/AlphaBlendModeTest/glTF/AlphaBlendModeTest.gltf"));

  // instances_.emplace_back(ResourceManager::get().load_model(
  //     "/Users/tony/clone/3D-Graphics-Rendering-Cookbook-Second-Edition/data/meshes/"
  //     "medieval_fantasy_book/scene.gltf"));

  // instances_.emplace_back(ResourceManager::get().load_model(
  //     "/Users/tony/models/Models/AnimatedCube/glTF/AnimatedCube.gltf"));
  // instances_.emplace_back(ResourceManager::get().load_model(
  //     "/Users/tony/models/Models/GlassBrokenWindow/glTF/GlassBrokenWindow.gltf"));
  // instances_.emplace_back(ResourceManager::get().load_model(
  //     "/Users/tony/models/Models/AlphaBlendModeTest/glTF/AlphaBlendModeTest.gltf"));

  add_instance("/Users/tony/Downloads/Bistro_Godot_opt.glb");

  // std::filesystem::path env_tex = local_models_dir / "quarry_04_puresky_4k.hdr";
  // std::filesystem::path env_tex = local_models_dir / "immenstadter_horn_2k.hdr";

  // instances_.emplace_back(ResourceManager::get().load_model(local_models_dir /
  // "sponza.glb"));

  // instances_.emplace_back(
  //     ResourceManager::get().load_model(resource_dir / "models/Cube/glTF/Cube.gltf"));
  std::filesystem::path env_tex = "/Users/tony/Downloads/newport_loft.hdr";
  // std::filesystem::path env_tex = "/home/tony/Downloads/quarry_04_puresky_4k.hdr";
  // std::filesystem::path env_tex = "/home/tony/Downloads/golden_gate_hills_4k.hdr";

  // for (int i = 0; i < 10; i++) {
  //   instances_.emplace_back(
  //       ResourceManager::get().load_model("/Users/tony/models/Models/Fox/glTF/Fox.gltf"));
  // }

  character_cam_.set_rotation(quat{1, 0, 0, 0});
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
    ImGuizmo::SetOrthographic(false);
    ImGuizmo::BeginFrame();

    update(dt);
    on_imgui();

    {
      ZoneScopedN("update transforms overall");
      static std::vector<i32> changed_nodes;
      for (auto& instance_handle : instances_) {
        auto* instance = ResourceManager::get().get_instance(instance_handle);
        if (!instance || !instance->is_model_loaded()) continue;
        VkRender2::get().update_animation(*instance, dt);
        changed_nodes.clear();
        validate_hierarchy(instance->scene_graph_data);
        bool dirty_transforms =
            recalc_global_transforms(instance->scene_graph_data, &changed_nodes);
        if (dirty_transforms) {
          VkRender2::get().update_transforms(*instance, changed_nodes);
        }
        renderer.update_skins(*instance);
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
  AnimationManager::shutdown();
  ResourceManager::shutdown();
  VkRender2::shutdown();
  Device::destroy();
  glfwTerminate();
  NFD_Quit();
}

void App::update(float dt) {
  ZoneScoped;
  cam.update_pos(dt);
  // if (Input::key_down(GLFW_KEY_LEFT_CONTROL)) {
  // } else {
  // update_character(dt);
  // }

  info_.view = cam_data.get_view();
  info_.view_pos = cam_data.pos;
  info_.light_dir = glm::normalize(light_dir_);

  int i = 0;
  static float offset{10.f};
  ImGui::DragFloat("offset", &offset);

  auto* instance = get_instance(character_instance_);
  if (instance) {
    // TODO: fix
    if (!character_fsm_.animation_id.is_valid()) {
      character_cam_.pos = instance->scene_graph_data.node_transforms[0].translation;
      character_fsm_.animation_id = instance->animation_id;
      VkRender2::get().draw_joints(*instance);
      auto* animation = AnimationManager::get().get_animation(instance->animation_id);
      auto* state = animation->get_state("Jump");
      state->play_once = true;
      u32 idle = animation->blend_tree.add_clip_node("Idle", "Idle");
      u32 walk = animation->blend_tree.add_clip_node("Walk", "Walk");
      u32 jump = animation->blend_tree.add_clip_node("Jump", "Jump");
      animation->blend_tree.add_lerp_node("IdleWalkBlend", "Idle", "Walk");
      // animation->blend_tree.set_root_node("IdleWalkBlend");
      animation->blend_tree.add_lerp_node("BaseJumpBlend", "IdleWalkBlend", "Jump");
      animation->blend_tree.set_root_node("BaseJumpBlend");
    }
  }
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

void App::on_cursor_event(vec2 pos) const {
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
  if (ImGui::Begin("Player")) {
    character_cam_controller_.on_imgui();
  }
  ImGui::End();
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
        instances_.emplace_back(ResourceManager::get().load_model(outpath));
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
    if (ImGui::TreeNodeEx("Camera")) {
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
    }

    ImGui::ColorEdit3("Sunlight Color", &info_.light_color.x, ImGuiColorEditFlags_Float);
    ImGui::DragFloat("Ambient Intensity", &info_.ambient_intensity);

    if (ImGui::Button("add sponza")) {
      static int offset = 1;
      instances_.emplace_back(ResourceManager::get().load_model(
          local_models_dir / "sponza.glb", glm::translate(mat4{1}, vec3{0, 0, offset * 40})));
      offset++;
    }

    if (selected_obj_ >= 0) {
      if (ImGui::Begin("Node")) {
        auto* instance = ResourceManager::get().get_instance(instances_[selected_obj_]);
        assert(instance && instance->is_model_loaded());
        auto* model = ResourceManager::get().get_model(instance->model_handle);
        static ImGuizmo::MODE mode{ImGuizmo::MODE::WORLD};
        static ImGuizmo::OPERATION operation{ImGuizmo::OPERATION::TRANSLATE};
        if (ImGui::TreeNodeEx("Transform")) {
          auto& scene = instance->scene_graph_data;
          int node = selected_node_;
          int parent = scene.hierarchies[node].parent;

          if (ImGui::DragFloat3("translation", &scene.node_transforms[node].translation.x)) {
            scene.node_transforms[node].to_mat4(scene.local_transforms[node]);
            mark_changed(scene, node);
          }
          int x, y;
          glfwGetWindowSize(window, &x, &y);
          ImGuizmo::SetRect(0, 0, x, y);
          auto aspect = aspect_ratio();
          mat4 proj = glm::perspective(glm::radians(info_.fov_degrees), aspect, .1f, 10000.f);
          mat4 src_transform = scene.local_transforms[node];
          mat4 delta_mat{1};
          ImGuizmo::PushID(node);
          ImGuizmo::OPERATION operations[] = {ImGuizmo::OPERATION::TRANSLATE,
                                              ImGuizmo::OPERATION::ROTATE,
                                              ImGuizmo::OPERATION::SCALEU};
          for (auto& operation : operations) {
            if (ImGuizmo::Manipulate(&info_.view[0][0], &proj[0][0], operation,
                                     ImGuizmo::MODE::LOCAL, &src_transform[0][0],
                                     &delta_mat[0][0])) {
              mat4 new_t = delta_mat * scene.local_transforms[node];
              scene.local_transforms[node] = new_t;
              // if (parent < 0) {
              //   // scene.local_transforms[node] = new_global;
              // } else {
              //   // mat4 parent_global = scene.global_transforms[parent];
              //   // scene.local_transforms[node] = glm::inverse(parent_global) * new_global;
              // }
              decompose_matrix(
                  scene.local_transforms[node], scene.node_transforms[node].translation,
                  scene.node_transforms[node].rotation, scene.node_transforms[node].scale);
              mark_changed(scene, node);
            }
          }
          ImGuizmo::PopID();
          ImGui::TreePop();
        }
        // if (ImGui::TreeNodeEx("Blend Tree", ImGuiTreeNodeFlags_DefaultOpen)) {
        //   auto& nodes = instance->scene_graph_data.animation_data.blend_tree_nodes;
        //   if (nodes.size()) {
        //     ImGui::SliderFloat("weight", &nodes[0].weight, 0.0f, 1.0f);
        //   }
        //   ImGui::TreePop();
        // }
      }
      ImGui::End();
    }

    if (ImGui::TreeNodeEx("Scene")) {
      util::fixed_vector<u32, 8> to_delete;
      size_t i = 0;
      for (auto& instance_handle : instances_) {
        auto* instance = ResourceManager::get().get_instance(instance_handle);
        if (!instance || !instance->is_model_loaded()) {
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
        if (ImGui::Button("Edit")) {
          selected_node_ = 0;
          selected_obj_ = i;
        }
        if (ImGui::TreeNodeEx("%s", 0, "%s", model->path.string().c_str())) {
          scene_node_imgui(instance->scene_graph_data, 0, i);
          ImGui::TreePop();
        }

        auto* animation = AnimationManager::get().get_animation(instance->animation_id);
        if (animation) {
          for (size_t anim_i = 0; anim_i < animation->states.size(); anim_i++) {
            ImGui::PushID(anim_i);
            auto& anim = model->animations[anim_i];
            auto& state = animation->states[anim_i];
            ImGui::Checkbox(anim.name.c_str(), &state.active);
            ImGui::PopID();
          }
          for (auto& [name, node_i] : animation->blend_tree.name_to_blend_tree_node) {
            const auto& node = animation->blend_tree.blend_tree_nodes[node_i];
            ImGui::Text("Node %s weight: %f", name.c_str(),
                        node.weight_idx < animation->blend_tree.blend_tree_nodes.size()
                            ? animation->blend_tree.control_vars[node.weight_idx]
                            : 0.f);
          }
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
      // TODO: fix
      instances_.emplace_back(ResourceManager::get().load_model(paths[i]));
    }
  }
}

void App::scene_node_imgui(gfx::Scene2& scene, int node, u32 obj_id) {
  assert(node != -1);
  auto it = scene.node_to_node_name_idx.find(node);
  ImGui::PushID(node);
  if (ImGui::TreeNode("%s", "%s",
                      it == scene.node_to_node_name_idx.end()
                          ? "Node"
                          : scene.node_names[it->second].c_str())) {
    ImGui::Text("node %i", node);
    if (ImGui::Button("Edit")) {
      selected_node_ = node;
      selected_obj_ = obj_id;
    }
    auto decomp = [&](const mat4& transform) {
      vec3 pos, scale;
      quat rot;
      decompose_matrix(transform, pos, rot, scale);
      ImGui::Text("Translation: %f %f %f", pos.x, pos.y, pos.z);
      ImGui::Text("rot: %f %f %f %f", rot.x, rot.y, rot.z, rot.w);
      ImGui::Text("scale: %f %f %f", scale.x, scale.y, scale.z);
    };
    auto& local_transform = scene.local_transforms[node];
    ImGui::PushID(&local_transform);

    decomp(local_transform);
    ImGui::PopID();
    ImGui::PushID(&scene.global_transforms[node]);
    decomp(scene.global_transforms[node]);
    ImGui::PopID();

    for (int c = scene.hierarchies[node].first_child; c != -1;
         c = scene.hierarchies[c].next_sibling) {
      scene_node_imgui(scene, c, obj_id);
    }
    ImGui::TreePop();
  }
  ImGui::PopID();
}

u32 App::add_instance(const std::filesystem::path& model, const mat4& transform) {
  u32 ret = instances_.size();
  instances_.emplace_back(ResourceManager::get().load_model(model, transform));
  return ret;
}

LoadedInstanceData* App::get_instance(u32 instance) {
  return ResourceManager::get().get_instance(instances_[instance]);
}
void App::update_character(float dt) {
  auto* instance = get_instance(character_instance_);
  if (!instance) return;
  auto* animation = AnimationManager::get().get_animation(instance->animation_id);
  if (character_cam_controller_.update_pos(dt)) {
    auto& nt = instance->scene_graph_data.node_transforms[0];
    nt.translation = character_cam_.pos;
    if (glm::length(character_cam_controller_.velocity) < min_speed_thresh) {
      character_cam_.front = last_look_dir_;
    } else {
      character_cam_.front = glm::normalize(character_cam_controller_.velocity);
      last_look_dir_ = character_cam_.front;
    }
    float speed = glm::length(character_cam_controller_.velocity);
    if (Input::key_down(GLFW_KEY_SPACE)) {
      character_fsm_.jump_time_remaining = 1.f;
      auto* state = animation->get_state("Jump");
      state->active = true;
      state->curr_t = 0.f;
      state->play_once = true;
    }
    character_fsm_.update(dt, speed);
    auto desired_rot = glm::quatLookAt(last_look_dir_, character_cam_.up);
    nt.rotation = glm::slerp(nt.rotation, desired_rot, .1f);
    instance->scene_graph_data.node_transforms[0].to_mat4(
        instance->scene_graph_data.local_transforms[0]);
    mark_changed(instance->scene_graph_data, 0);
  }
}

// void CharacterFSM::transition_to(State state) {
//   prev_state = curr_state;
//   curr_state = state;
//   auto* animation = AnimationManager::get().get_animation(animation_id);
//   switch (curr_state) {
//     case State::Idle:
//       animation->set_target("Idle", .3f);
//       break;
//     case State::Walk:
//       animation->set_target("Walk", .3f);
//       break;
//     default:
//       break;
//   }
// }

void CharacterFSM::update([[maybe_unused]] float dt, float speed) {
  jump_time_remaining = std::max(0.f, jump_time_remaining - dt);
  State new_state = determine_state(speed);
  prev_state = curr_state;
  curr_state = new_state;
  float target = curr_state == State::Walk ? 1.f : 0.f;
  blend_weight = glm::mix(blend_weight, target, dt * 16.f);
  auto* animation = AnimationManager::get().get_animation(animation_id);
  animation->set_blend_state("IdleWalkBlend", blend_weight);
  animation->set_blend_state("BaseJumpBlend", jump_time_remaining);
  LINFO("{} curr", state_to_string(curr_state));
  LINFO("{} prev", state_to_string(prev_state));
  LINFO("{} jump", jump_time_remaining);
}
