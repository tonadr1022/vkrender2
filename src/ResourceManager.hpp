#pragma once

// TODO: not include this
#include "SceneLoader.hpp"
#include "SceneResources.hpp"
#include "Types.hpp"

struct LoadedModelData {
  gfx::Scene2 scene_graph_data;
  gfx::ModelGPUResourceHandle gpu_resource_handle;
  std::filesystem::path path;
};

struct LoadedInstanceData {
  ModelHandle model_handle;
  gfx::Scene2 scene_graph_data;
  gfx::StaticModelInstanceResourcesHandle instance_resources_handle;
  [[nodiscard]] bool is_valid() const { return model_handle.is_valid(); }
};
using InstanceHandle = GenerationalHandle<LoadedInstanceData>;

class ResourceManager {
 public:
  static void init();
  static void shutdown();
  static ResourceManager& get();
  void on_imgui();
  void update();
  InstanceHandle load_model(const std::filesystem::path& path, const mat4& transform = mat4{1});
  void remove_model(InstanceHandle handle);
  LoadedModelData* get_model(ModelHandle handle) { return loaded_model_pool_.get(handle); }
  LoadedInstanceData* get_instance(InstanceHandle handle) { return instance_pool_.get(handle); }

 private:
  void add_instance(ModelHandle model_handle, InstanceHandle instance_handle,
                    const mat4& transform);
  ResourceManager() = default;
  struct LoadSceneResult {
    std::filesystem::path path;
    gfx::LoadedSceneData result;
  };

  std::mutex scene_load_mtx_;
  Pool<ModelHandle, LoadedModelData> loaded_model_pool_;
  Pool<InstanceHandle, LoadedInstanceData> instance_pool_;
  std::unordered_map<std::string, ModelHandle> model_name_to_handle_;

  struct InstanceLoadRequest {
    mat4 transform;
    InstanceHandle instance_handle;
    ModelHandle model_handle;
  };
  std::mutex instance_load_req_mtx_;
  std::vector<InstanceLoadRequest> instance_load_requests_;
};
