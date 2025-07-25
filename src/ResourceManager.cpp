#include "ResourceManager.hpp"

#include <glm/gtc/epsilon.hpp>
#include <glm/gtc/matrix_access.hpp>
#include <tracy/Tracy.hpp>

#include "AnimationManager.hpp"
#include "Scene.hpp"
#include "ThreadPool.hpp"
#include "VkRender2.hpp"
#include "core/Logger.hpp"
#include "util/MathUtil.hpp"

InstanceHandle ResourceManager::load_model(const std::filesystem::path& path,
                                           const mat4& transform) {
  ZoneScoped;
  if (!std::filesystem::exists(path)) {
    LERROR("load_static_model: path doesn't exist: {}", path.string());
    return {};
  }

  auto instance_handle = instance_pool_.alloc();
  ModelHandle model_handle;
  bool need_to_load = false;

  {
    std::scoped_lock lock(model_name_mtx_);
    auto it = model_name_to_handle_.find(path);
    if (it != model_name_to_handle_.end()) {
      model_handle = it->second;
    } else {
      model_handle = loaded_model_pool_.alloc();
      model_name_to_handle_.emplace(path, model_handle);  // <-- prevent duplicates early
      auto* d = loaded_model_pool_.get(model_handle);
      d->path = path;
      need_to_load = true;
    }
  }

  if (need_to_load) {
    threads::pool.submit_task([this, path, transform, instance_handle, model_handle]() {
      if (!gfx::VkRender2::get().load_model2(path, *loaded_model_pool_.get(model_handle))) {
        assert(0 && "todo handle error");
        return;
      }

      std::scoped_lock lock(instance_load_req_mtx_);
      instance_load_requests_.emplace_back(transform, instance_handle, model_handle);
    });
  } else {
    std::scoped_lock lock(instance_load_req_mtx_);
    instance_load_requests_.emplace_back(transform, instance_handle, model_handle);
  }
  return instance_handle;
}

namespace {
ResourceManager* instance{};
}

void ResourceManager::shutdown() {
  assert(instance);
  delete instance;
  instance = nullptr;
}

void ResourceManager::init() {
  assert(!instance);
  instance = new ResourceManager;
}

ResourceManager& ResourceManager::get() {
  assert(instance);
  return *instance;
}

bool ResourceManager::add_instance(ModelHandle model_handle, InstanceHandle instance_handle,
                                   const mat4& transform) {
  ZoneScoped;
  auto* instance = instance_pool_.get(instance_handle);
  if (!instance) {
    return false;
  }
  auto* model = loaded_model_pool_.get(model_handle);
  if (!model || model->scene_graph_data.hierarchies.empty()) {
    return false;
  }
  instance->scene_graph_data = model->scene_graph_data;

  if (!util::math::is_identity(transform)) {
    // set initial transform
    instance->scene_graph_data.local_transforms[0] =
        transform * instance->scene_graph_data.local_transforms[0];
    gfx::decompose_matrix(instance->scene_graph_data.local_transforms[0],
                          instance->scene_graph_data.node_transforms[0].translation,
                          instance->scene_graph_data.node_transforms[0].rotation,
                          instance->scene_graph_data.node_transforms[0].scale);
    gfx::mark_changed(instance->scene_graph_data, 0);
  }
  instance->instance_resources_handle = gfx::VkRender2::get().add_instance(model_handle);
  if (!model->animations.empty()) {
    instance->animation_id = AnimationManager::get().add_animation(*instance, *model);
  }

  // TODO: move to animation
  instance->transform_accumulators.resize(instance->scene_graph_data.hierarchies.size());
  instance->model_handle = model_handle;
  return true;
};

void ResourceManager::update() {
  ZoneScoped;
  std::scoped_lock lock(instance_load_req_mtx_);
  auto new_end = std::ranges::remove_if(instance_load_requests_, [this](InstanceLoadRequest& req) {
    ZoneScoped;
    return add_instance(req.model_handle, req.instance_handle, req.transform);
  });
  instance_load_requests_.erase(new_end.begin(), new_end.end());
}

void ResourceManager::remove_model(InstanceHandle handle) {
  auto* instance = instance_pool_.get(handle);
  assert(instance);
  if (instance) {
    gfx::VkRender2::get().remove_instance(instance->instance_resources_handle);
  }
  instance_pool_.destroy(handle);
}
