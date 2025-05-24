#include "ResourceManager.hpp"

#include <tracy/Tracy.hpp>

#include "ThreadPool.hpp"
#include "VkRender2.hpp"
#include "core/Logger.hpp"

InstanceHandle ResourceManager::load_model(const std::filesystem::path& path,
                                           const mat4& transform) {
  ZoneScoped;
  if (!std::filesystem::exists(path)) {
    LERROR("load_static_model: path doesn't exist: {}", path.string());
    return {};
  }

  auto model_handle_it = model_name_to_handle_.find(path);
  auto instance_handle = instance_pool_.alloc();

  if (model_handle_it != model_name_to_handle_.end()) {
    add_instance(model_handle_it->second, instance_handle, transform);
  } else {
    threads::pool.submit_task([this, path, transform, instance_handle]() {
      auto model_handle = loaded_model_pool_.alloc();
      auto* d = loaded_model_pool_.get(model_handle);
      if (!gfx::VkRender2::get().load_model2(path, *d)) {
        assert(0 && "todo handle error");
        return;
      }
      d->path = path;
      add_instance(model_handle, instance_handle, transform);
    });
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

void ResourceManager::add_instance(ModelHandle model_handle, InstanceHandle instance_handle,
                                   const mat4& transform) {
  auto* instance = instance_pool_.get(instance_handle);
  instance->instance_resources_handle = gfx::VkRender2::get().add_instance(model_handle, transform);
  auto* model = loaded_model_pool_.get(model_handle);
  assert(model);
  instance->scene_graph_data = model->scene_graph_data;
  instance->model_handle = model_handle;
};
