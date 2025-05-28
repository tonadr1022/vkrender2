#pragma once

#include "SceneLoader.hpp"
#include "Types.hpp"

using ModelHandle = GenerationalHandle<struct ::gfx::LoadedSceneData>;
using InstanceHandle = GenerationalHandle<struct LoadedInstanceData>;
namespace gfx {
struct ModelGPUResources;
using ModelGPUResourceHandle = GenerationalHandle<struct ::gfx::ModelGPUResources>;

struct StaticModelInstanceResources;
using StaticModelInstanceResourcesHandle = GenerationalHandle<StaticModelInstanceResources>;
}  // namespace gfx
