#pragma once

#include "SceneLoader.hpp"
#include "Types.hpp"

using ModelHandle = GenerationalHandle<struct ::gfx::LoadedSceneData>;
namespace gfx {
struct ModelGPUResources;
using ModelGPUResourceHandle = GenerationalHandle<struct ::gfx::ModelGPUResources>;

struct StaticModelInstanceResources;
using StaticModelInstanceResourcesHandle = GenerationalHandle<StaticModelInstanceResources>;
}  // namespace gfx
