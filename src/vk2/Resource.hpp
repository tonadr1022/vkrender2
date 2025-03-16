#pragma once

#include <vk_mem_alloc.h>
#include <vulkan/vulkan_core.h>

#include "Common.hpp"

namespace vk2 {

enum class ResourceType : u8 {
  StorageImage,
  StorageBuffer,
  // TODO: need one or other?
  SampledImage,
  CombinedImageSampler,
  Sampler,
};

struct BindlessResourceInfo {
  ResourceType type;
  u32 handle;
};

}  // namespace vk2
