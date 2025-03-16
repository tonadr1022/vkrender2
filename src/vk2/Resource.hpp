#pragma once

#include <vk_mem_alloc.h>
#include <vulkan/vulkan_core.h>

#include "Common.hpp"

namespace vk2 {

enum class ResourceType : u8 {
  STORAGE_IMAGE,
  STORAGE_BUFFER,
  // TODO: need one or other?
  SAMPLED_IMAGE,
  COMBINED_IMAGE_SAMPLER,
  SAMPLER,
};

struct BindlessResourceInfo {
  ResourceType type;
  u32 handle;
};

}  // namespace vk2
