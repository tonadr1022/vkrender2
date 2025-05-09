#pragma once

#include "Common.hpp"

namespace gfx {

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

}  // namespace gfx
