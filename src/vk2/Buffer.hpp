#pragma once

#include <vk_mem_alloc.h>
#include <vulkan/vulkan_core.h>

#include <optional>

#include "Common.hpp"
#include "vk2/Resource.hpp"
namespace vk2 {

struct BufferCreateInfo {
  u64 size{};
  u32 queue_idx{};
  VkBufferUsageFlagBits usage{};
  VmaMemoryUsage mem_usage{};
};

class Buffer {
 public:
  Buffer(const Buffer &) = delete;
  Buffer &operator=(const Buffer &) = delete;
  Buffer(Buffer &&) noexcept;
  Buffer &operator=(Buffer &&) noexcept;
  ~Buffer();
  explicit Buffer(const BufferCreateInfo &cinfo);

 private:
  VkBuffer buffer_;
  VmaAllocation allocation_;
  std::optional<BindlessResourceInfo> storage_image_resource_info_;
};

Buffer new_staging_buffer();

}  // namespace vk2
