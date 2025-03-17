#pragma once

#include <vk_mem_alloc.h>
#include <vulkan/vulkan_core.h>

#include <optional>

#include "Common.hpp"
#include "vk2/Resource.hpp"
namespace vk2 {

struct BufferCreateInfo {
  u64 size{};
  VkBufferUsageFlags usage{};
  VmaMemoryUsage mem_usage{VMA_MEMORY_USAGE_AUTO};
  VmaAllocationCreateFlags alloc_flags{};
};

class Buffer {
 public:
  Buffer(const Buffer &) = delete;
  Buffer &operator=(const Buffer &) = delete;
  Buffer(Buffer &&) noexcept;
  Buffer &operator=(Buffer &&) noexcept;
  ~Buffer();
  explicit Buffer(const BufferCreateInfo &cinfo);
  [[nodiscard]] void *mapped_data() const { return info_.pMappedData; }

  [[nodiscard]] VkBuffer buffer() const { return buffer_; }

 private:
  VmaAllocationInfo info_{};
  VkBuffer buffer_{};
  VmaAllocation allocation_{};
  std::optional<BindlessResourceInfo> resource_info_;
};

struct BufferDeleteInfo {
  VkBuffer buffer{};
  VmaAllocation allocation{};
  std::optional<BindlessResourceInfo> resource_info;
};

Buffer create_staging_buffer(u64 size);

}  // namespace vk2
