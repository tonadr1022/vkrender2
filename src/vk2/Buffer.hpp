#pragma once

#include <vk_mem_alloc.h>
#include <vulkan/vulkan_core.h>

#include <optional>
#include <string>

#include "Common.hpp"
#include "vk2/Resource.hpp"
namespace vk2 {

struct BufferCreateInfo {
  u64 size{};
  VkBufferUsageFlags usage{};
  VmaMemoryUsage mem_usage{VMA_MEMORY_USAGE_AUTO};
  VmaAllocationCreateFlags alloc_flags{};
  bool buffer_device_address{false};
};

class Buffer {
 public:
  Buffer(const Buffer &) = delete;
  Buffer &operator=(const Buffer &) = delete;
  Buffer(Buffer &&) noexcept;
  Buffer &operator=(Buffer &&) noexcept;
  ~Buffer();
  explicit Buffer(const BufferCreateInfo &cinfo, std::string name = "Buffer");
  [[nodiscard]] void *mapped_data() const { return info_.pMappedData; }

  [[nodiscard]] VkBuffer buffer() const { return buffer_; }
  [[nodiscard]] VkDeviceAddress device_addr() const { return buffer_address_; }
  [[nodiscard]] u64 size() const { return cinfo_.size; }
  [[nodiscard]] const std::string &name() const { return name_; }

 private:
  BufferCreateInfo cinfo_;
  std::string name_;
  VmaAllocationInfo info_;
  VkBuffer buffer_{};
  VkDeviceAddress buffer_address_{};
  VmaAllocation allocation_{};

  // TODO: private
 public:
  std::optional<BindlessResourceInfo> resource_info_;
};

struct BufferDeleteInfo {
  VkBuffer buffer{};
  VmaAllocation allocation{};
  std::optional<BindlessResourceInfo> resource_info;
};

Buffer create_staging_buffer(u64 size);
Buffer create_storage_buffer(u64 size);

}  // namespace vk2
