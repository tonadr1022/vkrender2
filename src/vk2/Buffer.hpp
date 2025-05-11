#pragma once

#include <vk_mem_alloc.h>
#include <vulkan/vulkan_core.h>

#include <optional>
#include <string>

#include "Common.hpp"
#include "vk2/Resource.hpp"
namespace gfx {

using BufferUsageFlags = u8;
enum BufferUsage : u8 {
  BufferUsage_None = 0,
  BufferUsage_Storage = 1 << 0,
  BufferUsage_Indirect = 1 << 1,
  BufferUsage_Vertex = 1 << 2,
  BufferUsage_Index = 1 << 3,
  BufferUsage_Uniform = 1 << 4,
};

enum BufferCreateFlags : u8 {
  BufferCreateFlags_HostVisible = 1 << 0,
  // use in tandem with host visible. if host visible but not random, access will be sequential
  BufferCreateFlags_HostAccessRandom = 1 << 1,
};

struct BufferCreateInfo {
  u64 size{};
  BufferUsageFlags usage{};
  BufferCreateFlags flags{};
};

class Buffer {
 public:
  Buffer() = default;
  [[nodiscard]] void *mapped_data() const { return info_.pMappedData; }
  [[nodiscard]] VkBuffer buffer() const { return buffer_; }
  [[nodiscard]] VkDeviceAddress device_addr() const { return buffer_address_; }
  [[nodiscard]] u64 size() const { return size_; }
  [[nodiscard]] const std::string &name() const { return name_; }

 private:
  friend class Device;
  VmaAllocationInfo info_;
  std::string name_;
  BufferUsageFlags usage_{};
  u64 size_{};
  VkBuffer buffer_{};
  VkDeviceAddress buffer_address_{};
  VmaAllocation allocation_{};

 public:
  std::optional<BindlessResourceInfo> resource_info_;
};

struct BufferDeleteInfo {
  VkBuffer buffer{};
  VmaAllocation allocation{};
  std::optional<BindlessResourceInfo> resource_info;
};

}  // namespace gfx
