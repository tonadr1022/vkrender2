#include "Buffer.hpp"

#include <volk.h>
#include <vulkan/vulkan_core.h>

#include <cassert>
#include <utility>

#include "vk2/BindlessResourceAllocator.hpp"
#include "vk2/Device.hpp"

namespace gfx::vk2 {

// https://gpuopen-librariesandsdks.github.io/VulkanMemoryAllocator/html/usage_patterns.html
Buffer::Buffer(const BufferCreateInfo &cinfo, std::string name) : name_(std::move(name)) {
  auto usage = cinfo.usage;
  bool buffer_device_address = true;
  auto alloc_flags = cinfo.alloc_flags;
  if (buffer_device_address) {
    usage |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    alloc_flags |= VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
  }
  VkBufferCreateInfo buffer_create_info{
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = cinfo.size, .usage = usage};
  VmaAllocationCreateInfo alloc_info{.flags = alloc_flags, .usage = cinfo.mem_usage};
  vk2::get_device().create_buffer(&buffer_create_info, &alloc_info, buffer_, allocation_, info_);
  assert(info_.size);
  if (cinfo.usage & VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) {
    resource_info_ = BindlessResourceAllocator::get().allocate_storage_buffer_descriptor(buffer_);
  }
  if (buffer_device_address) {
    VkBufferDeviceAddressInfo info{.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
                                   .buffer = buffer_};
    buffer_address_ = vkGetBufferDeviceAddress(vk2::get_device().device(), &info);
    assert(buffer_address_);
  }
  cinfo_ = cinfo;
}

Buffer::Buffer(Buffer &&other) noexcept
    : cinfo_(std::exchange(other.cinfo_, {})),
      name_(std::move(other.name_)),
      info_(std::exchange(other.info_, {})),
      buffer_(std::exchange(other.buffer_, nullptr)),
      buffer_address_(std::exchange(other.buffer_address_, 0ll)),
      allocation_(std::exchange(other.allocation_, nullptr)),
      resource_info_(std::exchange(other.resource_info_, std::nullopt)) {}

Buffer &Buffer::operator=(Buffer &&other) noexcept {
  if (&other == this) {
    return *this;
  }
  this->~Buffer();
  info_ = std::exchange(other.info_, {});
  cinfo_ = std::exchange(other.cinfo_, {});
  name_ = std::move(other.name_);
  buffer_address_ = std::exchange(other.buffer_address_, 0ll);
  buffer_ = std::exchange(other.buffer_, nullptr);
  allocation_ = std::exchange(other.allocation_, nullptr);
  resource_info_ = std::exchange(other.resource_info_, std::nullopt);
  return *this;
}

Buffer::~Buffer() {
  if (buffer_) {
    assert(allocation_);
    BindlessResourceAllocator::get().delete_buffer(
        BufferDeleteInfo{buffer_, allocation_, resource_info_});
    buffer_ = nullptr;
  }
}

Buffer create_staging_buffer(u64 size) {
  return Buffer{
      BufferCreateInfo{.size = size,
                       .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                       .alloc_flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                                      VMA_ALLOCATION_CREATE_MAPPED_BIT}};
}

Buffer create_storage_buffer(u64 size) {
  return Buffer{BufferCreateInfo{
      .size = size,
      .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
  }};
}
}  // namespace gfx::vk2
