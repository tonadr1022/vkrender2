#include "Buffer.hpp"

#include <volk.h>
#include <vulkan/vulkan_core.h>

#include <cassert>
#include <utility>

#include "vk2/BindlessResourceAllocator.hpp"
#include "vk2/Device.hpp"

namespace gfx {

Buffer::Buffer(const BufferCreateInfo &cinfo, std::string name) : name_(std::move(name)) {
  if (cinfo.size == 0) {
    return;
  }
  // https://gpuopen-librariesandsdks.github.io/VulkanMemoryAllocator/html/usage_patterns.html
  VmaAllocationCreateInfo alloc_info{.usage = VMA_MEMORY_USAGE_AUTO};
  VkBufferUsageFlags usage{};
  // if no usage, it's 99% chance a staging buffer
  if (cinfo.usage == 0) {
    usage |= VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
  }
  if (cinfo.flags & BufferCreateFlags_HostVisible) {
    alloc_info.flags |= VMA_ALLOCATION_CREATE_MAPPED_BIT |
                        ((cinfo.flags & BufferCreateFlags_HostAccessRandom)
                             ? VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT
                             : VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);
  } else {
    // device
    usage |= VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
  }
  if (cinfo.usage & BufferUsage_Index) {
    usage |= VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
  }
  if (cinfo.usage & BufferUsage_Vertex) {
    usage |= VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
  }
  if (cinfo.usage & BufferUsage_Storage) {
    usage |= VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
             VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
  }
  if (cinfo.usage & BufferUsage_Indirect) {
    usage |= VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
  }

  VkBufferCreateInfo buffer_create_info{
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = cinfo.size, .usage = usage};
  get_device().create_buffer(&buffer_create_info, &alloc_info, buffer_, allocation_, info_);
  if (info_.size == 0) {
    return;
  }
  if (cinfo.usage & BufferUsage_Storage) {
    resource_info_ = ResourceAllocator::get().allocate_storage_buffer_descriptor(buffer_);
  }
  if (usage & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT) {
    VkBufferDeviceAddressInfo info{.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
                                   .buffer = buffer_};
    buffer_address_ = vkGetBufferDeviceAddress(get_device().device(), &info);
    assert(buffer_address_);
  }

  size_ = cinfo.size;
  usage_ = cinfo.usage;
}

Buffer::Buffer(Buffer &&other) noexcept
    : info_(std::exchange(other.info_, {})),
      name_(std::move(other.name_)),
      usage_(std::exchange(other.usage_, BufferUsage_None)),
      size_(std::exchange(other.size_, 0)),
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
    ResourceAllocator::get().delete_buffer(BufferDeleteInfo{buffer_, allocation_, resource_info_});
    buffer_ = nullptr;
  }
}

}  // namespace gfx
