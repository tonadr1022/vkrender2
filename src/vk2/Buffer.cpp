#include "Buffer.hpp"

#include <vulkan/vulkan_core.h>

#include <utility>

#include "vk2/BindlessResourceAllocator.hpp"
#include "vk2/Device.hpp"
#include "vk2/VkCommon.hpp"

namespace vk2 {

Buffer::Buffer(const BufferCreateInfo &cinfo) {
  VkBufferCreateInfo buffer_create_info{.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                                        .size = cinfo.size,
                                        .usage = cinfo.usage,
                                        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
                                        .queueFamilyIndexCount = 1,
                                        .pQueueFamilyIndices = &cinfo.queue_idx};
  VmaAllocationCreateInfo alloc_info{.usage = cinfo.mem_usage};
  VK_CHECK(vmaCreateBuffer(vk2::get_device().allocator(), &buffer_create_info, &alloc_info,
                           &buffer_, &allocation_, nullptr));
  storage_image_resource_info_ =
      BindlessResourceAllocator::get().allocate_storage_buffer_descriptor(buffer_);
}

Buffer::Buffer(Buffer &&other) noexcept
    : buffer_(std::exchange(other.buffer_, nullptr)),
      allocation_(std::exchange(other.allocation_, nullptr)),
      storage_image_resource_info_(
          std::exchange(other.storage_image_resource_info_, std::nullopt)) {}

Buffer &Buffer::operator=(Buffer &&other) noexcept {
  if (&other == this) {
    return *this;
  }
  this->~Buffer();
  buffer_ = std::exchange(other.buffer_, nullptr);
  allocation_ = std::exchange(other.allocation_, nullptr);
  storage_image_resource_info_ = std::exchange(other.storage_image_resource_info_, std::nullopt);
  return *this;
}

Buffer::~Buffer() { assert(0); }

}  // namespace vk2
