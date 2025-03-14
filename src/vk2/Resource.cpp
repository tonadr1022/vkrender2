#include "Resource.hpp"

#include <cmath>
#include <utility>

#include "vk2/Device.hpp"

namespace vk2 {
uint32_t get_mip_levels(VkExtent2D size) {
  return static_cast<uint32_t>(std::floor(std::log2(std::max(size.width, size.height)))) + 1;
}

UniqueImage::~UniqueImage() {
  if (image) {
    assert(allocation);
    vmaDestroyImage(vk2::device().allocator(), image, allocation);
  }
}

UniqueImage::UniqueImage(UniqueImage&& other) noexcept {
  image = std::exchange(other.image, nullptr);
  view = std::exchange(other.view, nullptr);
  allocation = std::exchange(other.allocation, nullptr);
  extent = std::exchange(other.extent, {});
  format = std::exchange(other.format, VK_FORMAT_UNDEFINED);
}
UniqueImage& UniqueImage::operator=(UniqueImage&& other) noexcept {
  image = std::exchange(other.image, nullptr);
  view = std::exchange(other.view, nullptr);
  allocation = std::exchange(other.allocation, nullptr);
  extent = std::exchange(other.extent, {});
  format = std::exchange(other.format, VK_FORMAT_UNDEFINED);
  return *this;
}
}  // namespace vk2
