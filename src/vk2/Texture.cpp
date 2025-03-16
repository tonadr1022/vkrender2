#include "Texture.hpp"

#include <volk.h>

#include <cmath>
#include <utility>

#include "vk2/BindlessResourceAllocator.hpp"

namespace vk2 {
uint32_t get_mip_levels(VkExtent2D size) {
  return static_cast<uint32_t>(std::floor(std::log2(std::max(size.width, size.height)))) + 1;
}

Texture::~Texture() {
  if (image_) {
    BindlessResourceAllocator::get().destroy_image(*this);
  }
}

Texture::Texture(Texture&& other) noexcept {
  image_ = std::exchange(other.image_, nullptr);
  view_ = std::exchange(other.view_, nullptr);
  allocation_ = std::exchange(other.allocation_, nullptr);
  extent_ = std::exchange(other.extent_, {});
  format_ = std::exchange(other.format_, VK_FORMAT_UNDEFINED);
}

Texture& Texture::operator=(Texture&& other) noexcept {
  image_ = std::exchange(other.image_, nullptr);
  view_ = std::exchange(other.view_, nullptr);
  allocation_ = std::exchange(other.allocation_, nullptr);
  extent_ = std::exchange(other.extent_, {});
  format_ = std::exchange(other.format_, VK_FORMAT_UNDEFINED);
  return *this;
}

Texture::Texture() = default;

}  // namespace vk2
