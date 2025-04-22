#include "VkTypes.hpp"
namespace gfx::vk2 {
VkFormat to_vkformat(Format format) { return static_cast<VkFormat>(format); }
Format vkformat_to_format(VkFormat format) { return static_cast<Format>(format); }

VkImageAspectFlags format_to_aspect_flags(VkFormat format) {
  switch (format) {
    case VK_FORMAT_UNDEFINED:
      return 0;

    case VK_FORMAT_S8_UINT:
      return VK_IMAGE_ASPECT_STENCIL_BIT;

    case VK_FORMAT_D16_UNORM_S8_UINT:
    case VK_FORMAT_D24_UNORM_S8_UINT:
    case VK_FORMAT_D32_SFLOAT_S8_UINT:
      return VK_IMAGE_ASPECT_STENCIL_BIT | VK_IMAGE_ASPECT_DEPTH_BIT;

    case VK_FORMAT_D16_UNORM:
    case VK_FORMAT_D32_SFLOAT:
    case VK_FORMAT_X8_D24_UNORM_PACK32:
      return VK_IMAGE_ASPECT_DEPTH_BIT;

    default:
      return VK_IMAGE_ASPECT_COLOR_BIT;
  }
}

VkImageAspectFlags format_to_aspect_flags(Format format) {
  return format_to_aspect_flags(to_vkformat(format));
}

}  // namespace gfx::vk2
