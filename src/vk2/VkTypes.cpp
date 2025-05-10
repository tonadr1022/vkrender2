#include "VkTypes.hpp"

#include <vulkan/vulkan_core.h>
namespace gfx::vk2 {
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
  return 0;
}

VkImageAspectFlags format_to_aspect_flags(Format format) {
  return format_to_aspect_flags(to_vkformat(format));
}

VkCullModeFlags convert_cull_mode(CullMode mode) {
  switch (mode) {
    case CullMode::None:
      return VK_CULL_MODE_NONE;
    case CullMode::Back:
      return VK_CULL_MODE_BACK_BIT;
    case CullMode::Front:
      return VK_CULL_MODE_FRONT_BIT;
    default:
      return 0;
  }
  return 0;
}

VkImageViewType convert_image_view_type(ImageViewType type) {
  switch (type) {
    case ImageViewType::OneD:
      return VK_IMAGE_VIEW_TYPE_1D;
    case ImageViewType::TwoD:
      return VK_IMAGE_VIEW_TYPE_2D;
    case ImageViewType::ThreeD:
      return VK_IMAGE_VIEW_TYPE_3D;
    case ImageViewType::Cube:
      return VK_IMAGE_VIEW_TYPE_CUBE;
    case ImageViewType::OneDArray:
      return VK_IMAGE_VIEW_TYPE_1D_ARRAY;
    case ImageViewType::TwoDArray:
      return VK_IMAGE_VIEW_TYPE_2D_ARRAY;
    case ImageViewType::CubeArray:
      return VK_IMAGE_VIEW_TYPE_CUBE_ARRAY;
    default:
      assert(0);
      return {};
  }
}

}  // namespace gfx::vk2
