#include "Texture.hpp"

#include <volk.h>
#include <vulkan/vulkan_core.h>

#include <cmath>

#include "vk2/VkTypes.hpp"

namespace gfx {
uint32_t get_mip_levels(VkExtent2D size) {
  return static_cast<uint32_t>(std::floor(std::log2(std::max(size.width, size.height)))) + 1;
}

bool format_is_color(Format format) {
  return !(format_is_stencil(format) || format_is_depth(format));
}
bool format_is_srgb(Format format) {
  switch (vk2::convert_format(format)) {
    case VK_FORMAT_R8_SRGB:
    case VK_FORMAT_R8G8_SRGB:
    case VK_FORMAT_R8G8B8_SRGB:
    case VK_FORMAT_B8G8R8_SRGB:
    case VK_FORMAT_R8G8B8A8_SRGB:
    case VK_FORMAT_B8G8R8A8_SRGB:
    case VK_FORMAT_A8B8G8R8_SRGB_PACK32:
    case VK_FORMAT_BC1_RGB_SRGB_BLOCK:
    case VK_FORMAT_BC1_RGBA_SRGB_BLOCK:
    case VK_FORMAT_BC2_SRGB_BLOCK:
    case VK_FORMAT_BC3_SRGB_BLOCK:
    case VK_FORMAT_BC7_SRGB_BLOCK:
      return true;
    default:
      return false;
  }
}
bool format_is_depth(Format format) {
  switch (vk2::convert_format(format)) {
    case VK_FORMAT_D32_SFLOAT:
    case VK_FORMAT_D16_UNORM:
      return true;
    default:
      return false;
  }
  return false;
}
bool format_is_stencil(Format format) {
  switch (vk2::convert_format(format)) {
    case VK_FORMAT_D16_UNORM_S8_UINT:
    case VK_FORMAT_D24_UNORM_S8_UINT:
    case VK_FORMAT_D32_SFLOAT_S8_UINT:
      return true;
    default:
      return false;
  }
}
VkImageType vkviewtype_to_img_type(VkImageViewType view_type) {
  switch (view_type) {
    case VK_IMAGE_VIEW_TYPE_2D:
    case VK_IMAGE_VIEW_TYPE_2D_ARRAY:
    case VK_IMAGE_VIEW_TYPE_CUBE:
    case VK_IMAGE_VIEW_TYPE_CUBE_ARRAY:
      return VK_IMAGE_TYPE_2D;

    case VK_IMAGE_VIEW_TYPE_3D:
      return VK_IMAGE_TYPE_3D;

    case VK_IMAGE_VIEW_TYPE_1D:
    case VK_IMAGE_VIEW_TYPE_1D_ARRAY:
      return VK_IMAGE_TYPE_1D;
    default:
      assert(0);
      return {};
  }
}

uint32_t format_storage_size(Format format) {
  switch (vk2::convert_format(format)) {
    case VK_FORMAT_R8_UNORM:
    case VK_FORMAT_R8_SNORM:
    case VK_FORMAT_R8_SINT:
    case VK_FORMAT_R8_SRGB:
    case VK_FORMAT_R8_UINT:
      return 1;

    case VK_FORMAT_R16_UNORM:
    case VK_FORMAT_R16_SNORM:
    case VK_FORMAT_R8G8_UNORM:
    case VK_FORMAT_R8G8_SNORM:
    case VK_FORMAT_R4G4B4A4_UNORM_PACK16:
    case VK_FORMAT_R5G5B5A1_UNORM_PACK16:
    case VK_FORMAT_R16_SFLOAT:
    case VK_FORMAT_R16_SINT:
    case VK_FORMAT_R16_UINT:
    case VK_FORMAT_R8G8_SINT:
    case VK_FORMAT_R8G8_UINT:
    case VK_FORMAT_D16_UNORM:
    case VK_FORMAT_R8G8_SRGB:
      return 2;

    case VK_FORMAT_R16G16_UNORM:
    case VK_FORMAT_R16G16_SNORM:
    case VK_FORMAT_R8G8B8A8_UNORM:
    case VK_FORMAT_B8G8R8A8_UNORM:
    case VK_FORMAT_B8G8R8A8_SNORM:
    case VK_FORMAT_R8G8B8A8_SNORM:
    case VK_FORMAT_A2R10G10B10_UNORM_PACK32:
    case VK_FORMAT_A2R10G10B10_UINT_PACK32:
    case VK_FORMAT_R8G8B8A8_SRGB:
    case VK_FORMAT_B8G8R8A8_SRGB:
    case VK_FORMAT_R16G16_SFLOAT:
    case VK_FORMAT_R32_SFLOAT:
    case VK_FORMAT_B10G11R11_UFLOAT_PACK32:
    case VK_FORMAT_E5B9G9R9_UFLOAT_PACK32:
    case VK_FORMAT_R32_SINT:
    case VK_FORMAT_R32_UINT:
    case VK_FORMAT_R16G16_SINT:
    case VK_FORMAT_R16G16_UINT:
    case VK_FORMAT_R8G8B8A8_SINT:
    case VK_FORMAT_R8G8B8A8_UINT:
    case VK_FORMAT_D32_SFLOAT:
    case VK_FORMAT_X8_D24_UNORM_PACK32:
    case VK_FORMAT_D24_UNORM_S8_UINT:
      return 4;

    case VK_FORMAT_D32_SFLOAT_S8_UINT:
      return 5;

    case VK_FORMAT_R16G16B16A16_UNORM:
    case VK_FORMAT_R16G16B16A16_SNORM:
    case VK_FORMAT_R16G16B16A16_SFLOAT:
    case VK_FORMAT_R32G32_SFLOAT:
    case VK_FORMAT_R32G32_SINT:
    case VK_FORMAT_R32G32_UINT:
    case VK_FORMAT_R16G16B16A16_SINT:
    case VK_FORMAT_R16G16B16A16_UINT:
    case VK_FORMAT_BC1_RGB_UNORM_BLOCK:
    case VK_FORMAT_BC1_RGB_SRGB_BLOCK:
    case VK_FORMAT_BC1_RGBA_UNORM_BLOCK:
    case VK_FORMAT_BC1_RGBA_SRGB_BLOCK:
    case VK_FORMAT_BC4_UNORM_BLOCK:
    case VK_FORMAT_BC4_SNORM_BLOCK:
      return 8;

    case VK_FORMAT_R32G32B32A32_SFLOAT:
    case VK_FORMAT_R32G32B32A32_SINT:
    case VK_FORMAT_R32G32B32A32_UINT:
    case VK_FORMAT_BC2_UNORM_BLOCK:
    case VK_FORMAT_BC2_SRGB_BLOCK:
    case VK_FORMAT_BC3_UNORM_BLOCK:
    case VK_FORMAT_BC3_SRGB_BLOCK:
    case VK_FORMAT_BC5_UNORM_BLOCK:
    case VK_FORMAT_BC5_SNORM_BLOCK:
    case VK_FORMAT_BC6H_UFLOAT_BLOCK:
    case VK_FORMAT_BC6H_SFLOAT_BLOCK:
    case VK_FORMAT_BC7_UNORM_BLOCK:
    case VK_FORMAT_BC7_SRGB_BLOCK:
      return 16;
    default:
      assert(0);
      return 0;
  }

  assert(false);
  return 0;
}
bool format_is_block_compreesed(Format format) {
  switch (vk2::convert_format(format)) {
    case VK_FORMAT_BC1_RGB_UNORM_BLOCK:
    case VK_FORMAT_BC1_RGB_SRGB_BLOCK:
    case VK_FORMAT_BC1_RGBA_UNORM_BLOCK:
    case VK_FORMAT_BC1_RGBA_SRGB_BLOCK:
    case VK_FORMAT_BC2_UNORM_BLOCK:
    case VK_FORMAT_BC2_SRGB_BLOCK:
    case VK_FORMAT_BC3_UNORM_BLOCK:
    case VK_FORMAT_BC3_SRGB_BLOCK:
    // r
    case VK_FORMAT_BC4_UNORM_BLOCK:
    case VK_FORMAT_BC4_SNORM_BLOCK:
    // rg
    case VK_FORMAT_BC5_UNORM_BLOCK:
    case VK_FORMAT_BC5_SNORM_BLOCK:
    case VK_FORMAT_BC6H_UFLOAT_BLOCK:
    case VK_FORMAT_BC6H_SFLOAT_BLOCK:
    case VK_FORMAT_BC7_UNORM_BLOCK:
    case VK_FORMAT_BC7_SRGB_BLOCK:
      return true;
    default:
      return false;
  }
  return false;
}
u64 block_compressed_image_size(Format format, uvec3 extent) {
  u64 rounded_w = (extent.x + 3) & ~3;
  u64 rounded_h = (extent.y + 3) & ~3;

  u64 num_blocks_w = rounded_w / 4;
  u64 num_blocks_h = rounded_h / 4;
  u64 num_blocks = num_blocks_w * num_blocks_h * extent.z;

  // BC1 and BC4 use 8 bytes per block, others use 16 bytes per block
  switch (vk2::convert_format(format)) {
    case VK_FORMAT_BC1_RGB_UNORM_BLOCK:
    case VK_FORMAT_BC1_RGB_SRGB_BLOCK:
    case VK_FORMAT_BC1_RGBA_UNORM_BLOCK:
    case VK_FORMAT_BC1_RGBA_SRGB_BLOCK:
    case VK_FORMAT_BC4_UNORM_BLOCK:
    case VK_FORMAT_BC4_SNORM_BLOCK:
      return num_blocks * 8;  // 64 bits per block

    case VK_FORMAT_BC2_UNORM_BLOCK:
    case VK_FORMAT_BC2_SRGB_BLOCK:
    case VK_FORMAT_BC3_UNORM_BLOCK:
    case VK_FORMAT_BC3_SRGB_BLOCK:
    case VK_FORMAT_BC5_UNORM_BLOCK:
    case VK_FORMAT_BC5_SNORM_BLOCK:
    case VK_FORMAT_BC6H_UFLOAT_BLOCK:
    case VK_FORMAT_BC6H_SFLOAT_BLOCK:
    case VK_FORMAT_BC7_UNORM_BLOCK:
    case VK_FORMAT_BC7_SRGB_BLOCK:
      return num_blocks * 16;  // 128 bits per block

    default:
      assert(0);
      return 0;
  }
}
u64 img_to_buffer_size(Format format, uvec3 extent) {
  // https://registry.khronos.org/DataFormat/specs/1.3/dataformat.1.3.html#BPTC
  if (format_is_block_compreesed(format)) {
    return block_compressed_image_size(format, extent);
  }
  return static_cast<u64>(extent.x) * extent.y * extent.z * format_storage_size(format);
}

uint32_t get_mip_levels(uvec2 size) { return get_mip_levels(VkExtent2D{size.x, size.y}); }

}  // namespace gfx
