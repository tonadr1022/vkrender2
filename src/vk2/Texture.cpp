#include "Texture.hpp"

#include <volk.h>
#include <vulkan/vulkan_core.h>

#include <cmath>
#include <utility>

#include "vk2/BindlessResourceAllocator.hpp"
#include "vk2/Device.hpp"
#include "vk2/Resource.hpp"
#include "vk2/VkCommon.hpp"

namespace vk2 {
uint32_t get_mip_levels(VkExtent2D size) {
  return static_cast<uint32_t>(std::floor(std::log2(std::max(size.width, size.height)))) + 1;
}

Texture::~Texture() {
  if (image_) {
    assert(allocation_);
    BindlessResourceAllocator::get().delete_texture({image_, allocation_});
    image_ = nullptr;
  }
}

Texture::Texture(Texture&& other) noexcept
    : create_info_(std::exchange(other.create_info_, {})),
      view_(std::move(other.view_)),
      image_(std::exchange(other.image_, nullptr)),
      allocation_(std::exchange(other.allocation_, nullptr)) {}

Texture& Texture::operator=(Texture&& other) noexcept {
  if (&other == this) {
    return *this;
  }
  this->~Texture();
  create_info_ = std::exchange(other.create_info_, {});
  view_ = std::move(other.view_);
  image_ = std::exchange(other.image_, nullptr);
  allocation_ = std::exchange(other.allocation_, nullptr);
  return *this;
}

// TODO: need move constructor for this
TextureView::TextureView(const Texture& texture, const TextureViewCreateInfo& info)
    : create_info_(info) {
  VkImageViewType view_type = info.view_type != VK_IMAGE_VIEW_TYPE_MAX_ENUM
                                  ? info.view_type
                                  : texture.create_info_.view_type;
  auto view_info = VkImageViewCreateInfo{.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                                         .image = texture.image_,
                                         .viewType = view_type,
                                         .format = info.format,
                                         .components = info.components,
                                         .subresourceRange = info.range};
  VK_CHECK(vkCreateImageView(vk2::get_device().device(), &view_info, nullptr, &view_));

  // can only be a storage image if color and general usage
  if (format_is_color(info.format) && texture.create_info_.usage == TextureUsage::General) {
    storage_image_resource_info_ = BindlessResourceAllocator::get().allocate_storage_img_descriptor(
        view_, VK_IMAGE_LAYOUT_GENERAL);
  }
  sampled_image_resource_info_ = BindlessResourceAllocator::get().allocate_sampled_img_descriptor(
      view_, VK_IMAGE_LAYOUT_READ_ONLY_OPTIMAL);
}

Texture::Texture(const TextureCreateInfo& create_info) {
  VmaAllocationCreateFlags alloc_flags{};
  VkImageUsageFlags usage{};
  auto attachment_usage = format_is_color(create_info.format)
                              ? VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT
                              : VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;

  if (create_info.usage == TextureUsage::General) {
    // can't use srgb images for storage. wouldn't want to anyway
    auto storage_usage =
        (format_is_color(create_info.format) && !format_is_srgb(create_info.format))
            ? VK_IMAGE_USAGE_STORAGE_BIT
            : 0;
    usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
            VK_IMAGE_USAGE_SAMPLED_BIT | storage_usage | attachment_usage;

  } else if (create_info.usage == TextureUsage::AttachmentReadOnly) {
    // dedicated memory for attachment textures
    alloc_flags |= VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;
    // can't copy to attachment images, 99% want to sample them as well
    usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | attachment_usage;

  } else {  // ReadOnly
    // copy to/from, sample
    usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
            VK_IMAGE_USAGE_SAMPLED_BIT;
  }

  VmaAllocationCreateInfo alloc_create_info{
      .flags = alloc_flags,
      .usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
  };
  VkImageCreateFlags img_create_flags{};
  if (create_info.view_type == VK_IMAGE_VIEW_TYPE_CUBE ||
      create_info.view_type == VK_IMAGE_VIEW_TYPE_CUBE_ARRAY) {
    img_create_flags |= VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
  }

  VkImageCreateInfo info{.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
                         .flags = img_create_flags,
                         .imageType = vkviewtype_to_img_type(create_info.view_type),
                         .format = create_info.format,
                         .extent = create_info.extent,
                         .mipLevels = create_info.mip_levels,
                         .arrayLayers = create_info.array_layers,
                         .samples = create_info.samples,
                         .tiling = VK_IMAGE_TILING_OPTIMAL,
                         .usage = usage,
                         .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED};
  image_ = nullptr;
  VK_CHECK(vmaCreateImage(vk2::get_device().allocator(), &info, &alloc_create_info, &image_,
                          &allocation_, nullptr));
  if (!image_) {
    return;
  }

  VkImageAspectFlags aspect = VK_IMAGE_ASPECT_NONE;
  if (format_is_color(create_info.format)) {
    aspect |= VK_IMAGE_ASPECT_COLOR_BIT;
  }
  if (format_is_depth(create_info.format)) {
    aspect |= VK_IMAGE_ASPECT_DEPTH_BIT;
  }
  if (format_is_stencil(create_info.format)) {
    aspect |= VK_IMAGE_ASPECT_STENCIL_BIT;
  }

  create_info_ = create_info;
  view_ = TextureView{*this, TextureViewCreateInfo{.format = create_info.format,
                                                   .range =
                                                       {
                                                           .aspectMask = aspect,
                                                           .baseMipLevel = 0,
                                                           .levelCount = VK_REMAINING_MIP_LEVELS,
                                                           .baseArrayLayer = 0,
                                                           .layerCount = VK_REMAINING_ARRAY_LAYERS,
                                                       },
                                                   .components = {}}};
}

bool format_is_color(VkFormat format) {
  return !(format_is_stencil(format) || format_is_depth(format));
}
bool format_is_srgb(VkFormat format) {
  switch (format) {
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
bool format_is_depth(VkFormat format) {
  switch (format) {
    case VK_FORMAT_D32_SFLOAT:
    case VK_FORMAT_D16_UNORM:
      return true;
    default:
      return false;
  }
  return false;
}
bool format_is_stencil(VkFormat format) {
  switch (format) {
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
TextureView::TextureView(TextureView&& other) noexcept
    : view_(std::exchange(other.view_, nullptr)),
      create_info_(std::exchange(other.create_info_, {})),
      storage_image_resource_info_(std::exchange(other.storage_image_resource_info_, std::nullopt)),
      sampled_image_resource_info_(
          std::exchange(other.sampled_image_resource_info_, std::nullopt)) {}

TextureView& TextureView::operator=(TextureView&& other) noexcept {
  if (&other == this) {
    return *this;
  }
  this->~TextureView();

  view_ = std::exchange(other.view_, nullptr);
  create_info_ = std::exchange(other.create_info_, {});
  storage_image_resource_info_ = std::exchange(other.storage_image_resource_info_, std::nullopt);
  sampled_image_resource_info_ = std::exchange(other.sampled_image_resource_info_, std::nullopt);

  return *this;
}

TextureView::~TextureView() {
  if (view_) {
    BindlessResourceAllocator::get().delete_texture_view(
        {storage_image_resource_info_, sampled_image_resource_info_, view_});
    view_ = nullptr;
  }
}

Texture create_texture_2d(VkFormat format, uvec3 dims, TextureUsage usage) {
  return Texture{TextureCreateInfo{.view_type = VK_IMAGE_VIEW_TYPE_2D,
                                   .format = format,
                                   .extent = VkExtent3D{dims.x, dims.y, dims.z},
                                   .usage = usage}};
}
Texture create_texture_2d_mip(VkFormat format, uvec3 dims, TextureUsage usage, u32 levels) {
  return Texture{TextureCreateInfo{.view_type = VK_IMAGE_VIEW_TYPE_2D,
                                   .format = format,
                                   .extent = VkExtent3D{dims.x, dims.y, dims.z},
                                   .mip_levels = levels,
                                   .usage = usage}};
}

void blit_img(VkCommandBuffer cmd, VkImage src, VkImage dst, VkExtent3D extent,
              VkImageAspectFlags aspect) {
  VkImageBlit2 region{
      .sType = VK_STRUCTURE_TYPE_IMAGE_BLIT_2,
      .srcSubresource = {.aspectMask = aspect, .mipLevel = 0, .baseArrayLayer = 0, .layerCount = 1},
      .srcOffsets = {{},
                     {static_cast<i32>(extent.width), static_cast<i32>(extent.height),
                      static_cast<i32>(extent.depth)}},
      .dstSubresource = {.aspectMask = aspect, .mipLevel = 0, .baseArrayLayer = 0, .layerCount = 1},
      .dstOffsets = {{},
                     {static_cast<i32>(extent.width), static_cast<i32>(extent.height),
                      static_cast<i32>(extent.depth)}}

  };
  VkBlitImageInfo2 blit_info{.sType = VK_STRUCTURE_TYPE_BLIT_IMAGE_INFO_2,
                             .srcImage = src,
                             .srcImageLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                             .dstImage = dst,
                             .dstImageLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                             .regionCount = 1,
                             .pRegions = &region,
                             .filter = VK_FILTER_NEAREST};
  vkCmdBlitImage2KHR(cmd, &blit_info);
};

// Sampler::Sampler(const VkSamplerCreateInfo& info) {
//   sampler_ = SamplerCache::get().get_or_create_sampler(info);
//   assert(sampler_);
//   resource_info_ = BindlessResourceAllocator::get().allocate_sampler_descriptor(sampler_);
// }

uint32_t format_storage_size(VkFormat format) {
  switch (format) {
    case VK_FORMAT_R8_UNORM:
    case VK_FORMAT_R8_SNORM:
    case VK_FORMAT_R8_SINT:
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
bool format_is_block_compreesed(VkFormat format) {
  switch (format) {
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
u64 block_compressed_image_size(VkFormat format, uvec3 extent) {
  u64 rounded_w = (extent.x + 3) & ~3;
  u64 rounded_h = (extent.y + 3) & ~3;

  u64 num_blocks_w = rounded_w / 4;
  u64 num_blocks_h = rounded_h / 4;
  u64 num_blocks = num_blocks_w * num_blocks_h * extent.z;

  // BC1 and BC4 use 8 bytes per block, others use 16 bytes per block
  switch (format) {
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
u64 img_to_buffer_size(VkFormat format, uvec3 extent) {
  // https://registry.khronos.org/DataFormat/specs/1.3/dataformat.1.3.html#BPTC
  if (format_is_block_compreesed(format)) {
    return block_compressed_image_size(format, extent);
  }
  return static_cast<u64>(extent.x) * extent.y * extent.z * format_storage_size(format);
}
}  // namespace vk2
