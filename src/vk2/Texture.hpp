#pragma once

#include <vk_mem_alloc.h>
#include <vulkan/vulkan_core.h>

#include <array>
#include <optional>

#include "Types.hpp"
#include "vk2/Resource.hpp"

namespace gfx {

enum class ImageUsage : u8 {
  // general computation, i.e. StorageImage
  General,
  // asset textures read from shader
  ReadOnly,
  // attachment textures like gbuffer
  AttachmentReadOnly
};

struct ImageDesc {
  enum class Type : u8 { OneD, TwoD, ThreeD };
  Type type{Type::TwoD};
  Format format{Format::Undefined};
  uvec3 dims{};
  u32 mip_levels{1};
  u32 array_layers{1};
  u32 sample_count{};
  BindFlag bind_flags{};
  ResourceMiscFlag misc_flags{};
  Usage usage{Usage::Default};
};

struct ImageCreateInfo {
  VkImageViewType view_type{};
  VkFormat format{};
  VkExtent3D extent{};
  u32 mip_levels{1};
  u32 array_layers{1};
  VkSampleCountFlagBits samples{VK_SAMPLE_COUNT_1_BIT};
  ImageUsage usage{ImageUsage::General};
  VkImageUsageFlags override_usage_flags{};
  bool make_view{true};
};

struct ImageViewCreateInfo {
  VkFormat format;
  VkImageSubresourceRange range;
  VkComponentMapping components{};
  VkImageViewType view_type{VK_IMAGE_VIEW_TYPE_MAX_ENUM};
};

class Image;
class ImageView {
 public:
  explicit ImageView(const Image& texture, const ImageViewCreateInfo& info);
  ImageView() = default;
  ~ImageView();
  ImageView(ImageView&& other) noexcept;
  ImageView& operator=(ImageView&& other) noexcept;
  ImageView(const ImageView&) = delete;
  ImageView& operator=(const ImageView&) = delete;

  // TODO: use pointers or optionals?
  [[nodiscard]] const BindlessResourceInfo& storage_img_resource() const {
    return storage_image_resource_info_.value();
  }
  [[nodiscard]] const BindlessResourceInfo& sampled_img_resource() const {
    return sampled_image_resource_info_.value();
  }
  [[nodiscard]] VkImageView view() const { return view_; }

 private:
  friend class Device;
  VkImageView view_{};
  ImageViewCreateInfo create_info_;
  // TODO: make a bindless texture view class for this
  std::optional<BindlessResourceInfo> storage_image_resource_info_;
  std::optional<BindlessResourceInfo> sampled_image_resource_info_;
};

class Image {
 public:
  explicit Image(const ImageCreateInfo& create_info);
  Image() = default;
  ~Image();
  Image& operator=(const Image& other) = delete;
  Image(const Image& other) = delete;
  Image(Image&& other) noexcept;
  Image& operator=(Image&& other) noexcept;

  [[nodiscard]] VkExtent2D extent_2d() const {
    return {create_info_.extent.width, create_info_.extent.height};
  }
  [[nodiscard]] VkExtent3D extent() const { return create_info_.extent; }
  [[nodiscard]] VkImage image() const { return image_; }
  [[nodiscard]] VkFormat format() const { return create_info_.format; }

  [[nodiscard]] ImageView& view() { return view_.value(); }
  [[nodiscard]] const ImageView& view() const { return view_.value(); }
  [[nodiscard]] const ImageCreateInfo& create_info() const { return create_info_; }

  [[nodiscard]] VkImageUsageFlags usage() const { return usage_; }

  VkImageLayout curr_layout{};

 private:
  friend class Device;
  friend class BindlessResourceAllocator;
  friend class ImageView;

  ImageCreateInfo create_info_;
  std::optional<ImageView> view_;
  VkImage image_{};
  VkImageUsageFlags usage_{};
  VmaAllocation allocation_{};
};

struct TextureCubeAndViews {
  explicit TextureCubeAndViews(const ImageCreateInfo& info);
  std::optional<Image> texture;
  std::array<std::optional<ImageView>, 6> img_views;
};

void blit_img(VkCommandBuffer cmd, VkImage src, VkImage dst, VkExtent3D extent,
              VkImageAspectFlags aspect);

struct TextureDeleteInfo {
  VkImage img;
  VmaAllocation allocation;
};

struct TextureViewDeleteInfo {
  std::optional<BindlessResourceInfo> storage_image_resource_info;
  std::optional<BindlessResourceInfo> sampled_image_resource_info;
  VkImageView view;
};

Image create_texture_2d(VkFormat format, uvec3 dims, ImageUsage usage);
Image create_texture_2d_mip(VkFormat format, uvec3 dims, ImageUsage usage, u32 levels);

uint32_t get_mip_levels(VkExtent2D size);
uint32_t get_mip_levels(uvec2 size);
VkImageType vkviewtype_to_img_type(VkImageViewType view_type);
bool format_is_stencil(VkFormat format);
bool format_is_depth(VkFormat format);
bool format_is_srgb(VkFormat format);
bool format_is_color(VkFormat format);

uint32_t format_storage_size(VkFormat format);
bool format_is_block_compreesed(VkFormat format);
u64 block_compressed_image_size(VkFormat format, uvec3 extent);
u64 img_to_buffer_size(VkFormat format, uvec3 extent);

}  // namespace gfx
