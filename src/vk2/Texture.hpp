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
  u32 sample_count{1};
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
  // explicit ImageView(const Image& texture, const ImageViewCreateInfo& info);
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
  Image() = default;
  ~Image();
  Image& operator=(const Image& other) = delete;
  Image(const Image& other) = delete;
  Image(Image&& other) noexcept;
  Image& operator=(Image&& other) noexcept;

  [[nodiscard]] VkImage image() const { return image_; }
  [[nodiscard]] Format format() const { return desc_.format; }
  [[nodiscard]] uvec3 size() const { return desc_.dims; }
  // [[nodiscard]] VkExtent2D extent_2d() const { return VkExtent2D{desc_.dims.x, desc_.dims.y}; }

  [[nodiscard]] ImageView& view() { return view_.value(); }
  [[nodiscard]] const ImageView& view() const { return view_.value(); }
  [[nodiscard]] const ImageDesc& get_desc() const { return desc_; }

  VkImageLayout curr_layout{};

 private:
  friend class Device;
  friend class BindlessResourceAllocator;
  friend class ImageView;

  ImageDesc desc_;
  std::optional<ImageView> view_;
  VkImage image_{};
  VmaAllocation allocation_{};
};

void blit_img(VkCommandBuffer cmd, VkImage src, VkImage dst, VkExtent3D extent,
              VkImageAspectFlags aspect);

// TODO: device handle deletions
struct TextureDeleteInfo {
  VkImage img;
  VmaAllocation allocation;
};

struct TextureViewDeleteInfo {
  std::optional<BindlessResourceInfo> storage_image_resource_info;
  std::optional<BindlessResourceInfo> sampled_image_resource_info;
  VkImageView view;
};

uint32_t get_mip_levels(VkExtent2D size);
uint32_t get_mip_levels(uvec2 size);
VkImageType vkviewtype_to_img_type(VkImageViewType view_type);
bool format_is_stencil(Format format);
bool format_is_depth(Format format);
bool format_is_srgb(Format format);
bool format_is_color(Format format);

uint32_t format_storage_size(Format format);
bool format_is_block_compreesed(Format format);
u64 block_compressed_image_size(Format format, uvec3 extent);
u64 img_to_buffer_size(Format format, uvec3 extent);

}  // namespace gfx
