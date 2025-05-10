#pragma once

#include "Types.hpp"

namespace gfx::vk2 {

constexpr VkFormat to_vkformat(Format format) { return static_cast<VkFormat>(format); }
Format vkformat_to_format(VkFormat format);
VkImageAspectFlags format_to_aspect_flags(VkFormat format);
VkImageAspectFlags format_to_aspect_flags(Format format);
VkCullModeFlags convert_cull_mode(CullMode mode);
VkImageViewType convert_image_view_type(ImageViewType type);

}  // namespace gfx::vk2
