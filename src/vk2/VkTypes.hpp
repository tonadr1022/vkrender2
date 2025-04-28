#pragma once

#include "Types.hpp"
namespace gfx::vk2 {

constexpr VkFormat to_vkformat(Format format) { return static_cast<VkFormat>(format); }
Format vkformat_to_format(VkFormat format);
VkImageAspectFlags format_to_aspect_flags(VkFormat format);
VkImageAspectFlags format_to_aspect_flags(Format format);

}  // namespace gfx::vk2
