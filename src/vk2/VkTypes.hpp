#pragma once

#include "Types.hpp"
namespace gfx::vk2 {

VkFormat to_vkformat(Format format);
VkImageAspectFlags format_to_aspect_flags(VkFormat format);
VkImageAspectFlags format_to_aspect_flags(Format format);

}  // namespace gfx::vk2
