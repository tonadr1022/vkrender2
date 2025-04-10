#pragma once

#include <vulkan/vulkan_core.h>
namespace gfx::vk2 {

void set_viewport_and_scissor(VkCommandBuffer cmd, VkExtent2D extent);
}
