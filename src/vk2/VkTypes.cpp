#include "VkTypes.hpp"
namespace gfx::vk2 {
VkFormat to_vkformat(Format format) { return static_cast<VkFormat>(format); }
}  // namespace gfx::vk2
