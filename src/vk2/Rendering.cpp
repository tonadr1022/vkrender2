#include "Rendering.hpp"

#include <volk.h>

namespace gfx::vk2 {
void set_viewport_and_scissor(VkCommandBuffer cmd, VkExtent2D extent) {
  VkViewport viewport{.x = 0,
                      .y = 0,
                      .width = static_cast<float>(extent.width),
                      .height = static_cast<float>(extent.height),
                      .minDepth = 0.f,
                      .maxDepth = 1.f};

  vkCmdSetViewport(cmd, 0, 1, &viewport);
  VkRect2D scissor{.offset = VkOffset2D{.x = 0, .y = 0},
                   .extent = VkExtent2D{.width = extent.width, .height = extent.height}};
  vkCmdSetScissor(cmd, 0, 1, &scissor);
}
}  // namespace gfx::vk2
