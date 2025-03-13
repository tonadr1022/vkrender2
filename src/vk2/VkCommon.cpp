#include "VkCommon.hpp"

#include <vulkan/vk_enum_string_helper.h>

#include "Logger.hpp"

namespace vk2 {

void print_vk_error(size_t x) {
  auto err = static_cast<VkResult>(x);
  if (err) {
    LERROR("Detected Vulkan error: {}", string_VkResult(err));
    // exit(1);
  }
}

}  // namespace vk2
