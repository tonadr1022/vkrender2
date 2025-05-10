#pragma once

#include <cstddef>

namespace gfx::vk2 {

void print_vk_error(size_t x, bool exit_prog = false);

}  // namespace gfx::vk2

#ifndef NDEBUG
#define VK_CHECK(x)                      \
  do {                                   \
    ::gfx::vk2::print_vk_error(x, true); \
  } while (0)
#else
#define VK_CHECK(x)                       \
  do {                                    \
    ::gfx::vk2::print_vk_error(x, false); \
  } while (0)
#endif
