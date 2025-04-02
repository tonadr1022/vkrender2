#pragma once

#include <cstddef>
#include <memory>

namespace vk2 {
constexpr int min_api_version_major = 1;
// TODO: runtime selection
#ifdef __APPLE__
constexpr int min_api_version_minor = 2;
#else
constexpr int min_api_version_minor = 3;
#endif

void print_vk_error(size_t x, bool exit_prog = false);

template <class T>
[[nodiscard]] T* addr(T&& v) {
  return std::addressof(v);
}

}  // namespace vk2

#ifndef NDEBUG
#define VK_CHECK(x)                 \
  do {                              \
    ::vk2::print_vk_error(x, true); \
  } while (0)
#else
#define VK_CHECK(x)                  \
  do {                               \
    ::vk2::print_vk_error(x, false); \
  } while (0)
#endif
