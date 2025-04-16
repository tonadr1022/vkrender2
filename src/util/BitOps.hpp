#pragma once

#include <bit>

#include "Common.hpp"

namespace util {

template <typename F>
void for_each_bit(u64 value, const F& func) {
  while (value) {
    u32 bit = std::countr_zero(value);
    func(bit);
    value &= ~(1ull << bit);
  }
}

template <typename F>
void for_each_bit(u32 value, const F& func) {
  while (value) {
    u32 bit = std::countr_zero(value);
    func(bit);
    value &= ~(1u << bit);
  }
}

}  // namespace util
