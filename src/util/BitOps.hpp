#pragma once

#include <bit>

#include "Common.hpp"

namespace util {

inline u32 trailing_zeros(u64 value) { return std::countr_zero(value); }
inline u32 trailing_zeros(u32 value) { return std::countr_zero(value); }

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
