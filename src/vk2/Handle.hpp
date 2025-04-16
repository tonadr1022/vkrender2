#pragma once
#include "Common.hpp"

template <typename Tag>
struct HandleOld {
  using HandleT = u64;

  constexpr HandleOld() : value_(0) {}
  constexpr explicit HandleOld(HandleT v) : value_(v) {}
  friend constexpr bool operator==(HandleOld a, HandleOld b) { return a.value_ == b.value_; }
  friend constexpr bool operator!=(HandleOld a, HandleOld b) { return a.value_ != b.value_; }
  friend constexpr bool operator<(HandleOld a, HandleOld b) { return a.value_ < b.value_; }
  constexpr HandleT operator()() const { return value_; }
  [[nodiscard]] constexpr HandleT get() const { return value_; }
  explicit operator bool() { return value_ != 0; }

 private:
  HandleT value_;
};

namespace std {
template <typename Tag>
struct hash<HandleOld<Tag>> {
  std::size_t operator()(const HandleOld<Tag>& obj) const noexcept {
    return std::hash<u32>{}(obj());
  }
};
}  // namespace std

#define VK2_DEFINE_HANDLE(x) \
  struct x##Tag {};          \
  using x##Handle = HandleOld<x##Tag>;

#define VK2_DEFINE_HANDLE_WITH_NAME(name, type) \
  struct type##Tag {};                          \
  using name##Handle = HandleOld<type##Tag>;
