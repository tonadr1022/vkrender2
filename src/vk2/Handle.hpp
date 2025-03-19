#pragma once
#include "Common.hpp"

template <typename Tag>
struct Handle {
  using HandleT = u64;

  constexpr Handle() : value_(0) {}
  constexpr explicit Handle(HandleT v) : value_(v) {}
  friend constexpr bool operator==(Handle a, Handle b) { return a.value_ == b.value_; }
  friend constexpr bool operator!=(Handle a, Handle b) { return a.value_ != b.value_; }
  friend constexpr bool operator<(Handle a, Handle b) { return a.value_ < b.value_; }
  constexpr HandleT operator()() const { return value_; }
  [[nodiscard]] constexpr HandleT get() const { return value_; }
  explicit operator bool() { return value_ != 0; }

 private:
  HandleT value_;
};

namespace std {
template <typename Tag>
struct hash<Handle<Tag>> {
  std::size_t operator()(const Handle<Tag>& obj) const noexcept { return std::hash<u32>{}(obj()); }
};
}  // namespace std

#define VK2_DEFINE_HANDLE(x) \
  struct x##Tag {};          \
  using x##Handle = Handle<x##Tag>;

#define VK2_DEFINE_HANDLE_WITH_NAME(name, type) \
  struct type##Tag {};                          \
  using name##Handle = Handle<type##Tag>;
