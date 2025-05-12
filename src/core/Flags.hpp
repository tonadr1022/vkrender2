#pragma once

#include <type_traits>
template <typename E>
struct EnableBitmaskOperators {
  static constexpr bool enable = false;
};
template <typename E>
constexpr E operator|(E lhs, E rhs)
  requires EnableBitmaskOperators<E>::enable
{
  using underlying = std::underlying_type_t<E>;
  return static_cast<E>(static_cast<underlying>(lhs) | static_cast<underlying>(rhs));
}
template <typename E>
constexpr E& operator|=(E& lhs, E rhs)
  requires EnableBitmaskOperators<E>::enable
{
  using underlying = std::underlying_type_t<E>;
  lhs = static_cast<E>(static_cast<underlying>(lhs) | static_cast<underlying>(rhs));
  return lhs;
}
template <typename E>
constexpr E operator&(E lhs, E rhs)
  requires EnableBitmaskOperators<E>::enable
{
  using underlying = std::underlying_type_t<E>;
  return static_cast<E>(static_cast<underlying>(lhs) & static_cast<underlying>(rhs));
}
template <typename E>
constexpr E& operator&=(E& lhs, E rhs)
  requires EnableBitmaskOperators<E>::enable
{
  using underlying = std::underlying_type_t<E>;
  lhs = static_cast<E>(static_cast<underlying>(lhs) & static_cast<underlying>(rhs));
  return lhs;
}
template <typename E>
constexpr E operator~(E rhs)
  requires EnableBitmaskOperators<E>::enable
{
  using underlying = std::underlying_type_t<E>;
  rhs = static_cast<E>(~static_cast<underlying>(rhs));
  return rhs;
}
template <typename E>
constexpr bool has_flag(E lhs, E rhs) {
  return (size_t)(lhs & rhs) != 0;
}
