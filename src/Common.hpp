#pragma once

#include <cstddef>
#include <cstdint>

using u8 = uint8_t;
using b8 = bool;
using u16 = uint16_t;
using u32 = uint32_t;
using u64 = uint64_t;
using i8 = int8_t;
using i16 = int16_t;
using i32 = int32_t;
using i64 = int64_t;
// TODO: when stdfloat comes out on all compilers use that. for now this is for fun
using f32 = float;
using f64 = double;

template <typename T, size_t Size>
char (*countof_helper(T (&Array_omgwow_123454321)[Size]))[Size];
#define COUNTOF(array) (sizeof(*countof_helper(array)) + 0)
