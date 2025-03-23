#pragma once

#include <cstddef>
#include <cstdint>
#include <glm/mat4x4.hpp>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>

#include "glm/ext/quaternion_float.hpp"

using vec2 = glm::vec2;
using vec3 = glm::vec3;
using mat4 = glm::mat4;
using ivec2 = glm::ivec2;
using uvec2 = glm::uvec2;
using uvec3 = glm::uvec3;
using quat = glm::quat;

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
