#pragma once

#include "Types.hpp"
namespace gfx::util {

bool is_read_access(Access access);
bool is_write_access(Access access);
const char* to_string_access(Access access);

}  // namespace gfx::util
