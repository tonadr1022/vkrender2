#include "RenderGraphUtil.hpp"

#include "Types.hpp"
namespace gfx::util {

bool is_read_access(Access access) {
  return access == Access::ColorRead || access == Access::DepthStencilRead;
}

bool is_write_access(Access access) { return !is_read_access(access); }

const char* to_string_access(Access access) {
  switch (access) {
    case Access::ColorRead:
      return "ColorRead";
    case Access::ColorWrite:
      return "ColorWrite";
    case Access::DepthStencilRead:
      return "DepthStencilRead";
    case Access::DepthStencilWrite:
      return "DepthStencilWrite";
  }
}
}  // namespace gfx::util
