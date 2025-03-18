#pragma once

#include <vector>

#include "Common.hpp"

namespace util {

struct IndexAllocator {
  explicit IndexAllocator(u32 size, bool expandable = true);
  void free(u32 idx);
  [[nodiscard]] u32 alloc();

 private:
  bool expandable_{true};
  std::vector<u32> free_list_;
  u32 next_index_{};
};

}  // namespace util
