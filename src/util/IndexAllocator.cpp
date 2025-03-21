#include "IndexAllocator.hpp"

namespace util {

u32 IndexAllocator::alloc() {
  if (!expandable_) {
    assert(free_list_.size());
    if (free_list_.empty()) {
      return UINT32_MAX;
    }
  } else if (free_list_.empty()) {
    return next_index_++;
  }

  auto ret = free_list_.back();
  free_list_.pop_back();
  return ret;
}

void IndexAllocator::free(u32 idx) { free_list_.push_back(idx); }

IndexAllocator::IndexAllocator(u32 size, bool expandable) : expandable_(expandable) {
  free_list_.reserve(size);
  u32 j = 0;
  while (j < size) {
    free_list_.push_back(size - j);
    next_index_++;
    j++;
  }
}

}  // namespace util
