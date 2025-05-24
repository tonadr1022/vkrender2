#include "IndexAllocator.hpp"

#include "core/Logger.hpp"

namespace util {

void FreeListAllocator::init(u32 size_bytes, u32 alignment, u32 element_reserve_count) {
  allocs_.reserve(element_reserve_count);
  alignment_ = alignment;
  // align the size
  size_bytes += (alignment_ - (size_bytes % alignment_)) % alignment_;
  capacity_ = size_bytes;

  // create one large free block
  Slot empty_alloc{};
  empty_alloc.size_ = size_bytes;
  empty_alloc.offset_ = 0;
  empty_alloc.mark_free();
  allocs_.push_back(empty_alloc);
}
bool FreeListAllocator::reserve(u32 size_bytes) {
  // top bit in the offset is reserved
  if (size_bytes >= UINT32_MAX / 2) {
    return false;
  }
  if (size_bytes <= capacity_) {
    return true;
  }
  allocs_.emplace_back(Slot{capacity_, size_bytes - capacity_});
  coalesce(--allocs_.end());
  LINFO("reserving space: old cap {}, new cap {}", capacity_, size_bytes);
  capacity_ = size_bytes;
  return true;
}
FreeListAllocator::Slot FreeListAllocator::allocate(u32 size_bytes) {
  // align the size
  size_bytes += (alignment_ - (size_bytes % alignment_)) % alignment_;
  auto smallest_free_alloc = allocs_.end();
  {
    // find the smallest free allocation that is large enough
    for (auto it = allocs_.begin(); it != allocs_.end(); it++) {
      // adequate if free and size fits
      if (it->is_free() && it->get_size() >= size_bytes) {
        // if it's the first or it's smaller, set it to the new smallest free alloc
        if (smallest_free_alloc == allocs_.end() ||
            it->get_size() < smallest_free_alloc->get_size()) {
          smallest_free_alloc = it;
        }
      }
    }
    // if there isn't an allocation small enough, return 0, null handle
    if (smallest_free_alloc == allocs_.end()) {
      if (!reserve(capacity_ * 2)) {
        LWARN("failed to reserve greater capacity for FreeListAllocator");
        return {};
      }
      return allocate(size_bytes);
    }
  }

  u32 real_offset = smallest_free_alloc->get_offset();

  Slot new_alloc{real_offset, size_bytes};
  new_alloc.mark_used();

  // update free allocation
  smallest_free_alloc->size_ -= size_bytes;
  smallest_free_alloc->offset_ = (real_offset + size_bytes) | 0x80000000;

  ++num_active_allocs_;
  max_seen_active_allocs_ = std::max<u32>(max_seen_active_allocs_, num_active_allocs_);
  size_ += size_bytes;

  if (smallest_free_alloc->size_ == 0) {
    *smallest_free_alloc = new_alloc;
    Slot s{smallest_free_alloc->get_offset(), smallest_free_alloc->size_};
    s.mark_used();
    return s;
  }
  auto res = allocs_.insert(smallest_free_alloc, new_alloc);
  Slot s{res->get_offset(), res->size_};
  s.mark_used();
  return s;
}

u32 FreeListAllocator::free(Slot slot) {
  if (slot.size_ == 0) return 0;
  auto it = allocs_.end();
  u32 real_offset = slot.get_offset();

  for (it = allocs_.begin(); it != allocs_.end(); it++) {
    if (it->get_offset() == real_offset) break;
  }

  if (it == allocs_.end()) {
    LINFO("alloc not found offset: {} size: {}", real_offset, slot.size_);
    return 0;
  }

  size_ -= slot.size_;
  u32 ret = it->size_;
  it->mark_free();
  coalesce(it);
  --num_active_allocs_;
  return ret;
}
void FreeListAllocator::coalesce(Iterator& it) {
  assert(it != allocs_.end() && "Don't coalesce a non-existent allocation");
  bool remove_it = false;
  bool remove_next = false;

  // merge with next alloc
  if (it != allocs_.end() - 1) {
    auto next = it + 1;
    if (next->is_free()) {
      it->size_ += next->size_;
      remove_next = true;
    }
  }

  // merge with previous alloc
  if (it != allocs_.begin()) {
    auto prev = it - 1;
    if (prev->is_free()) {
      prev->size_ += it->size_;
      remove_it = true;
    }
  }

  // erase merged allocations
  if (remove_it && remove_next) {
    allocs_.erase(it, it + 2);  // curr and next
  } else if (remove_it) {
    allocs_.erase(it);  // only curr
  } else if (remove_next) {
    allocs_.erase(it + 1);  // only next
  }
}

FreeListAllocator::Slot::Slot(u32 offset, u32 size) : offset_(offset | 0x80000000), size_(size) {}

}  // namespace util
