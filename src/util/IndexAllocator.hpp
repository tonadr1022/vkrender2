#pragma once

#include <cassert>
#include <span>
#include <vector>

#include "Common.hpp"
#include "Logger.hpp"

namespace util {

struct IndexAllocator {
  explicit IndexAllocator(u32 size = 64, bool expandable = true);
  void free(u32 idx);
  [[nodiscard]] u32 alloc();

 private:
  bool expandable_{true};
  std::vector<u32> free_list_;
  u32 next_index_{};
};

template <typename T>
struct SlotAllocator {
  explicit SlotAllocator(u64 size) : count_(size) {
    free_list_.resize(size);
    for (u64 i = 0; i < size; i++) {
      free_list_[i] = Slot{size - 1 - i};
    }
  }
  struct Slot {
    Slot() : idx_(0) {}
    explicit Slot(u64 idx) : idx_(idx) {}
    [[nodiscard]] u64 offset() const { return idx_ * sizeof(T); }
    [[nodiscard]] u64 idx() const { return idx_; }

   private:
    u64 idx_;
  };

  Slot alloc() {
    if (free_list_.empty()) {
      assert(0);
      u64 idx = count_++;
      return Slot{idx};
    }
    auto slot = free_list_.back();
    free_list_.pop_back();
    return slot;
  }
  // TODO: resizeable
  void alloc_range(std::span<Slot> result) {
    for (auto& r : result) {
      if (free_list_.size()) {
        r = free_list_.back();
        free_list_.pop_back();
      } else {
        break;
      }
    }
  }

  void free(Slot slot) { free_list_.emplace_back(slot); }

  [[nodiscard]] u64 size() const { return count_; }
  [[nodiscard]] u64 size_bytes() const { return count_ * sizeof(T); }

 private:
  std::vector<Slot> free_list_;
  u64 count_;
};

class FreeListAllocator {
 public:
  FreeListAllocator() = default;
  struct Slot {
    u32 offset{};
    u32 size{};
    [[nodiscard]] bool valid() const { return size != 0; }
    [[nodiscard]] bool is_free() const { return valid() && (offset & 0x80000000) != 0; }
    void mark_free() { offset |= 0x80000000; }
    void mark_used() { offset &= 0x7FFFFFFF; }
    [[nodiscard]] u32 get_offset() const { return offset & 0x7FFFFFFF; }
  };

  void init(u32 size_bytes, u32 alignment, u32 element_reserve_count) {
    allocs_.reserve(element_reserve_count);
    alignment_ = alignment;
    // align the size
    size_bytes += (alignment_ - (size_bytes % alignment_)) % alignment_;
    capacity_ = size_bytes;

    // create one large free block
    Slot empty_alloc{};
    empty_alloc.size = size_bytes;
    empty_alloc.offset = 0;
    empty_alloc.mark_free();
    allocs_.push_back(empty_alloc);
  }

  // returns true if success
  bool reserve(u32 size_bytes) {
    LINFO("reserving space: old cap {}, new cap {}", capacity_, capacity_ + size_bytes);
    // top bit in the offset is reserved
    if (size_bytes >= UINT32_MAX / 2) {
      return false;
    }
    if (size_bytes <= capacity_) {
      return true;
    }
    Slot empty_alloc{.offset = capacity_, .size = size_bytes - capacity_};
    empty_alloc.mark_free();
    allocs_.emplace_back(empty_alloc);
    return true;
  }

  [[nodiscard]] constexpr u32 alloc_size() const { return sizeof(Slot); }

  [[nodiscard]] u32 capacity() const { return capacity_; }
  [[nodiscard]] Slot allocate(u32 size_bytes) {
    // align the size
    size_bytes += (alignment_ - (size_bytes % alignment_)) % alignment_;
    auto smallest_free_alloc = allocs_.end();
    {
      // find the smallest free allocation that is large enough
      for (auto it = allocs_.begin(); it != allocs_.end(); it++) {
        // adequate if free and size fits
        if (it->is_free() && it->size >= size_bytes) {
          // if it's the first or it's smaller, set it to the new smallest free alloc
          if (smallest_free_alloc == allocs_.end() || it->size < smallest_free_alloc->size) {
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

    Slot new_alloc{};
    new_alloc.offset = real_offset;
    new_alloc.size = size_bytes;
    new_alloc.mark_used();

    // update free allocation
    smallest_free_alloc->size -= size_bytes;
    smallest_free_alloc->offset = (real_offset + size_bytes) | 0x80000000;

    if (smallest_free_alloc->size == 0) {
      *smallest_free_alloc = new_alloc;
    } else {
      allocs_.insert(smallest_free_alloc, new_alloc);
    }

    ++num_active_allocs_;
    max_seen_active_allocs_ = std::max<u32>(max_seen_active_allocs_, num_active_allocs_);
    return new_alloc;
  }

  // returns number of bytes freed
  u32 free(Slot slot) {
    if (slot.size == 0) return 0;
    auto it = allocs_.end();
    u32 real_offset = slot.get_offset();

    for (it = allocs_.begin(); it != allocs_.end(); it++) {
      if (it->get_offset() == real_offset) break;
    }

    if (it == allocs_.end()) {
      LINFO("alloc not found offset: {} size: {}", real_offset, slot.size);
      return 0;
    }

    u32 ret = it->size;
    it->mark_free();
    coalesce(it);
    --num_active_allocs_;
    return ret;
  }

  [[nodiscard]] u32 num_active_allocs() const { return num_active_allocs_; }
  [[nodiscard]] u32 max_seen_active_allocs() const { return max_seen_active_allocs_; }

 private:
  u32 alignment_{0};
  u32 num_active_allocs_{0};
  u32 max_seen_active_allocs_{0};
  u32 capacity_{};

  std::vector<Slot> allocs_;

  using Iterator = decltype(allocs_.begin());
  void coalesce(Iterator& it) {
    assert(it != allocs_.end() && "Don't coalesce a non-existent allocation");
    bool remove_it = false;
    bool remove_next = false;

    // merge with next alloc
    if (it != allocs_.end() - 1) {
      auto next = it + 1;
      if (next->is_free()) {
        it->size += next->size;
        remove_next = true;
      }
    }

    // merge with previous alloc
    if (it != allocs_.begin()) {
      auto prev = it - 1;
      if (prev->is_free()) {
        prev->size += it->size;
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
};

}  // namespace util
