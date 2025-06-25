#pragma once

#include <cassert>
#include <span>
#include <vector>

#include "Common.hpp"

namespace util {

struct IndexAllocator {
  explicit IndexAllocator(u32 size = 64);
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
  FreeListAllocator(const FreeListAllocator&) = delete;
  FreeListAllocator(FreeListAllocator&&) = delete;
  FreeListAllocator& operator=(const FreeListAllocator&) = delete;
  FreeListAllocator& operator=(FreeListAllocator&&) = delete;
  // ~FreeListAllocator() { assert(size_ == 0); }
  struct Slot {
    friend class FreeListAllocator;
    Slot() = default;

   private:
    Slot(u32 offset, u32 size);
    u32 offset_{};
    u32 size_{};

   public:
    [[nodiscard]] bool valid() const { return size_ != 0; }
    [[nodiscard]] bool is_free() const { return valid() && (offset_ & 0x80000000) != 0; }
    void mark_free() { offset_ |= 0x80000000; }
    void mark_used() { offset_ &= 0x7FFFFFFF; }
    [[nodiscard]] u32 get_offset() const { return offset_ & 0x7FFFFFFF; }
    [[nodiscard]] u32 get_size() const { return size_; }
    [[nodiscard]] u32 get_off_plus_size() const { return get_offset() + get_size(); }
  };

  void init(u32 size_bytes, u32 alignment, u32 element_reserve_count = 100);

  // returns true if success
  bool reserve(u32 size_bytes);

  [[nodiscard]] constexpr u32 alloc_size() const { return sizeof(Slot); }

  [[nodiscard]] u32 capacity() const { return capacity_; }
  [[nodiscard]] Slot allocate(u32 size_bytes);

  // returns number of bytes freed
  u32 free(Slot slot);

  [[nodiscard]] u32 num_active_allocs() const { return num_active_allocs_; }
  [[nodiscard]] u32 max_seen_active_allocs() const { return max_seen_active_allocs_; }
  [[nodiscard]] u32 max_seen_size() const { return max_seen_size_; }

 private:
  u32 size_{};
  u32 alignment_{};
  u32 num_active_allocs_{};
  u32 max_seen_active_allocs_{};
  u32 max_seen_size_{};
  u32 capacity_{};
  std::vector<Slot> allocs_;
  bool initialized_{};

  using Iterator = decltype(allocs_.begin());
  void coalesce(Iterator& it);
};

}  // namespace util
