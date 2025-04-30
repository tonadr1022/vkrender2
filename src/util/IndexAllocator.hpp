#pragma once

#include <cassert>
#include <span>
#include <vector>

#include "Common.hpp"

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

  void init(u32 size_bytes, u32 alignment, u32 element_reserve_count);

  // returns true if success
  bool reserve(u32 size_bytes);

  [[nodiscard]] constexpr u32 alloc_size() const { return sizeof(Slot); }

  [[nodiscard]] u32 capacity() const { return capacity_; }
  [[nodiscard]] Slot allocate(u32 size_bytes);

  // returns number of bytes freed
  u32 free(Slot slot);

  [[nodiscard]] u32 num_active_allocs() const { return num_active_allocs_; }
  [[nodiscard]] u32 max_seen_active_allocs() const { return max_seen_active_allocs_; }

 private:
  u32 alignment_{0};
  u32 num_active_allocs_{0};
  u32 max_seen_active_allocs_{0};
  u32 capacity_{};

  std::vector<Slot> allocs_;

  using Iterator = decltype(allocs_.begin());
  void coalesce(Iterator& it);
};

}  // namespace util
