#pragma once

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

}  // namespace util
