#pragma once

#include <cstdint>
#include <vector>

#include "vk2/Hash.hpp"
template <typename, typename>
struct Pool;

// ObjectT should be default constructible and have sane default constructed state
template <typename HandleT, typename ObjectT>
struct Pool {
  static_assert(std::is_default_constructible_v<ObjectT>, "ObjectT must be default constructible");

  using IndexT = uint32_t;
  Pool() { entries_.reserve(20); }
  Pool& operator=(Pool&& other) = delete;
  Pool& operator=(const Pool& other) = delete;
  Pool(const Pool& other) = delete;
  Pool(Pool&& other) = delete;

  explicit Pool(IndexT size) : entries_(size) {}

  void clear() {
    entries_.clear();
    size_ = 0;
  }

  struct Entry {
    explicit Entry(auto&&... args) : object(std::forward<decltype(args)>(args)...) {}

    ObjectT object{};
    uint32_t gen_{1};
  };

  template <typename... Args>
  HandleT alloc(Args&&... args) {
    HandleT handle;
    if (!free_list_.empty()) {
      handle.idx_ = free_list_.back();
      free_list_.pop_back();
      ::new (std::addressof(entries_[handle.idx_].object)) ObjectT{std::forward<Args>(args)...};
    } else {
      handle.idx_ = entries_.size();
      entries_.emplace_back(std::forward<Args>(args)...);
    }
    handle.gen_ = entries_[handle.idx_].gen_;
    num_created_++;
    size_++;
    return handle;
  }

  [[nodiscard]] IndexT size() const { return size_; }
  [[nodiscard]] size_t get_num_created() const { return num_created_; }
  [[nodiscard]] size_t get_num_destroyed() const { return num_destroyed_; }

  void destroy(HandleT handle) {
    if (handle.idx_ >= entries_.size()) {
      return;
    }
    if (entries_[handle.idx_].gen_ != handle.gen_) {
      return;
    }
    entries_[handle.idx_].gen_++;
    entries_[handle.idx_].object = {};
    free_list_.emplace_back(handle.idx_);
    size_--;
    num_destroyed_++;
  }

  ObjectT* get(HandleT handle) {
    if (!handle.gen_) return nullptr;
    if (handle.idx_ >= entries_.size()) {
      return nullptr;
    }
    if (entries_[handle.idx_].gen_ != handle.gen_) {
      return nullptr;
    }
    return &entries_[handle.idx_].object;
  }

 private:
  IndexT get_new_idx() {}

  std::vector<IndexT> free_list_;
  std::vector<Entry> entries_;
  IndexT size_{};
  size_t num_created_{};
  size_t num_destroyed_{};
};

template <typename HandleT>
struct Handle {
  using HandleIdxT = uint32_t;
  Handle() = default;
  explicit Handle(HandleIdxT idx) : idx_(idx) {}
  friend bool operator==(const HandleT& a, const HandleT& b) { return a.idx_ == b.idx_; }

  [[nodiscard]] bool is_valid() const { return idx_ != null_handle; }
  [[nodiscard]] HandleIdxT idx() const { return idx_; }
  constexpr static HandleIdxT null_handle{UINT32_MAX};

 private:
  HandleIdxT idx_{null_handle};
};

template <typename HandleT>
struct GenerationalHandle {
  GenerationalHandle() = default;

  explicit GenerationalHandle(uint32_t idx, uint32_t gen) : idx_(idx), gen_(gen) {}

  [[nodiscard]] bool is_valid() const { return gen_ != 0; }

  [[nodiscard]] uint32_t get_gen() const { return gen_; }
  [[nodiscard]] uint32_t get_idx() const { return idx_; }
  friend bool operator!=(const GenerationalHandle& a, const GenerationalHandle& b) {
    return a.idx_ != b.idx_ || a.gen_ != b.gen_;
  }
  friend bool operator==(const GenerationalHandle& a, const GenerationalHandle& b) {
    return a.idx_ == b.idx_ && a.gen_ == b.gen_;
  }

  template <typename, typename>
  friend struct Pool;

 private:
  uint32_t idx_{};
  uint32_t gen_{};
};

namespace std {
template <typename HandleT>
struct hash<GenerationalHandle<HandleT>> {
  std::size_t operator()(const GenerationalHandle<HandleT>& handle) const noexcept {
    auto h = std::make_tuple(handle.get_idx(), handle.get_gen());
    return gfx::vk2::detail::hashing::hash<decltype(h)>{}(h);
  }
};

}  // namespace std
