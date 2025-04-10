#pragma once

#include <algorithm>
#include <cstring>
#include <memory>
#include <utility>

namespace util {
// https://gist.github.com/ThePhD/8153067
template <typename T, std::size_t N, std::size_t A = std::alignment_of_v<T>>
class fixed_vector {
 public:
  using value_type = T;
  using reference = T&;
  using const_reference = const T&;
  using pointer_type = T*;
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;
  using iterator = pointer_type;
  using const_iterator = const pointer_type;

 private:
  alignas(A) std::byte items_[sizeof(T) * N];
  std::size_t len_;

  constexpr T* ptrat(std::size_t idx) { return static_cast<T*>(static_cast<void*>(&items_)) + idx; }

  constexpr const T* ptrat(std::size_t idx) const {
    return static_cast<const T*>(static_cast<const void*>(&items_)) + idx;
  }

  constexpr T& refat(std::size_t idx) { return *ptrat(idx); }

  constexpr const T& refat(std::size_t idx) const { return *ptrat(idx); }

 public:
  constexpr static std::size_t max_size() { return N; }

  constexpr fixed_vector() : len_(0) { memset(&items_, 0, sizeof(T) * N); }

  constexpr explicit fixed_vector(std::size_t capacity) : len_(std::min(N, capacity)) {
    memset(&items_, 0, sizeof(T) * N);
  }

  template <std::size_t C>
  constexpr fixed_vector(const T (&arr)[C]) : len_(C) {
    memset(&items_, 0, sizeof(T) * N);
    static_assert(C < N, "Array too large to initialize fixed_vector");
    std::copy(std::addressof(arr[0]), std::addressof(arr[C]), data());
  }

  constexpr fixed_vector(std::initializer_list<T> initializer)
      : len_(std::min(N, initializer.size())) {
    memset(&items_, 0, sizeof(T) * N);
    std::copy(initializer.begin(), initializer.begin() + len_, data());
  }

  constexpr fixed_vector(const fixed_vector& o) {
    memset(&items_, 0, sizeof(T) * N);
    std::uninitialized_copy(o.begin(), o.end(), begin());
    len_ = o.len_;
  }

  constexpr fixed_vector& operator=(const fixed_vector& o) {
    auto existing = std::min(len_, o.len_);
    std::copy_n(o.begin(), existing, begin());
    std::uninitialized_copy(o.begin() + existing, o.end(), begin() + existing);
    resize(o.len_);
    return *this;
  }

  constexpr fixed_vector(fixed_vector&& o) noexcept {
    memset(&items_, 0, sizeof(T) * N);
    std::uninitialized_move(o.begin(), o.end(), begin());
    len_ = o.len_;
    o.resize(0);
  }

  constexpr fixed_vector& operator=(fixed_vector&& o) noexcept {
    auto existing = std::min(len_, o.len_);
    std::copy_n(std::make_move_iterator(o.begin()), existing, begin());
    std::uninitialized_move(o.begin() + existing, o.end(), begin() + existing);
    resize(o.len_);
    o.resize(0);
    return *this;
  }

  constexpr ~fixed_vector() {
    for (std::size_t i = 0; i < len_; i++) {
      ptrat(i)->~T();
    }
  }

  [[nodiscard]] constexpr bool empty() const { return len_ < 1; }

  [[nodiscard]] constexpr bool not_empty() const { return len_ > 0; }

  [[nodiscard]] constexpr bool full() const { return len_ >= N; }

  constexpr void push_back(const T& item) { new (ptrat(len_++)) T(item); }

  constexpr void push_back(T&& item) { new (ptrat(len_++)) T(std::move(item)); }

  template <typename... Tn>
  constexpr T& emplace_back(Tn&&... argn) {
    return *(new (ptrat(len_++)) T(std::forward<Tn>(argn)...));
  }

  constexpr void pop_back() {
    T& addr = refat(--len_);
    addr.~T();
  }

  constexpr void clear() {
    for (; len_ > 0;) {
      pop_back();
    }
  }

  [[nodiscard]] constexpr std::size_t size() const { return len_; }

  [[nodiscard]] constexpr std::size_t capacity() const { return N; }

  constexpr void resize(std::size_t sz) {
    auto old_len = len_;
    while (len_ > sz) pop_back();
    if (old_len > len_) {
      memset(reinterpret_cast<char*>(&items_) + (len_ * sizeof(T)), 0,
             sizeof(T) * (old_len - len_));
    }
    len_ = std::min(sz, N);
  }

  constexpr void resize(std::size_t sz, const value_type& value) {
    auto old_len = len_;
    while (len_ > sz) pop_back();
    if (old_len > len_) {
      memset(reinterpret_cast<char*>(&items_) + (len_ * sizeof(T)), 0,
             sizeof(T) * (old_len - len_));
    }

    len_ = std::min(sz, N);
    if (len_ > old_len) {
      std::uninitialized_fill(begin() + old_len, begin() + len_, value);
    }
  }

  constexpr T* data() { return ptrat(0); }

  constexpr const T* data() const { return ptrat(0); }

  constexpr T& operator[](std::size_t idx) { return refat(idx); }

  constexpr const T& operator[](std::size_t idx) const { return refat(idx); }

  constexpr T& front() { return refat(0); }

  constexpr T& back() { return refat(len_ - 1); }

  constexpr const T& front() const { return refat(0); }

  constexpr const T& back() const { return refat(len_ - 1); }

  constexpr T* begin() { return data(); }

  constexpr const T* cbegin() { return data(); }

  constexpr const T* begin() const { return data(); }

  constexpr const T* cbegin() const { return data(); }

  constexpr T* end() { return data() + len_; }

  constexpr const T* cend() { return data() + len_; }

  constexpr const T* end() const { return data() + len_; }

  constexpr const T* cend() const { return data() + len_; }

  /*
  iterator insert(const_iterator pos, const T& value);
  iterator insert(const_iterator pos, T&& value);
  iterator insert( const_iterator pos, std::initializer_list<T> ilist );
  */

  template <class InputIt>
  constexpr iterator insert(const_iterator pos, InputIt first, InputIt last) {
    auto clen = std::distance(first, last);
    if (clen == 0) return pos;
    std::copy(pos, pos + (len_ - std::distance(begin(), pos)), pos + clen);
    std::copy(first, last, pos);
    len_ += clen;
    return pos;
  }

  constexpr bool operator==(const fixed_vector& o) const noexcept {
    return std::equal(begin(), end(), o.begin(), o.end());
  }
};
}  // namespace util
