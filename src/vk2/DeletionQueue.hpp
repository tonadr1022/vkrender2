#pragma once

#include <deque>
#include <functional>

namespace gfx::vk2 {

class DeletionQueue {
 public:
  void push(std::function<void()>&& func);
  void flush();
  ~DeletionQueue();

 private:
  std::deque<std::function<void()>> q_;
};

}  // namespace gfx::vk2
