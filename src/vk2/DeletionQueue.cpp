#include "vk2/DeletionQueue.hpp"

#include <ranges>
#include <tracy/Tracy.hpp>

namespace vk2 {

DeletionQueue::~DeletionQueue() { flush(); }

void DeletionQueue::push(std::function<void()>&& func) { q_.emplace_back(std::move(func)); }

void DeletionQueue::flush() {
  ZoneScoped;
  for (auto& it : std::ranges::reverse_view(q_)) {
    it();
  }
  q_.clear();
}
}  // namespace vk2
