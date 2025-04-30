#include "StagingBufferPool.hpp"

#include <tracy/Tracy.hpp>

namespace gfx::vk2 {

namespace {
StagingBufferPool* instance{};
}

StagingBufferPool& StagingBufferPool::get() {
  assert(instance);
  return *instance;
}

void StagingBufferPool::init() {
  assert(!instance);
  instance = new StagingBufferPool;
}

void StagingBufferPool::destroy() {
  assert(instance);
  delete instance;
}

vk2::Buffer* StagingBufferPool::acquire(u64 size) {
  ZoneScoped;
  {
    std::lock_guard lock(mtx_);
    for (auto it = free_buffers_.begin(); it != free_buffers_.end(); it++) {
      if (it->get()->size() >= size) {
        allocated_buffers_.push_back(std::move(*it));
        free_buffers_.erase(it);
        return allocated_buffers_.back().get();
      }
    }
  }
  auto new_buf =
      std::make_unique<vk2::Buffer>(vk2::BufferCreateInfo{.size = std::max<u64>(size, 4096),
                                                          .flags = BufferCreateFlags_HostVisible,
                                                          .debug_name = "Staging buffer"});
  {
    std::lock_guard lock(mtx_);
    allocated_buffers_.emplace_back(std::move(new_buf));
    return allocated_buffers_.back().get();
  }
}

void StagingBufferPool::free(vk2::Buffer* buffer) {
  ZoneScoped;
  assert(buffer);
  std::lock_guard lock(mtx_);
  for (auto it = allocated_buffers_.begin(); it != allocated_buffers_.end(); it++) {
    if (it->get() == buffer) {
      free_buffers_.push_back(std::move(*it));
      allocated_buffers_.erase(it);
      return;
    }
  }
}
}  // namespace gfx::vk2
