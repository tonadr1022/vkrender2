#include "StagingBufferPool.hpp"

#include <tracy/Tracy.hpp>

#include "vk2/Device.hpp"

namespace gfx {

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

Holder<BufferHandle> StagingBufferPool::acquire(u64 size) {
  ZoneScoped;
  {
    std::lock_guard lock(mtx_);
    for (size_t i = 0; i < free_buffers_.size(); i++) {
      if (get_device().get_buffer(free_buffers_[i].handle)->size() >= size) {
        auto result = std::move(free_buffers_[i]);
        if (free_buffers_.size() > 1) {
          free_buffers_[i] = std::move(free_buffers_.back());
        }
        free_buffers_.pop_back();
        return result;
      }
    }
  }
  return get_device().create_buffer_holder(
      BufferCreateInfo{.size = std::max<u64>(size, 4096), .flags = BufferCreateFlags_HostVisible});
}

void StagingBufferPool::free(Holder<BufferHandle>&& buffer) {
  ZoneScoped;
  std::lock_guard lock(mtx_);
  free_buffers_.emplace_back(std::move(buffer));
}

StagingBufferPool::~StagingBufferPool() {
  for (auto& buf : free_buffers_) {
    get_device().destroy(buf.handle);
  }
}

}  // namespace gfx
