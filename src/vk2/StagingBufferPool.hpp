#pragma once

#include <mutex>
#include <vector>

#include "Types.hpp"
#include "vk2/Pool.hpp"

namespace gfx {

struct StagingBufferPool {
  static StagingBufferPool& get();
  static void shutdown();
  ~StagingBufferPool();
  static void destroy();
  static void init();
  Holder<BufferHandle> acquire(u64 size);
  void free(Holder<BufferHandle>&& buffer);

 private:
  std::mutex mtx_;
  std::vector<Holder<BufferHandle>> free_buffers_;
  std::vector<Holder<BufferHandle>> allocated_buffers_;
};
}  // namespace gfx
