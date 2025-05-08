#pragma once

#include <memory>
#include <mutex>
#include <vector>

#include "vk2/Buffer.hpp"
namespace gfx {

struct StagingBufferPool {
  static StagingBufferPool& get();
  static void destroy();
  static void init();
  Buffer* acquire(u64 size);
  void free(Buffer* buffer);

 private:
  std::mutex mtx_;
  std::vector<std::unique_ptr<Buffer>> free_buffers_;
  std::vector<std::unique_ptr<Buffer>> allocated_buffers_;
};
}  // namespace gfx
