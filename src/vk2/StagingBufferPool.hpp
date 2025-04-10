#pragma once

#include <memory>
#include <mutex>
#include <vector>

#include "vk2/Buffer.hpp"
namespace gfx::vk2 {

struct StagingBufferPool {
  static StagingBufferPool& get();
  static void destroy();
  static void init();
  vk2::Buffer* acquire(u64 size);
  void free(vk2::Buffer* buffer);

 private:
  std::mutex mtx_;
  std::vector<std::unique_ptr<vk2::Buffer>> free_buffers_;
  std::vector<std::unique_ptr<vk2::Buffer>> allocated_buffers_;
};
}  // namespace gfx::vk2
