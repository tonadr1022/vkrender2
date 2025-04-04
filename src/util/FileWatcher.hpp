#pragma once

#include <atomic>
#include <chrono>
#include <filesystem>
#include <functional>
#include <future>

namespace util {
class FileWatcher {
 public:
  using OnDirtyFunc = std::function<void(std::span<std::filesystem::path>)>;
  void update();
  explicit FileWatcher(std::filesystem::path base_path, OnDirtyFunc func,
                       std::chrono::milliseconds sleep_time = std::chrono::milliseconds(500),
                       std::filesystem::path cache_path = "");

  void start();
  void shutdown();
  ~FileWatcher();

 private:
  void update_loop();
  OnDirtyFunc on_dirty_func_;
  std::filesystem::path base_path_;
  std::filesystem::path cache_path_;
  std::atomic<bool> running_{true};
  std::chrono::milliseconds sleep_time_{500};
  std::unordered_map<std::filesystem::path, std::filesystem::file_time_type> modified_time_stamps_;
  std::vector<std::filesystem::path> dirty_files_;
  std::future<void> update_task_;
  bool cache_file_dirty_{false};
};
}  // namespace util
