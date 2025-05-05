#pragma once

#include <atomic>
#include <chrono>
#include <filesystem>
#include <functional>
#include <future>
#include <span>

namespace util {
class FileWatcher {
 public:
  using OnDirtyFunc = std::function<void(std::span<std::filesystem::path>)>;
  void update();
  explicit FileWatcher(std::filesystem::path base_path, OnDirtyFunc func,
                       std::vector<const char*>&& watch_extensions,
                       std::chrono::milliseconds sleep_time = std::chrono::milliseconds(500));

  void start();
  void shutdown();
  ~FileWatcher();
  void add_timestamps(
      std::span<std::pair<std::string, std::filesystem::file_time_type>> timestamps);

  const std::unordered_map<std::filesystem::path, std::filesystem::file_time_type>&
  get_modified_timestamps() const {
    return modified_time_stamps_;
  }

 private:
  void update_loop();
  OnDirtyFunc on_dirty_func_;
  std::filesystem::path base_path_;
  std::vector<const char*> watch_extensions_;
  std::atomic<bool> running_;
  std::chrono::milliseconds sleep_time_{500};
  std::unordered_map<std::filesystem::path, std::filesystem::file_time_type> modified_time_stamps_;
  std::vector<std::filesystem::path> dirty_files_;
  std::future<void> update_task_;
};
}  // namespace util
