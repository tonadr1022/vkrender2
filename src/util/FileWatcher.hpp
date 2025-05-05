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
  FileWatcher() = default;
  explicit FileWatcher(std::filesystem::path base_path, OnDirtyFunc func,
                       std::chrono::milliseconds sleep_time = std::chrono::milliseconds(500),
                       bool enabled = true);

  std::optional<std::filesystem::file_time_type> cached_write_time(
      const std::filesystem::path& path) {
    auto it = modified_time_stamps_.find(path);
    if (it != modified_time_stamps_.end()) {
      return it->second;
    }
    return {};
  }
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
  std::atomic<bool> running_;
  std::chrono::milliseconds sleep_time_{500};
  std::unordered_map<std::filesystem::path, std::filesystem::file_time_type> modified_time_stamps_;
  std::vector<std::filesystem::path> dirty_files_;
  std::future<void> update_task_;
  bool enabled_{};
};
}  // namespace util
