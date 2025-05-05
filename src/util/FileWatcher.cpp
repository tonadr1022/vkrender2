#include "FileWatcher.hpp"

#include <utility>

#include "ThreadPool.hpp"

namespace util {

void FileWatcher::start() {
  running_ = true;
  update_loop();
}

void FileWatcher::update_loop() {
  using namespace std::chrono_literals;
  update();
  update_task_ = threads::pool.submit_task([this]() {
    if (!running_) {
      return;
    }
    std::this_thread::sleep_for(sleep_time_);
    if (!running_) {
      return;
    }
    update_loop();
  });
}

void FileWatcher::update() {
  for (const auto& dir : std::filesystem::recursive_directory_iterator(base_path_)) {
    if (dir.is_directory()) continue;
    auto last_write_time = dir.last_write_time();
    auto it = modified_time_stamps_.find(dir);
    if (it == modified_time_stamps_.end()) {
      modified_time_stamps_.emplace(dir, last_write_time);
      dirty_files_.emplace_back(dir);
    } else if (it->second < last_write_time) {
      it->second = last_write_time;
      dirty_files_.emplace_back(dir);
    }
  }

  if (dirty_files_.size() && on_dirty_func_) {
    on_dirty_func_(dirty_files_);
  }
  dirty_files_.clear();
}

void FileWatcher::shutdown() {
  if (running_) {
    running_ = false;
    if (update_task_.valid()) {
      update_task_.get();
    }
  }
}

FileWatcher::~FileWatcher() { shutdown(); }

FileWatcher::FileWatcher(std::filesystem::path base_path, OnDirtyFunc func,
                         std::vector<const char*>&& watch_extensions,
                         std::chrono::milliseconds sleep_time)
    : on_dirty_func_(std::move(func)),
      base_path_(std::move(base_path)),
      watch_extensions_(std::move(watch_extensions)),
      sleep_time_(sleep_time) {}

void FileWatcher::add_timestamps(
    std::span<std::pair<std::string, std::filesystem::file_time_type>> timestamps) {
  for (const auto& [filepath, modified_time] : timestamps) {
    modified_time_stamps_.emplace(filepath, modified_time);
  }
}

}  // namespace util
