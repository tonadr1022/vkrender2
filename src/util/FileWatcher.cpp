#include "FileWatcher.hpp"

#include <filesystem>
#include <fstream>
#include <utility>

#include "ThreadPool.hpp"

namespace util {

void FileWatcher::start() {
  if (!enabled_) return;
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
  for (const auto& file : std::filesystem::recursive_directory_iterator(base_path_)) {
    if (file.is_directory()) continue;
    auto last_write_time = file.last_write_time();
    auto it = modified_time_stamps_.find(file);
    if (it == modified_time_stamps_.end()) {
      modified_time_stamps_.emplace(file, last_write_time);
      dirty_files_.emplace_back(file);
    } else if (it->second < last_write_time) {
      it->second = last_write_time;
      dirty_files_.emplace_back(file);
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
  std::ofstream ofs(base_path_ / ".cache" / "filewatcher_cache.txt");
  if (ofs.is_open()) {
    for (const auto& [filename, time] : modified_time_stamps_) {
      ofs << filename.string() << ' ' << static_cast<size_t>(time.time_since_epoch().count())
          << '\n';
    }
  }
}

FileWatcher::~FileWatcher() { shutdown(); }

FileWatcher::FileWatcher(std::filesystem::path base_path, OnDirtyFunc func,
                         std::chrono::milliseconds sleep_time, bool enabled)
    : on_dirty_func_(std::move(func)),
      base_path_(std::move(base_path)),
      sleep_time_(sleep_time),
      enabled_(enabled) {
  std::ifstream ifs(base_path_ / ".cache" / "filewatcher_cache.txt");
  if (ifs.is_open()) {
    std::string filename;
    uint64_t write_time;
    while (ifs >> filename >> write_time) {
      modified_time_stamps_.emplace(
          filename, std::filesystem::file_time_type(std::chrono::milliseconds(write_time)));
    }
  }
}

void FileWatcher::add_timestamps(
    std::span<std::pair<std::string, std::filesystem::file_time_type>> timestamps) {
  for (const auto& [filepath, modified_time] : timestamps) {
    modified_time_stamps_.emplace(filepath, modified_time);
  }
}

}  // namespace util
