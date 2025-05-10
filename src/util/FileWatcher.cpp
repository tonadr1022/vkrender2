#include "FileWatcher.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <tracy/Tracy.hpp>
#include <utility>

namespace util {

void FileWatcher::start() {
  if (!enabled_) return;
  running_ = true;
  update_loop();
}

void FileWatcher::update_loop() {
  using namespace std::chrono_literals;
  update();
  update_thread_ = std::thread([this]() {
    while (running_) {
      {
        std::unique_lock lock(mtx_);
        cv_.wait_for(lock, std::chrono::milliseconds(sleep_time_), [this]() { return !running_; });
      }
      update();
    }
  });
}

void FileWatcher::update() {
  ZoneScoped;
  std::scoped_lock lock(mtx_);
  for (const auto& file : std::filesystem::recursive_directory_iterator(base_path_)) {
    if (file.is_directory()) continue;
    auto last_write_time_new = file.last_write_time();
    auto it = modified_time_stamps_.find(file);
    if (it == modified_time_stamps_.end()) {
      modified_time_stamps_.emplace(file, last_write_time_new);
      dirty_files_.emplace_back(file);
    } else if (it->second < last_write_time_new) {
      it->second = last_write_time_new;
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
    cv_.notify_all();
  }
  if (update_thread_.joinable()) {
    update_thread_.join();
  }
  std::ofstream ofs(base_path_ / ".cache" / "filewatcher_cache.txt");
  if (ofs.is_open()) {
    std::scoped_lock lock(mtx_);
    for (const auto& [filename, time] : modified_time_stamps_) {
      if (std::filesystem::exists(filename) &&
          std::ranges::contains(file_extensions_, filename.extension().string())) {
        ofs << filename.string() << ' ' << static_cast<size_t>(time.time_since_epoch().count())
            << '\n';
      }
    }
  }
}

FileWatcher::~FileWatcher() { shutdown(); }

FileWatcher::FileWatcher(std::filesystem::path base_path, std::vector<std::string> file_extensions,
                         OnDirtyFunc func, std::chrono::milliseconds sleep_time, bool enabled)
    : on_dirty_func_(std::move(func)),
      base_path_(std::move(base_path)),
      file_extensions_(std::move(file_extensions)),
      sleep_time_(sleep_time),
      enabled_(enabled) {
  std::ifstream ifs(base_path_ / ".cache" / "filewatcher_cache.txt");
  if (ifs.is_open()) {
    std::string filename;
    uint64_t write_time;
    while (ifs >> filename >> write_time) {
      auto path = std::filesystem::path(filename);
      if (std::filesystem::exists(path) &&
          std::ranges::contains(file_extensions_, path.extension().string()) &&
          !modified_time_stamps_.contains(filename)) {
        modified_time_stamps_.emplace(
            filename, std::filesystem::file_time_type(std::chrono::nanoseconds(write_time)));
      }
    }
  }
}

}  // namespace util
