#include "FileWatcher.hpp"

#include <algorithm>
#include <fstream>
#include <ranges>
#include <stdexcept>
#include <utility>

#include "ThreadPool.hpp"

namespace util {

void FileWatcher::start() { update_loop(); }

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
  running_ = false;
  if (update_task_.valid()) {
    update_task_.get();
  }
  // TODO: race condition lol
  if (!cache_path_.empty()) {
    auto p = cache_path_;
    std::vector<std::filesystem::path> paths;
    while (p.has_parent_path() && p != p.root_path()) {
      paths.emplace_back(p.parent_path());
      p = p.parent_path();
    }
    std::ranges::reverse(paths);
    for (const auto& dir : paths) {
      if (!std::filesystem::exists(dir)) {
        std::filesystem::create_directory(dir);
      }
    }
    std::ofstream ofs(cache_path_);
    if (ofs.is_open()) {
      for (const auto& [file, time] : modified_time_stamps_) {
        ofs << file.string() << ' ' << static_cast<size_t>(time.time_since_epoch().count()) << '\n';
      }
    }
  }
}

FileWatcher::~FileWatcher() { shutdown(); }

FileWatcher::FileWatcher(std::filesystem::path base_path, OnDirtyFunc func,
                         std::chrono::milliseconds sleep_time, std::filesystem::path cache_path)
    : on_dirty_func_(std::move(func)),
      base_path_(std::move(base_path)),
      cache_path_(std::move(cache_path)),
      sleep_time_(sleep_time) {
  if (!cache_path_.empty() && std::filesystem::exists(cache_path_)) {
    std::ifstream file(cache_path_);
    if (file.is_open()) {
      try {
        std::filesystem::path filename;
        std::uint64_t timestamp;
        while (file >> filename >> timestamp) {
          modified_time_stamps_.emplace(
              filename, std::filesystem::file_time_type(std::chrono::nanoseconds(timestamp)));
        }
      } catch (const std::runtime_error&) {
        return;
      }
    }
  }
}
}  // namespace util
