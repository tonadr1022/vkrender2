#pragma once

#include <chrono>

#include "Logger.hpp"

class Timer {
 public:
  Timer() { start(); }
  ~Timer() = default;

  void start() { start_time_ = std::chrono::high_resolution_clock::now(); };

  double elapsed_seconds() { return elapsed_micro() * 0.000001; }

  double elapsed_ms() { return elapsed_micro() * 0.001; }

  uint64_t elapsed_micro() {
    auto end_time = std::chrono::high_resolution_clock::now();
    start_ = time_point_cast<std::chrono::microseconds>(start_time_).time_since_epoch().count();
    end_ = time_point_cast<std::chrono::microseconds>(end_time).time_since_epoch().count();
    return end_ - start_;
  }

  void print() { LINFO("ElapsedMS: {}", elapsed_ms()); }
  void print_micro() { LINFO("ElapsedMicro: {}", elapsed_micro()); }
  void print_ms() { LINFO("ElapsedMS: {}", elapsed_ms()); }
  void reset(std::string_view msg) {
    print(msg);
    reset();
  }
  void print(std::string_view msg) { LINFO("{}: {}", msg, elapsed_ms()); }
  void reset() { start_time_ = std::chrono::high_resolution_clock::now(); }

 private:
  uint64_t start_, end_;
  std::chrono::time_point<std::chrono::high_resolution_clock> start_time_;
};

class PrintTimerMS : public Timer {
 public:
  ~PrintTimerMS() { print_ms(); }
};
class PrintTimerMicro : public Timer {
 public:
  ~PrintTimerMicro() { print_micro(); }
};
