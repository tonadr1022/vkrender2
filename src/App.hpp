#pragma once

#include <string_view>

#include "Common.hpp"

class App {
 public:
  struct InitInfo {
    std::string_view name;
    u32 width;
    u32 height;
    bool maximize{false};
    bool decorate{true};
    bool vsync{true};
  };

  explicit App(const InitInfo& info);
  App(const App&) = delete;
  // TODO: others
  virtual ~App();
  void Run();
};
