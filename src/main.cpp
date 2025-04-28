
#include <cstring>
#include <print>

#include "App.hpp"

#define CMP(arg, cmp) strcmp(arg, cmp) == 0

using vr = std::expected<void, const char*>;
int main(int argc, char* argv[]) {
  u32 w{800}, h{800};
  bool vsync{true}, maximize{false};
  for (int i = 1; i < argc; i++) {
    char* arg = argv[i];
    if (CMP(arg, "-w") && i < argc - 1) {
      w = std::stoi(argv[i + 1]);
    } else if (CMP(arg, "-h") && i < argc - 1) {
      h = std::stoi(argv[i + 1]);
    } else if (CMP(arg, "--no-vsync")) {
      vsync = false;
    } else if (CMP(arg, "--maximize") || CMP(arg, "-m")) {
      maximize = true;
    }
  }
  w = std::min(std::max(w, 100u), 5000u);
  h = std::min(std::max(h, 100u), 5000u);
  App app{{.name = "VkRender2", .width = w, .height = h, .maximize = maximize, .vsync = vsync}};
  app.run();
}
