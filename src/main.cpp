
#include <print>

#include "App.hpp"
#include "vk_mem_alloc.h"

int main(int argc, char* argv[]) {
  u32 w{800}, h{800};
  if (argc >= 3) {
    w = std::stoi(argv[1]);
    h = std::stoi(argv[2]);
  }
  App app{{.name = "VkRender2", .width = w, .height = h}};
  app.run();
}
