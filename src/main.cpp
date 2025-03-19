
#include <print>

#include "App.hpp"
#include "vk_mem_alloc.h"

int main() {
  App app{{.name = "VkRender2"}};
  app.run();
}
