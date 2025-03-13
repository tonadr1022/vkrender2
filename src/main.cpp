
#include <print>

#include "VkRender2.hpp"
#include "vk_mem_alloc.h"

int main() {
  VkRender2 app{{.name = "VkRender2"}};
  app.run();
}
