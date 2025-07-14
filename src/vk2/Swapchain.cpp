#include "Swapchain.hpp"

#include <volk.h>
#include <vulkan/vk_enum_string_helper.h>
#include <vulkan/vulkan_core.h>

#include <tracy/Tracy.hpp>

#include "vk2/Device.hpp"

namespace gfx::vk2 {

void Swapchain::destroy(VkDevice device) {
  device_imgs.clear();
  for (auto& semaphore : acquire_semaphores) {
    if (semaphore) {
      vkDestroySemaphore(device, semaphore, nullptr);
      semaphore = nullptr;
    }
  }
  for (auto& semaphore : release_semaphores) {
    if (semaphore) {
      vkDestroySemaphore(device, semaphore, nullptr);
      semaphore = nullptr;
    }
  }
  vkDestroySwapchainKHR(device, swapchain, nullptr);
  swapchain = nullptr;
}

}  // namespace gfx::vk2
