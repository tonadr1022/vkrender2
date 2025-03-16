#include "BindlessResourceAllocator.hpp"

#include <volk.h>

#include <tracy/Tracy.hpp>

#include "vk2/Resource.hpp"
#include "vk2/VkCommon.hpp"
namespace vk2 {

BindlessResourceAllocator::BindlessResourceAllocator(VkDevice device, VmaAllocator allocator)
    : device_(device), allocator_(allocator) {
  ZoneScoped;
  VkDescriptorPoolSize sizes[] = {
      VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, max_resource_descriptors},
      VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, max_resource_descriptors},
      VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, max_resource_descriptors},
      VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_SAMPLER, max_sampler_descriptors},
  };
  VkDescriptorPoolCreateInfo info{.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                                  .flags = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT,
                                  // TODO: fine tune this? just one?
                                  .maxSets = 10,
                                  .poolSizeCount = COUNTOF(sizes),
                                  .pPoolSizes = sizes};
  VK_CHECK(vkCreateDescriptorPool(device_, &info, nullptr, &main_pool_));
  VkDescriptorSetAllocateInfo set_layout_info{
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .descriptorPool = main_pool_,
      .descriptorSetCount = 1,
      .pSetLayouts = &main_set_layout_};
  VK_CHECK(vkAllocateDescriptorSets(device_, &set_layout_info, &main_set_));

  VkDescriptorSetLayoutBinding bindings[] = {
      VkDescriptorSetLayoutBinding{
          .binding = 0,
          .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
          .descriptorCount = max_resource_descriptors,
          .stageFlags = VK_SHADER_STAGE_ALL,
      },
      VkDescriptorSetLayoutBinding{
          .binding = 1,
          .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
          .descriptorCount = max_resource_descriptors,
          .stageFlags = VK_SHADER_STAGE_ALL,
      },
      VkDescriptorSetLayoutBinding{
          .binding = 2,
          .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
          .descriptorCount = max_resource_descriptors,
          .stageFlags = VK_SHADER_STAGE_ALL,
      },
      VkDescriptorSetLayoutBinding{
          .binding = 3,
          .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER,
          .descriptorCount = max_sampler_descriptors,
          .stageFlags = VK_SHADER_STAGE_ALL,
      },
  };
  VkDescriptorSetLayoutCreateInfo set_info{
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT,
      .bindingCount = COUNTOF(bindings),
      .pBindings = bindings};
  VK_CHECK(vkCreateDescriptorSetLayout(device_, &set_info, nullptr, &main_set_layout_));
}

namespace {
BindlessResourceAllocator* instance = nullptr;
}

BindlessResourceAllocator& BindlessResourceAllocator::get() {
  assert(instance);
  return *instance;
}

void BindlessResourceAllocator::init(VkDevice device, VmaAllocator allocator) {
  assert(!instance);
  instance = new BindlessResourceAllocator{device, allocator};
}

void BindlessResourceAllocator::shutdown() {
  assert(instance);
  delete instance;
}

VkImageView BindlessResourceAllocator::create_image_view(const ImageViewCreateInfo& info) const {
  VkImageView view;
  VkImageViewCreateInfo i{.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                          .image = info.image,
                          .viewType = VK_IMAGE_VIEW_TYPE_2D,
                          .format = info.format,
                          .subresourceRange = info.subresource_range};
  VK_CHECK(vkCreateImageView(device_, &i, nullptr, &view));
  return view;
}

Texture BindlessResourceAllocator::alloc_img(const VkImageCreateInfo& create_info,
                                             VkMemoryPropertyFlags req_flags, bool mapped) {
  VmaAllocationCreateFlags alloc_flags{};
  if (mapped) {
    alloc_flags |= VMA_ALLOCATION_CREATE_MAPPED_BIT;
  }
  VmaAllocationCreateInfo alloc_create_info{
      .flags = alloc_flags,
      .usage = VMA_MEMORY_USAGE_AUTO,
      .requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | req_flags,
  };

  Texture new_img;
  new_img.extent_ = create_info.extent;
  new_img.format_ = create_info.format;
  VK_CHECK(vmaCreateImage(allocator_, &create_info, &alloc_create_info, &new_img.image_,
                          &new_img.allocation_, nullptr));

  // TODO: also check for color only
  if (create_info.usage & VK_IMAGE_USAGE_STORAGE_BIT) {
    new_img.storage_image_resource_info_ = BindlessResourceInfo{
        .type = ResourceType::STORAGE_IMAGE, .handle = storage_image_allocator_.alloc()};
  }

  new_img.sampled_image_resource_info_ = BindlessResourceInfo{
      .type = ResourceType::SAMPLED_IMAGE,
      .handle = sampled_image_allocator_.alloc(),
  };

  return new_img;
}

Texture BindlessResourceAllocator::alloc_img_with_view(const VkImageCreateInfo& create_info,
                                                       const VkImageSubresourceRange& range,
                                                       VkImageViewType type,
                                                       VkMemoryPropertyFlags req_flags,
                                                       bool mapped) {
  Texture new_img = alloc_img(create_info, req_flags, mapped);
  auto view_info = VkImageViewCreateInfo{.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                                         .image = new_img.image_,
                                         .viewType = type,
                                         .format = new_img.format_,
                                         .subresourceRange = range};
  VK_CHECK(vkCreateImageView(device_, &view_info, nullptr, &new_img.view_));
  return new_img;
}

void IndexAllocator::free(u32 idx) { free_list_.push_back(idx); }

u32 IndexAllocator::alloc() {
  assert(free_list_.size());
  if (free_list_.empty()) {
    return UINT32_MAX;
  }
  auto ret = free_list_.front();
  free_list_.pop_back();
  return ret;
}

IndexAllocator::IndexAllocator(u32 size) {
  for (u32 i = 0; i < size; i++) {
    free_list_.push_back(i);
  }
}

void BindlessResourceAllocator::destroy_image(Texture& image) {
  if (image.image_) {
    assert(image.allocation_);
    if (image.allocation_) {
      vmaDestroyImage(allocator_, image.image_, image.allocation_);
    }
    if (image.sampled_image_resource_info_) {
      sampled_image_allocator_.free(image.sampled_image_resource_info_->handle);
    }
    if (image.storage_image_resource_info_) {
      storage_image_allocator_.free(image.storage_image_resource_info_->handle);
    }
  }
  if (image.view_) {
    vkDestroyImageView(device_, image.view_, nullptr);
  }
}
BindlessResourceAllocator::~BindlessResourceAllocator() {
  vkDestroyDescriptorPool(device_, main_pool_, nullptr);
  vkDestroyDescriptorSetLayout(device_, main_set_layout_, nullptr);
}
}  // namespace vk2
