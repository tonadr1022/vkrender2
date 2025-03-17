#include "BindlessResourceAllocator.hpp"

#include <volk.h>
#include <vulkan/vulkan_core.h>

#include <tracy/Tracy.hpp>

#include "vk2/Resource.hpp"
#include "vk2/Texture.hpp"
#include "vk2/VkCommon.hpp"
namespace vk2 {

TextureDeleteFunc img_delete_func = [](TextureDeleteInfo info) {
  BindlessResourceAllocator::get().delete_texture(info);
};

TextureViewDeleteFunc texture_view_delete_func = [](TextureViewDeleteInfo info) {
  BindlessResourceAllocator::get().telete_texture_view(info);
};

BindlessResourceAllocator::BindlessResourceAllocator(VkDevice device, VmaAllocator allocator)
    : device_(device), allocator_(allocator) {
  ZoneScoped;
  VkDescriptorSetLayoutBinding bindings[] = {
      VkDescriptorSetLayoutBinding{
          .binding = bindless_storage_image_binding,
          .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
          .descriptorCount = max_resource_descriptors,
          .stageFlags = VK_SHADER_STAGE_ALL,
      },
      VkDescriptorSetLayoutBinding{
          .binding = bindless_combined_image_sampler_binding,
          .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
          .descriptorCount = max_resource_descriptors,
          .stageFlags = VK_SHADER_STAGE_ALL,
      },
      VkDescriptorSetLayoutBinding{
          .binding = bindless_storage_buffer_binding,
          .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
          .descriptorCount = max_resource_descriptors,
          .stageFlags = VK_SHADER_STAGE_ALL,
      },
      VkDescriptorSetLayoutBinding{
          .binding = bindless_sampled_image_binding,
          .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
          .descriptorCount = max_resource_descriptors,
          .stageFlags = VK_SHADER_STAGE_ALL,
      },
      VkDescriptorSetLayoutBinding{
          .binding = bindless_sampler_binding,
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

BindlessResourceAllocator::~BindlessResourceAllocator() {
  vkDestroyDescriptorPool(device_, main_pool_, nullptr);
  vkDestroyDescriptorSetLayout(device_, main_set_layout_, nullptr);
}

BindlessResourceInfo BindlessResourceAllocator::allocate_sampled_img_descriptor(
    VkImageView view, VkImageLayout layout) {
  u32 handle = sampled_image_allocator_.alloc();
  VkDescriptorImageInfo img{.sampler = nullptr, .imageView = view, .imageLayout = layout};
  allocate_bindless_resource(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &img, nullptr, handle,
                             bindless_sampled_image_binding);
  return {ResourceType::SampledImage, handle};
}

BindlessResourceInfo BindlessResourceAllocator::allocate_storage_img_descriptor(
    VkImageView view, VkImageLayout layout) {
  u32 handle = storage_image_allocator_.alloc();
  VkDescriptorImageInfo img{.sampler = nullptr, .imageView = view, .imageLayout = layout};
  allocate_bindless_resource(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &img, nullptr, handle,
                             resource_to_binding(ResourceType::StorageImage));
  return {ResourceType::StorageImage, handle};
}

void BindlessResourceAllocator::allocate_bindless_resource(VkDescriptorType descriptor_type,
                                                           VkDescriptorImageInfo* img,
                                                           VkDescriptorBufferInfo* buffer, u32 idx,
                                                           u32 binding) {
  VkWriteDescriptorSet write{.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                             .dstSet = main_set_,
                             .dstBinding = binding,
                             .dstArrayElement = idx,
                             .descriptorCount = 1,
                             .descriptorType = descriptor_type,
                             .pImageInfo = img,
                             .pBufferInfo = buffer};
  vkUpdateDescriptorSets(device_, 1, &write, 0, nullptr);
}

u32 BindlessResourceAllocator::resource_to_binding(ResourceType type) {
  switch (type) {
    case ResourceType::StorageImage:
      return bindless_storage_image_binding;
    case ResourceType::StorageBuffer:
      return bindless_storage_buffer_binding;
    case ResourceType::SampledImage:
      return bindless_sampled_image_binding;
    case ResourceType::Sampler:
      return bindless_sampler_binding;
    case ResourceType::CombinedImageSampler:
      return bindless_combined_image_sampler_binding;
  }
}
void BindlessResourceAllocator::set_frame_num(u32 frame_num) { frame_num_ = frame_num; }

void BindlessResourceAllocator::delete_texture(const TextureDeleteInfo& img) {
  texture_delete_q_.emplace_back(img, frame_num_);
}

void BindlessResourceAllocator::flush_deletions() {
  std::erase_if(texture_delete_q_, [this](const DeleteQEntry<TextureDeleteInfo>& entry) {
    if (entry.frame < frame_num_) {
      vmaDestroyImage(allocator_, entry.data.img, entry.data.allocation);
      return true;
    }
    return false;
  });

  std::erase_if(texture_view_delete_q_, [this](const DeleteQEntry<TextureViewDeleteInfo>& entry) {
    if (entry.frame < frame_num_) {
      vkDestroyImageView(device_, entry.data.view, nullptr);
      if (entry.data.sampled_image_resource_info.has_value()) {
        sampled_image_allocator_.free(entry.data.sampled_image_resource_info->handle);
      }
      if (entry.data.storage_image_resource_info.has_value()) {
        storage_image_allocator_.free(entry.data.storage_image_resource_info->handle);
      }
      return true;
    }
    return false;
  });
}

void BindlessResourceAllocator::telete_texture_view(const TextureViewDeleteInfo& info) {
  texture_view_delete_q_.emplace_back(info, frame_num_);
}

}  // namespace vk2
