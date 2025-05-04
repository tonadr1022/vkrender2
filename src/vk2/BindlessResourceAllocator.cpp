#include "BindlessResourceAllocator.hpp"

#include <volk.h>
#include <vulkan/vulkan_core.h>

#include <tracy/Tracy.hpp>

#include "vk2/Buffer.hpp"
#include "vk2/Resource.hpp"
#include "vk2/Texture.hpp"
#include "vk2/VkCommon.hpp"
namespace gfx::vk2 {

ResourceAllocator::ResourceAllocator(VkDevice device, VmaAllocator allocator)
    : device_(device), allocator_(allocator) {
  ZoneScoped;
  {
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
            .descriptorCount = max_sampler_descriptors,
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
    };

    std::array<VkDescriptorBindingFlags, COUNTOF(bindings)> flags;
    for (auto& f : flags) {
      f = VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT | VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT |
          VK_DESCRIPTOR_BINDING_UPDATE_UNUSED_WHILE_PENDING_BIT_EXT;
    }

    VkDescriptorSetLayoutBindingFlagsCreateInfo binding_flags{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO,
        .bindingCount = flags.size(),
        .pBindingFlags = flags.data()};
    VkDescriptorSetLayoutCreateInfo set_info{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .pNext = &binding_flags,
        .flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT,
        .bindingCount = COUNTOF(bindings),
        .pBindings = bindings};
    VK_CHECK(vkCreateDescriptorSetLayout(device_, &set_info, nullptr, &main_set_layout_));

    VkDescriptorPoolSize sizes[] = {
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, max_resource_descriptors},
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, max_resource_descriptors},
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_SAMPLER, max_sampler_descriptors},
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, max_resource_descriptors},
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, max_sampler_descriptors},
    };

    VkDescriptorPoolCreateInfo info{.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                                    .flags = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT,
                                    // TODO: fine tune this? just one?
                                    .maxSets = 10,
                                    .poolSizeCount = COUNTOF(sizes),
                                    .pPoolSizes = sizes};
    VK_CHECK(vkCreateDescriptorPool(device_, &info, nullptr, &main_pool_));
    assert(main_pool_);
    assert(main_set_layout_);
    VkDescriptorSetAllocateInfo set_layout_info{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = main_pool_,
        .descriptorSetCount = 1,
        .pSetLayouts = &main_set_layout_};
    VK_CHECK(vkAllocateDescriptorSets(device_, &set_layout_info, &main_set_));
    assert(main_set_);
  }
  {
    VkDescriptorSetLayoutBinding bindings[] = {
        VkDescriptorSetLayoutBinding{
            .binding = bindless_sampler_binding,
            .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER,
            .descriptorCount = max_sampler_descriptors,
            .stageFlags = VK_SHADER_STAGE_ALL,
        },
    };

    std::array<VkDescriptorBindingFlags, COUNTOF(bindings)> flags;
    for (auto& f : flags) {
      f = VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT | VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT |
          VK_DESCRIPTOR_BINDING_UPDATE_UNUSED_WHILE_PENDING_BIT_EXT;
    }

    VkDescriptorSetLayoutBindingFlagsCreateInfo binding_flags{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO,
        .bindingCount = flags.size(),
        .pBindingFlags = flags.data()};
    VkDescriptorSetLayoutCreateInfo set_info{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .pNext = &binding_flags,
        .flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT,
        .bindingCount = COUNTOF(bindings),
        .pBindings = bindings};
    VK_CHECK(vkCreateDescriptorSetLayout(device_, &set_info, nullptr, &main_set2_layout_));
    assert(main_pool_);
    assert(main_set2_layout_);
    VkDescriptorSetAllocateInfo set_layout_info{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = main_pool_,
        .descriptorSetCount = 1,
        .pSetLayouts = &main_set2_layout_};
    VK_CHECK(vkAllocateDescriptorSets(device_, &set_layout_info, &main_set2_));
    assert(main_set2_);
  }
}

namespace {
ResourceAllocator* instance = nullptr;
}

ResourceAllocator& ResourceAllocator::get() {
  assert(instance);
  return *instance;
}

void ResourceAllocator::init(VkDevice device, VmaAllocator allocator) {
  assert(!instance);
  instance = new ResourceAllocator{device, allocator};
}

void ResourceAllocator::shutdown() {
  assert(instance);
  delete instance;
}

ResourceAllocator::~ResourceAllocator() {
  set_frame_num(UINT32_MAX, 0);
  flush_deletions();
  vkDestroyDescriptorPool(device_, main_pool_, nullptr);
  vkDestroyDescriptorSetLayout(device_, main_set_layout_, nullptr);
  vkDestroyDescriptorSetLayout(device_, main_set2_layout_, nullptr);
}

BindlessResourceInfo ResourceAllocator::allocate_sampled_img_descriptor(VkImageView view,
                                                                        VkImageLayout layout) {
  u32 handle = sampled_image_allocator_.alloc();
  VkDescriptorImageInfo img{.sampler = nullptr, .imageView = view, .imageLayout = layout};
  allocate_bindless_resource(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &img, nullptr, handle,
                             bindless_sampled_image_binding);
  return {ResourceType::SampledImage, handle};
}

BindlessResourceInfo ResourceAllocator::allocate_storage_img_descriptor(VkImageView view,
                                                                        VkImageLayout layout) {
  u32 handle = storage_image_allocator_.alloc();
  VkDescriptorImageInfo img{.sampler = nullptr, .imageView = view, .imageLayout = layout};
  allocate_bindless_resource(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &img, nullptr, handle,
                             resource_to_binding(ResourceType::StorageImage));
  return {ResourceType::StorageImage, handle};
}

BindlessResourceInfo ResourceAllocator::allocate_sampler_descriptor(VkSampler sampler) {
  u32 handle = sampler_allocator_.alloc();
  VkDescriptorImageInfo info{.sampler = sampler};
  VkWriteDescriptorSet write{.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                             .dstSet = main_set2_,
                             .dstBinding = bindless_sampler_binding,
                             .dstArrayElement = handle,
                             .descriptorCount = 1,
                             .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER,
                             .pImageInfo = &info,
                             .pBufferInfo = nullptr};
  vkUpdateDescriptorSets(device_, 1, &write, 0, nullptr);
  return {ResourceType::Sampler, handle};
}

void ResourceAllocator::allocate_bindless_resource(VkDescriptorType descriptor_type,
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

u32 ResourceAllocator::resource_to_binding(ResourceType type) {
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
void ResourceAllocator::set_frame_num(u32 frame_num, u32 buffer_count) {
  frame_num_ = frame_num;
  buffer_count_ = buffer_count;
}

void ResourceAllocator::delete_texture(const TextureDeleteInfo& img) {
  texture_delete_q_.emplace_back(img, frame_num_);
}

void ResourceAllocator::flush_deletions() {
  std::erase_if(texture_delete_q_, [this](const DeleteQEntry<TextureDeleteInfo>& entry) {
    if (entry.frame + buffer_count_ < frame_num_) {
      vmaDestroyImage(allocator_, entry.data.img, entry.data.allocation);
      return true;
    }
    return false;
  });

  std::erase_if(swapchain_delete_q_, [this](const DeleteQEntry<VkSwapchainKHR>& entry) {
    if (entry.frame + buffer_count_ < frame_num_) {
      vkDestroySwapchainKHR(device_, entry.data, nullptr);
      return true;
    }
    return false;
  });

  std::erase_if(texture_view_delete_q_, [this](const DeleteQEntry<TextureViewDeleteInfo>& entry) {
    if (entry.frame + buffer_count_ < frame_num_) {
      if (entry.data.sampled_image_resource_info.has_value()) {
        sampled_image_allocator_.free(entry.data.sampled_image_resource_info->handle);
      }
      if (entry.data.storage_image_resource_info.has_value()) {
        storage_image_allocator_.free(entry.data.storage_image_resource_info->handle);
      }
      vkDestroyImageView(device_, entry.data.view, nullptr);
      return true;
    }
    return false;
  });

  std::erase_if(storage_buffer_delete_q_, [this](const DeleteQEntry<BufferDeleteInfo>& entry) {
    if (entry.frame < frame_num_) {
      assert(entry.data.buffer);
      if (entry.data.resource_info.has_value()) {
        storage_buffer_allocator_.free(entry.data.resource_info->handle);
      }
      vmaDestroyBuffer(allocator_, entry.data.buffer, entry.data.allocation);
      return true;
    }
    return false;
  });
}

void ResourceAllocator::delete_texture_view(const TextureViewDeleteInfo& info) {
  texture_view_delete_q_.emplace_back(info, frame_num_);
}

void ResourceAllocator::delete_buffer(const BufferDeleteInfo& info) {
  storage_buffer_delete_q_.emplace_back(info, frame_num_ + 10);
}

BindlessResourceInfo ResourceAllocator::allocate_storage_buffer_descriptor(VkBuffer buffer) {
  u32 handle = storage_buffer_allocator_.alloc();
  VkDescriptorBufferInfo buf{.buffer = buffer, .offset = 0, .range = VK_WHOLE_SIZE};
  allocate_bindless_resource(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &buf, handle,
                             resource_to_binding(ResourceType::StorageBuffer));
  return {ResourceType::StorageBuffer, handle};
}

void ResourceAllocator::enqueue_delete_swapchain(VkSwapchainKHR swapchain) {
  swapchain_delete_q_.emplace_back(swapchain, frame_num_);
}
}  // namespace gfx::vk2
