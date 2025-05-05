#include "CommandEncoder.hpp"

#include <volk.h>
#include <vulkan/vulkan_core.h>

#include "vk2/VkTypes.hpp"

namespace gfx {

void CmdEncoder::dispatch(u32 work_groups_x, u32 work_groups_y, u32 work_groups_z) {
  vkCmdDispatch(cmd_, work_groups_x, work_groups_y, work_groups_z);
}

void CmdEncoder::bind_compute_pipeline(VkPipeline pipeline) {
  vkCmdBindPipeline(cmd_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
}
void CmdEncoder::bind_descriptor_set(VkPipelineBindPoint bind_point, VkPipelineLayout layout,
                                     VkDescriptorSet* set, u32 idx) {
  vkCmdBindDescriptorSets(cmd_, bind_point, layout, idx, 1, set, 0, nullptr);
}

void CmdEncoder::push_constants(VkPipelineLayout layout, u32 size, void* data) {
  vkCmdPushConstants(cmd_, layout, VK_SHADER_STAGE_ALL, 0, size, data);
}

void CmdEncoder::push_constants(u32 size, void* data) {
  assert(size <= 128);
  push_constants(default_pipeline_layout_, size, data);
}

void CmdEncoder::barrier(VkPipelineStageFlags2 src_stage, VkAccessFlags2 src_access,
                         VkPipelineStageFlags2 dst_stage, VkAccessFlags2 dst_access) {
  VkMemoryBarrier2 b{.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
                     .srcStageMask = src_stage,
                     .srcAccessMask = src_access,
                     .dstStageMask = dst_stage,
                     .dstAccessMask = dst_access};
  VkDependencyInfo info{
      .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &b};
  vkCmdPipelineBarrier2KHR(cmd_, &info);
}

void CmdEncoder::set_viewport_and_scissor(u32 width, u32 height) {
  VkViewport viewport{.x = 0,
                      .y = 0,
                      .width = static_cast<float>(width),
                      .height = static_cast<float>(height),
                      .minDepth = 0.f,
                      .maxDepth = 1.f};

  vkCmdSetViewport(cmd_, 0, 1, &viewport);
  VkRect2D scissor{.offset = VkOffset2D{.x = 0, .y = 0},
                   .extent = VkExtent2D{.width = width, .height = height}};
  vkCmdSetScissor(cmd_, 0, 1, &scissor);
}

void CmdEncoder::set_cull_mode(CullMode mode) {
  vkCmdSetCullMode(cmd_, vk2::convert_cull_mode(mode));
}

}  // namespace gfx
