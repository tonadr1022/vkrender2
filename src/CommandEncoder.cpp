#include "CommandEncoder.hpp"

#include <volk.h>
#include <vulkan/vulkan_core.h>

#include "Types.hpp"
#include "core/FixedVector.hpp"
#include "core/Logger.hpp"
#include "vk2/Buffer.hpp"
#include "vk2/Device.hpp"
#include "vk2/Initializers.hpp"
#include "vk2/PipelineManager.hpp"
#include "vk2/VkTypes.hpp"

namespace gfx {

void CmdEncoder::dispatch(u32 work_groups_x, u32 work_groups_y, u32 work_groups_z) const {
  vkCmdDispatch(get_cmd_buf(), work_groups_x, work_groups_y, work_groups_z);
}

void CmdEncoder::bind_descriptor_set(VkPipelineBindPoint bind_point, VkPipelineLayout layout,
                                     VkDescriptorSet* set, u32 idx) const {
  vkCmdBindDescriptorSets(get_cmd_buf(), bind_point, layout, idx, 1, set, 0, nullptr);
}

void CmdEncoder::push_constants(VkPipelineLayout layout, u32 size, void* data) const {
  vkCmdPushConstants(get_cmd_buf(), layout, VK_SHADER_STAGE_ALL, 0, size, data);
}

void CmdEncoder::push_constants(u32 size, void* data) const {
  assert(size <= 128);
  push_constants(default_pipeline_layout_, size, data);
}

void CmdEncoder::barrier(VkPipelineStageFlags2 src_stage, VkAccessFlags2 src_access,
                         VkPipelineStageFlags2 dst_stage, VkAccessFlags2 dst_access) const {
  VkMemoryBarrier2 b{.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
                     .srcStageMask = src_stage,
                     .srcAccessMask = src_access,
                     .dstStageMask = dst_stage,
                     .dstAccessMask = dst_access};
  VkDependencyInfo info{
      .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &b};
  vkCmdPipelineBarrier2KHR(get_cmd_buf(), &info);
}

void CmdEncoder::set_viewport_and_scissor(vec2 extent, vec2 offset) const {
  VkViewport viewport{.x = offset.x,
                      .y = offset.y,
                      .width = static_cast<float>(extent.x),
                      .height = static_cast<float>(extent.y),
                      .minDepth = 0.f,
                      .maxDepth = 1.f};

  vkCmdSetViewport(get_cmd_buf(), 0, 1, &viewport);
  VkRect2D scissor{.offset = VkOffset2D{.x = 0, .y = 0},
                   .extent = VkExtent2D{.width = static_cast<uint32_t>(extent.x),
                                        .height = static_cast<uint32_t>(extent.y)}};
  vkCmdSetScissor(get_cmd_buf(), 0, 1, &scissor);
}

void CmdEncoder::set_viewport_and_scissor(u32 width, u32 height) const {
  VkViewport viewport{.x = 0,
                      .y = 0,
                      .width = static_cast<float>(width),
                      .height = static_cast<float>(height),
                      .minDepth = 0.f,
                      .maxDepth = 1.f};

  vkCmdSetViewport(get_cmd_buf(), 0, 1, &viewport);
  VkRect2D scissor{.offset = VkOffset2D{.x = 0, .y = 0},
                   .extent = VkExtent2D{.width = width, .height = height}};
  vkCmdSetScissor(get_cmd_buf(), 0, 1, &scissor);
}

void CmdEncoder::set_cull_mode(CullMode mode) const {
  vkCmdSetCullModeEXT(get_cmd_buf(), vk2::convert_cull_mode(mode));
}

void CmdEncoder::copy_buffer(const Buffer& src, const Buffer& dst, u64 src_offset, u64 dst_offset,
                             u64 size) const {
  VkBufferCopy2KHR copy{.sType = VK_STRUCTURE_TYPE_BUFFER_COPY_2_KHR,
                        .srcOffset = src_offset,
                        .dstOffset = dst_offset,
                        .size = size};

  VkCopyBufferInfo2KHR copy_info{.sType = VK_STRUCTURE_TYPE_COPY_BUFFER_INFO_2,
                                 .srcBuffer = src.buffer(),
                                 .dstBuffer = dst.buffer(),
                                 .regionCount = 1,
                                 .pRegions = &copy};
  vkCmdCopyBuffer2KHR(get_cmd_buf(), &copy_info);
}

void CmdEncoder::set_depth_bias(float constant_factor, float bias, float slope_factor) const {
  vkCmdSetDepthBias(get_cmd_buf(), constant_factor, bias, slope_factor);
}

void CmdEncoder::bind_pipeline(PipelineBindPoint bind_point, PipelineHandle pipeline) const {
  VkPipelineBindPoint bp{};
  switch (bind_point) {
    case PipelineBindPoint::Graphics:
      bp = VK_PIPELINE_BIND_POINT_GRAPHICS;
      break;
    case PipelineBindPoint::Compute:
      bp = VK_PIPELINE_BIND_POINT_COMPUTE;
      break;
  }
  vkCmdBindPipeline(get_cmd_buf(), bp, PipelineManager::get().get(pipeline)->pipeline);
}

void CmdEncoder::end_rendering() const { vkCmdEndRenderingKHR(get_cmd_buf()); }

namespace {

VkAttachmentLoadOp convert_load_op(LoadOp op) {
  switch (op) {
    default:
    case LoadOp::Load:
      return VK_ATTACHMENT_LOAD_OP_LOAD;
    case LoadOp::Clear:
      return VK_ATTACHMENT_LOAD_OP_CLEAR;
    case LoadOp::DontCare:
      return VK_ATTACHMENT_LOAD_OP_DONT_CARE;
  }
}

VkAttachmentStoreOp convert_store_op(StoreOp op) {
  switch (op) {
    default:
    case StoreOp::Store:
      return VK_ATTACHMENT_STORE_OP_STORE;
    case StoreOp::DontCare:
      return VK_ATTACHMENT_STORE_OP_DONT_CARE;
  }
}

}  // namespace

void CmdEncoder::begin_rendering(const RenderArea& render_area,
                                 std::initializer_list<RenderingAttachmentInfo> attachment_descs) {
  util::fixed_vector<VkRenderingAttachmentInfo, 30> color_atts;
  VkRenderingAttachmentInfo depth_att{};
  VkRenderingAttachmentInfo stencil_att{};
  if (attachment_descs.size() > 30) {
    LCRITICAL("cannot support > 30 color attachments, what are you doing lol");
    exit(1);
  }

  for (const auto& att_desc : attachment_descs) {
    VkRenderingAttachmentInfo& att = att_desc.type == RenderingAttachmentInfo::Type::Color
                                         ? color_atts.emplace_back()
                                         : depth_att;
    att.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    att.imageView =
        device_->get_image_view(att_desc.image, SubresourceType::Attachment, att_desc.subresource);

    if (att_desc.type == RenderingAttachmentInfo::Type::Color) {
      att.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    } else if (att_desc.type == RenderingAttachmentInfo::Type::DepthStencil) {
      att.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
    }

    att.loadOp = convert_load_op(att_desc.load_op);
    att.storeOp = convert_store_op(att_desc.store_op);
    if (att.loadOp == VK_ATTACHMENT_LOAD_OP_CLEAR) {
      att.clearValue.color.float32[0] = att_desc.clear_value.color.r;
      att.clearValue.color.float32[1] = att_desc.clear_value.color.g;
      att.clearValue.color.float32[2] = att_desc.clear_value.color.b;
      att.clearValue.color.float32[3] = att_desc.clear_value.color.a;
      att.clearValue.depthStencil.depth = att_desc.clear_value.depth_stencil.depth;
      att.clearValue.depthStencil.stencil = att_desc.clear_value.depth_stencil.stencil;
    }
  }

  VkRenderingInfo rendering_info{
      .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
      .renderArea = VkRect2D{{render_area.offset.x, render_area.offset.y},
                             {render_area.extent.x, render_area.extent.y}},
      .layerCount = 1,
      .colorAttachmentCount = static_cast<u32>(color_atts.size()),
      .pColorAttachments = color_atts.data(),
      .pDepthAttachment = depth_att.imageLayout != VK_IMAGE_LAYOUT_UNDEFINED ? &depth_att : nullptr,
      .pStencilAttachment =
          stencil_att.imageLayout != VK_IMAGE_LAYOUT_UNDEFINED ? &stencil_att : nullptr};
  vkCmdBeginRenderingKHR(get_cmd_buf(), &rendering_info);
}

void CmdEncoder::draw(u32 vertex_count, u32 instance_count, u32 first_vertex,
                      u32 first_instance) const {
  vkCmdDraw(get_cmd_buf(), vertex_count, instance_count, first_vertex, first_instance);
}

namespace {

VkIndexType convert_index_type(IndexType type) {
  switch (type) {
    case gfx::IndexType::uint8:
      return VK_INDEX_TYPE_UINT8;
    case gfx::IndexType::uint16:
      return VK_INDEX_TYPE_UINT16;
    case gfx::IndexType::uint32:
      return VK_INDEX_TYPE_UINT32;
  }
  return VK_INDEX_TYPE_NONE_KHR;
  ;
}

}  // namespace

void CmdEncoder::bind_index_buffer(BufferHandle buffer, u64 offset, IndexType type) {
  vkCmdBindIndexBuffer(get_cmd_buf(), device_->get_buffer(buffer)->buffer(), offset,
                       convert_index_type(type));
}

void CmdEncoder::fill_buffer(BufferHandle buffer, u64 offset, u64 size, u32 data) {
  vkCmdFillBuffer(get_cmd_buf(), device_->get_buffer(buffer)->buffer(), offset, size, data);
}
void CmdEncoder::draw_indexed_indirect(BufferHandle buffer, u64 offset, u32 draw_count,
                                       u32 stride) {
  vkCmdDrawIndexedIndirect(get_cmd_buf(), device_->get_buffer(buffer)->buffer(), offset, draw_count,
                           stride);
}

void CmdEncoder::draw_indexed_indirect_count(BufferHandle draw_cmd_buf, u64 draw_cmd_offset,
                                             BufferHandle draw_count_buf, u64 draw_count_offset,
                                             u32 draw_count, u32 stride) {
  vkCmdDrawIndexedIndirectCount(get_cmd_buf(), device_->get_buffer(draw_count_buf)->buffer(),
                                draw_cmd_offset, device_->get_buffer(draw_cmd_buf)->buffer(),
                                draw_count_offset, draw_count, stride);
}

void CmdEncoder::update_buffer(BufferHandle buffer, u64 offset, u64 size, void* data) {
  vkCmdUpdateBuffer(get_cmd_buf(), device_->get_buffer(buffer)->buffer(), offset, size, data);
}

void CmdEncoder::begin_region(const char* name) const {
#ifndef NDEBUG
  VkDebugUtilsLabelEXT debug_label_info{.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT,
                                        .pLabelName = name};
  vkCmdBeginDebugUtilsLabelEXT(get_cmd_buf(), &debug_label_info);
#endif
}

void CmdEncoder::end_region() const {
#ifndef NDEBUG
  vkCmdEndDebugUtilsLabelEXT(get_cmd_buf());
#endif
}
void CmdEncoder::transition_image(ImageHandle image, VkImageLayout new_layout,
                                  VkImageAspectFlags aspect) {
  auto* img = device_->get_image(image);
  transition_image(image, img->curr_layout, new_layout, aspect);
  img->curr_layout = new_layout;
}

void CmdEncoder::blit_img(ImageHandle src, ImageHandle dst, uvec3 extent,
                          VkImageAspectFlags aspect) {
  VkImageBlit2 region{
      .sType = VK_STRUCTURE_TYPE_IMAGE_BLIT_2,
      .srcSubresource = {.aspectMask = aspect, .mipLevel = 0, .baseArrayLayer = 0, .layerCount = 1},
      .srcOffsets = {{},
                     {static_cast<i32>(extent.x), static_cast<i32>(extent.y),
                      static_cast<i32>(extent.z)}},
      .dstSubresource = {.aspectMask = aspect, .mipLevel = 0, .baseArrayLayer = 0, .layerCount = 1},
      .dstOffsets = {{},
                     {static_cast<i32>(extent.x), static_cast<i32>(extent.y),
                      static_cast<i32>(extent.z)}}

  };
  VkBlitImageInfo2 blit_info{.sType = VK_STRUCTURE_TYPE_BLIT_IMAGE_INFO_2,
                             .srcImage = device_->get_image(src)->image(),
                             .srcImageLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                             .dstImage = device_->get_image(dst)->image(),
                             .dstImageLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                             .regionCount = 1,
                             .pRegions = &region,
                             .filter = VK_FILTER_NEAREST};
  vkCmdBlitImage2KHR(get_cmd_buf(), &blit_info);
};

void CmdEncoder::transition_image(ImageHandle image, VkImageLayout old_layout,
                                  VkImageLayout new_layout, VkImageAspectFlags aspect) {
  auto* img = device_->get_image(image);
  assert(img);
  VkImageMemoryBarrier2 b{.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
  b.image = img->image();
  b.srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;
  b.srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
  b.dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;
  b.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
  b.oldLayout = old_layout;
  b.newLayout = new_layout;
  img->curr_layout = new_layout;
  b.subresourceRange = VkImageSubresourceRange{.aspectMask = aspect,
                                               .baseMipLevel = 0,
                                               .levelCount = VK_REMAINING_MIP_LEVELS,
                                               .baseArrayLayer = 0,
                                               .layerCount = VK_REMAINING_ARRAY_LAYERS};
  auto dep_info = vk2::init::dependency_info({}, SPAN1(b));
  vkCmdPipelineBarrier2KHR(get_cmd_buf(), &dep_info);
}

void CmdEncoder::reset(u32 frame_in_flight) {
  frame_in_flight_ = frame_in_flight;
  submit_swapchains_.clear();
  wait_semaphores_.clear();
  signal_semaphores_.clear();
}
}  // namespace gfx
