#include "CommandEncoder.hpp"

#include <volk.h>
#include <vulkan/vulkan_core.h>

#include "Types.hpp"
#include "core/FixedVector.hpp"
#include "core/Logger.hpp"
#include "vk2/Buffer.hpp"
#include "vk2/Device.hpp"
#include "vk2/PipelineManager.hpp"
#include "vk2/Texture.hpp"
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

void CmdEncoder::set_viewport_and_scissor(vec2 extent, vec2 offset) {
  VkViewport viewport{.x = offset.x,
                      .y = offset.y,
                      .width = static_cast<float>(extent.x),
                      .height = static_cast<float>(extent.y),
                      .minDepth = 0.f,
                      .maxDepth = 1.f};

  vkCmdSetViewport(cmd_, 0, 1, &viewport);
  VkRect2D scissor{.offset = VkOffset2D{.x = 0, .y = 0},
                   .extent = VkExtent2D{.width = static_cast<uint32_t>(extent.x),
                                        .height = static_cast<uint32_t>(extent.y)}};
  vkCmdSetScissor(cmd_, 0, 1, &scissor);
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
  vkCmdSetCullModeEXT(cmd_, vk2::convert_cull_mode(mode));
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
  vkCmdCopyBuffer2KHR(cmd_, &copy_info);
}

void CmdEncoder::set_depth_bias(float constant_factor, float bias, float slope_factor) {
  vkCmdSetDepthBias(cmd_, constant_factor, bias, slope_factor);
}

void CmdEncoder::bind_pipeline(PipelineBindPoint bind_point, PipelineHandle pipeline) {
  VkPipelineBindPoint bp{};
  switch (bind_point) {
    case PipelineBindPoint::Graphics:
      bp = VK_PIPELINE_BIND_POINT_GRAPHICS;
      break;
    case PipelineBindPoint::Compute:
      bp = VK_PIPELINE_BIND_POINT_COMPUTE;
      break;
  }
  vkCmdBindPipeline(cmd_, bp, PipelineManager::get().get(pipeline)->pipeline);
}

void CmdEncoder::end_rendering() { vkCmdEndRenderingKHR(cmd_); }

namespace {

VkAttachmentLoadOp convert_load_op(RenderingAttachmentInfo::LoadOp op) {
  switch (op) {
    default:
    case RenderingAttachmentInfo::LoadOp::Load:
      return VK_ATTACHMENT_LOAD_OP_LOAD;
    case RenderingAttachmentInfo::LoadOp::Clear:
      return VK_ATTACHMENT_LOAD_OP_CLEAR;
    case RenderingAttachmentInfo::LoadOp::DontCare:
      return VK_ATTACHMENT_LOAD_OP_DONT_CARE;
  }
}

VkAttachmentStoreOp convert_store_op(RenderingAttachmentInfo::StoreOp op) {
  switch (op) {
    default:
    case RenderingAttachmentInfo::StoreOp::Store:
      return VK_ATTACHMENT_STORE_OP_STORE;
    case RenderingAttachmentInfo::StoreOp::DontCare:
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
    VkRenderingAttachmentInfo& att =
        att_desc.type == RenderingAttachmentInfo::Type::Color   ? color_atts.emplace_back()
        : att_desc.type == RenderingAttachmentInfo::Type::Depth ? depth_att
                                                                : stencil_att;
    att.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    auto* view = device_->get_image_view(att_desc.img_view);
    if (view) {
      att.imageView = view->view();
    } else {
      LCRITICAL("can't render, view not found");
      exit(1);
    }

    if (att_desc.type == RenderingAttachmentInfo::Type::Color) {
      att.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    } else if (att_desc.type == RenderingAttachmentInfo::Type::Depth) {
      att.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
    } else if (att_desc.type == RenderingAttachmentInfo::Type::DepthStencil) {
      att.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
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
  vkCmdBeginRenderingKHR(cmd_, &rendering_info);
}

void CmdEncoder::draw(u32 vertex_count, u32 instance_count, u32 first_vertex, u32 first_instance) {
  vkCmdDraw(cmd_, vertex_count, instance_count, first_vertex, first_instance);
}

}  // namespace gfx
