#include "RenderGraph.hpp"

#include <volk.h>
#include <vulkan/vk_enum_string_helper.h>
#include <vulkan/vulkan_core.h>

#include <algorithm>
#include <cstdint>
#include <expected>
#include <fstream>
#include <tracy/Tracy.hpp>
#include <utility>

#include "CommandEncoder.hpp"
#include "Logger.hpp"
#include "Types.hpp"
#include "util/BitOps.hpp"
#include "vk2/Buffer.hpp"
#include "vk2/Device.hpp"
#include "vk2/Texture.hpp"
#include "vk2/VkTypes.hpp"

namespace gfx {
namespace {
bool is_compute_access(Access access) {
  return access & Access::ComputeRead || access & Access::ComputeWrite;
}

void get_vk_stage_access(Access access, VkAccessFlags2& out_access,
                         VkPipelineStageFlags2& out_stages) {
  out_stages = {};
  out_access = {};
  if (is_compute_access(access)) {
    out_stages |= VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
  }
  if (access & Access::ComputeRead) {
    out_access |= VK_ACCESS_2_SHADER_READ_BIT;
  }
  if (access & Access::ComputeWrite) {
    out_access |= VK_ACCESS_2_SHADER_WRITE_BIT;
  }
}
constexpr auto read_flags = Access::ColorRead | Access::ComputeRead | Access::DepthStencilRead |
                            Access::VertexRead | Access::IndexRead | Access::IndirectRead;
bool is_read_access(Access access) { return access & read_flags; }

}  // namespace

void RenderGraphPass::add_buffer(const std::string& name, vk2::BufferHandle buffer, Access access) {
  uint32_t handle = graph_.get_or_add_buffer_resource(name);
  RenderResource& res = *graph_.get_resource(handle);
  res.written_in_pass(idx_);
  auto* buf = vk2::get_device().get_buffer(buffer);
  assert(buf);
  if (!buf) {
    return;
  }
  // TODO: buffer usage flags
  res.buffer_info = {buffer, buf->size()};
  // TODO: usage????????????????????????????????????????????????????
  UsageAndHandle res_usage{.idx = handle};
  get_vk_stage_access(access, res_usage.access, res_usage.stages);
  if (is_read_access(access)) {
    resource_read_indices_.emplace_back(resources_.size());
  }
  resources_.emplace_back(res_usage);
}

// RenderResourceHandle RenderGraphPass::add_buffer_output(const std::string& name, BufferInfo info,
//                                                         const std::string& input) {
//   uint32_t handle = graph_.get_or_add_buffer_resource(name);
//   RenderResource& res = *graph_.get_resource(handle);
//   auto usage = info.usage;
//   assert(info.size > 0);
//   res.buffer_info = info;
//   res.written_in_pass(idx_);
//   UsageAndHandle res_usage{.idx = handle, .usage = ResourceUsage::BufferOutput};
//   if (usage & BufferUsageStorageBufferBit) {
//     res_usage.access |= VK_ACCESS_2_SHADER_WRITE_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT;
//   }
//
//   resources_.emplace_back(res_usage);
//
//   if (!input.empty()) {
//     res.read_in_pass(idx_);
//     uint32_t read_handle = graph_.get_or_add_buffer_resource(input);
//     RenderResource& read_resource = *graph_.get_resource(read_handle);
//     if (read_resource.buffer_info.size != info.size) {
//       LERROR("uh oh read buffer size doesn't match write buffer size! TODO: handle this better");
//       exit(1);
//     }
//     read_resource.buffer_info = info;
//     UsageAndHandle res_usage{.idx = read_handle, .usage = ResourceUsage::BufferInput};
//     if (usage & BufferUsageStorageBufferBit) {
//       res_usage.access |= VK_ACCESS_2_SHADER_READ_BIT;
//     }
//     resources_.emplace_back(res_usage);
//   }
//
//   return handle;
// }

// RenderResourceHandle RenderGraphPass::add_buffer_input(const std::string& name, BufferInfo info)
// {
//   uint32_t handle = graph_.get_or_add_buffer_resource(name);
//   RenderResource& res = *graph_.get_resource(handle);
//   auto usage = info.usage;
//   assert(info.size > 0);
//   res.buffer_info = info;
//   res.read_in_pass(idx_);
//
//   // TODO: resourceusage
//   UsageAndHandle res_usage{.idx = handle, .usage = ResourceUsage::BufferInput};
//   if (usage & BufferUsageIndexBufferBit) {
//     res_usage.access |= VK_ACCESS_2_INDEX_READ_BIT;
//     res_usage.stages |= VK_PIPELINE_STAGE_2_VERTEX_INPUT_BIT;
//   }
//   if (usage & BufferUsageIndirectBufferBit) {
//     res_usage.access |= VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT;
//     res_usage.stages |= VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT;
//   }
//   if (usage & BufferUsageVertexBufferBit) {
//     res_usage.access |= VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT;
//     res_usage.stages |= VK_PIPELINE_STAGE_2_VERTEX_INPUT_BIT;
//   }
//   if (usage & BufferUsageUniformBufferBit) {
//     assert(0 && "unhandled");
//     LERROR("uniform buffer not handled right now");
//     exit(1);
//     res_usage.access |= VK_ACCESS_2_UNIFORM_READ_BIT;
//   }
//   if (usage & BufferUsageStorageBufferBit) {
//     res_usage.access |= VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
//   }
//   resources_.emplace_back(res_usage);
//   return handle;
// }

RenderResourceHandle RenderGraphPass::add_color_output(const std::string& name,
                                                       const AttachmentInfo& info,
                                                       const std::string& input) {
  // TODO: queue
  uint32_t handle = graph_.get_or_add_texture_resource(name);
  RenderResource* res = graph_.get_resource(handle);
  res->image_usage |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
  res->written_in_pass(idx_);
  // TODO: more robust
  assert(info.format != Format::Undefined);
  res->info = info;
  if (input.size()) {
    assert(0 && "unimplemented");
  }
  UsageAndHandle usage{
      .idx = handle,
      .access = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
      .usage = ResourceUsage::ColorOutput,
  };
  resources_.emplace_back(usage);
  return handle;
}

RenderResourceHandle RenderGraphPass::add_storage_image_input(const std::string& name) {
  uint32_t handle = graph_.get_or_add_texture_resource(name);
  RenderResource* res = graph_.get_resource(handle);
  res->read_in_pass(idx_);
  res->image_usage |= VK_IMAGE_USAGE_STORAGE_BIT;
  UsageAndHandle usage{.idx = handle, .usage = ResourceUsage::StorageImageInput};
  usage.access = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
  usage.stages = type_ == Type::Graphics ? VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT
                                         : VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;

  resource_read_indices_.emplace_back(resources_.size());
  resources_.emplace_back(usage);

  return handle;
}
RenderResourceHandle RenderGraphPass::add_texture_input(const std::string& name) {
  uint32_t handle = graph_.get_or_add_texture_resource(name);
  RenderResource* res = graph_.get_resource(handle);
  res->read_in_pass(idx_);
  res->image_usage |= VK_IMAGE_USAGE_SAMPLED_BIT;
  UsageAndHandle usage{.idx = handle, .usage = ResourceUsage::TextureInput};
  usage.access = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
  usage.stages = type_ == Type::Graphics ? VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT
                                         : VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;

  resource_read_indices_.emplace_back(resources_.size());
  resources_.emplace_back(usage);

  return handle;
}

RenderResourceHandle RenderGraphPass::set_depth_stencil_input(const std::string& name) {
  uint32_t handle = graph_.get_or_add_texture_resource(name);
  RenderResource* res = graph_.get_resource(handle);
  res->read_in_pass(idx_);
  res->image_usage |= VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
  UsageAndHandle usage{.idx = handle, .usage = ResourceUsage::DepthStencilInput};
  resource_read_indices_.emplace_back(resources_.size());
  resources_.emplace_back(usage);
  return handle;
}

RenderResourceHandle RenderGraphPass::set_depth_stencil_output(const std::string& name,
                                                               const AttachmentInfo& info) {
  uint32_t handle = graph_.get_or_add_texture_resource(name);
  RenderResource* res = graph_.get_resource(handle);
  res->written_in_pass(idx_);
  res->image_usage |= VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
  res->info = info;
  assert(info.format != Format::Undefined);
  UsageAndHandle usage{.idx = handle, .usage = ResourceUsage::DepthStencilOutput};
  resources_.emplace_back(usage);
  return handle;
}

RenderGraphPass::RenderGraphPass(std::string name, RenderGraph& graph, uint32_t idx, Type type)
    : name_(std::move(name)), graph_(graph), idx_(idx), type_(type) {}

RenderGraph::RenderGraph(std::string name) : name_(std::move(name)) {}

RenderGraphPass& RenderGraph::add_pass(const std::string& name, RenderGraphPass::Type type) {
  auto idx = passes_.size();
  passes_.emplace_back(name, *this, idx, type);
  return passes_.back();
}

VoidResult RenderGraph::validate() {
  return VoidResult{};
  for (auto& pass : passes_) {
    if (!pass.execute_) {
      return std::unexpected("pass missing execute function");
    }
  }
  // TODO: more validation lmao
}

VoidResult RenderGraph::bake() {
  ZoneScoped;
  bool log = false;
  // validate
  // go through each input and make sure it's an output of a previous pass if it needs to read from
  // it for (auto& pass : passes_) {
  //   // for (auto& tex_resource : pass.)
  // }

  if (auto ok = validate(); !ok) {
    return ok;
  }

  if (log) {
    for (const auto& resource : resources_) {
      for (const auto& b : resource.get_read_passes()) {
        LINFO("{}: read in {}", resource.name, passes_[b].get_name());
      }
      for (const auto& b : resource.get_written_passes()) {
        LINFO("{}: written in {}", resource.name, passes_[b].get_name());
      }
    }
  }

  // TODO: validate that backbuffer img has color write

  {
    ZoneScopedN("find sinks");
    // find sinks
    pass_stack_.clear();
    swapchain_writer_passes_.clear();
    pass_dependencies_.resize(passes_.size());
    for (uint32_t pass_i = 0; pass_i < passes_.size(); pass_i++) {
      auto& pass = passes_[pass_i];
      for (const auto& usage : pass.get_resources()) {
        if (resources_[usage.idx].name == backbuffer_img_) {
          // TODO: move this to validation phase
          if (!(usage.access & VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT)) {
            return std::unexpected("backbuffer output is not of usage ColorOutput");
          }
          pass_stack_.emplace_back(pass_i);
          swapchain_writer_passes_.emplace_back(pass_i);
        }
      }
    }
  }
  auto sink_cnt = pass_stack_.size();
  if (sink_cnt == 0) {
    return std::unexpected("no backbuffer writes found");
  }

  {
    ZoneScopedN("topo sort passes");
    // starting from the sinks, traverse the dependencies
    for (uint32_t i = 0; i < sink_cnt; i++) {
      assert(pass_stack_[i] < passes_.size());
      if (auto res = traverse_dependencies_recursive(pass_stack_[i], 0); !res) {
        return res;
      }
    }
    // reverse since sinks were added first, initial passes added last
    std::ranges::reverse(pass_stack_);
    // only execute a pass once
    prune_duplicates(pass_stack_);

    if (log) {
      LINFO("pass order: ");
      for (const auto& s : pass_stack_) {
        LINFO("{}", passes_[s].get_name());
      }
    }
  }

  build_physical_resource_reqs();

  if (log) {
    for (auto& res : resources_) {
      if (res.get_type() == ResourceType::Texture) {
        assert(res.physical_idx != RenderResource::unused);
        LINFO("{} {}", res.name, res.physical_idx);
        assert(res.physical_idx < physical_resource_dims_.size());
        auto& dims = physical_resource_dims_[res.physical_idx];
        LINFO("{} {}", res.physical_idx, string_VkImageUsageFlags(dims.image_usage_flags));
      }
    }
  }

  {
    ZoneScopedN("build physical passes");
    clear_physical_passes();
    physical_passes_.reserve(passes_.size());
    for (const auto& pass_i : pass_stack_) {
      const auto& pass = passes_[pass_i];
      PhysicalPass phys_pass = get_physical_pass();
      // TODO: don't allocate string here
      phys_pass.name = pass.get_name();

      for (const auto& output : pass.get_resources()) {
        if (is_texture_usage(output.usage)) {
          auto* tex = get_resource(output.idx);
          if (output.access & VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT) {
            phys_pass.physical_color_attachments.emplace_back(tex->physical_idx);
          } else if (output.access & VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT) {
            phys_pass.physical_depth_stencil = tex->physical_idx;
          }
        }
      }
      physical_passes_.emplace_back(std::move(phys_pass));
    }
  }

  if (log) {
    LINFO("\nphysical passes\n");
    for (auto& pass : physical_passes_) {
      LINFO("phys pass: {}", pass.name);
      for (auto& out : pass.physical_color_attachments) {
        LINFO("color out: {}", out);
      }
      if (pass.physical_depth_stencil != RenderResource::unused) {
        LINFO("depth stencil out: {}", pass.physical_depth_stencil);
      }
    }
  }

  build_barrier_infos();

  {
    const auto flush_access_to_invalidate = [](VkAccessFlags2 flags) -> VkAccessFlags2 {
      if (flags & VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT) {
        flags |= VK_ACCESS_COLOR_ATTACHMENT_READ_BIT;
      }
      if (flags & VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT) {
        flags |= VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT;
      }
      if (flags & VK_ACCESS_SHADER_WRITE_BIT) flags |= VK_ACCESS_SHADER_READ_BIT;
      if (flags & VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT) {
        flags |= VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
      }
      return flags;
    };

    const auto flush_stage_to_invalidate =
        [](VkPipelineStageFlags2 flags) -> VkPipelineStageFlags2 {
      if (flags & VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT) {
        flags |= VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
      }
      return flags;
    };
    struct ResourceState {
      VkImageLayout initial_layout{};
      VkImageLayout final_layout{};
      VkAccessFlags2 invalidated_accesses{};
      VkAccessFlags2 flushed_accesses{};
      VkPipelineStageFlags2 invalidated_stages{};
      VkPipelineStageFlags2 flushed_stages{};
    };
    // TODO: don't realloc
    std::vector<ResourceState> resource_state;
    resource_state.reserve(physical_resource_dims_.size());
    for (auto& phys_pass : physical_passes_) {
      resource_state.clear();
      resource_state.resize(physical_resource_dims_.size());
      for (auto& invalidate : phys_pass.invalidate_barriers) {
        auto& res = resource_state[invalidate.resource_idx];

        // only first use of resource in physical pass needs to be handled
        if (res.initial_layout == VK_IMAGE_LAYOUT_UNDEFINED) {
          res.invalidated_accesses |= invalidate.access;
          res.invalidated_stages |= invalidate.stages;
        }
        // keep storage images in general layout rather than transition to/from
        // SHADER_READ_ONLY_OPTIMAL
        if (physical_resource_dims_[invalidate.resource_idx].is_storage_image()) {
          res.initial_layout = VK_IMAGE_LAYOUT_GENERAL;
        } else {
          res.initial_layout = invalidate.layout;
        }

        if (physical_resource_dims_[invalidate.resource_idx].is_storage_image()) {
          res.final_layout = VK_IMAGE_LAYOUT_GENERAL;
        } else {
          res.final_layout = invalidate.layout;
        }

        // pending flushes have been invalidated already
        res.flushed_stages = 0;
        res.flushed_accesses = 0;
      }

      for (auto& flush : phys_pass.flush_barriers) {
        auto& res = resource_state[flush.resource_idx];
        res.flushed_stages |= flush.stages;
        res.flushed_accesses |= flush.access;

        // keep storage images in general layout
        if (physical_resource_dims_[flush.resource_idx].is_storage_image()) {
          res.final_layout = VK_IMAGE_LAYOUT_GENERAL;
        } else {
          res.final_layout = flush.layout;
        }

        if (res.initial_layout == VK_IMAGE_LAYOUT_UNDEFINED) {
          // swapchain writes
          if (flush.layout == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL) {
            res.initial_layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            res.invalidated_accesses = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
            res.invalidated_accesses =
                VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
          } else {
            res.initial_layout = flush.layout;
            res.invalidated_stages = flush_stage_to_invalidate(flush.stages);
            res.invalidated_accesses = flush_access_to_invalidate(flush.access);
          }
          phys_pass.discard_resources.emplace_back(flush.resource_idx);
        }
      }

      for (u32 resource_i = 0; resource_i < resource_state.size(); resource_i++) {
        auto& res = resource_state[resource_i];
        if (physical_resource_dims_[resource_i].is_image()) {
          if (res.final_layout == VK_IMAGE_LAYOUT_UNDEFINED &&
              res.initial_layout == VK_IMAGE_LAYOUT_UNDEFINED) {
            continue;
          }
          assert(res.final_layout != VK_IMAGE_LAYOUT_UNDEFINED);
          phys_pass.invalidate_barriers.emplace_back(Barrier{.resource_idx = resource_i,
                                                             .layout = res.final_layout,
                                                             .access = res.invalidated_accesses,
                                                             .stages = res.invalidated_stages});
          if (res.flushed_accesses) {
            phys_pass.flush_barriers.emplace_back(Barrier{.resource_idx = resource_i,
                                                          .layout = res.final_layout,
                                                          .access = res.flushed_accesses,
                                                          .stages = res.flushed_stages});
          } else if (res.invalidated_accesses) {
            // if pass read something that needs protection before writing, need flush with no
            // access sets the last pass which the resource was used as a stage
            phys_pass.flush_barriers.emplace_back(Barrier{.resource_idx = resource_i,
                                                          .layout = res.final_layout,
                                                          .access = 0,
                                                          .stages = res.invalidated_stages});
          }
        } else {
          // if (res.flushed_stages || res.flushed_accesses) {
          //   phys_pass.flush_barriers.emplace_back(Barrier{.resource_idx = resource_i,
          //                                                 .layout = res.final_layout,
          //                                                 .access = 0,
          //                                                 .stages = res.invalidated_stages});
          // }
          // phys_pass.flush_barriers.emplace_back(Barrier{.resource_idx = resource_i})
          // LINFO("what?");
        }
      }
    }
  }

  return {};
}

void RenderGraph::execute(CmdEncoder& cmd) {
  ZoneScoped;
  if (swapchain_info_.curr_img == nullptr || swapchain_info_.height == 0 ||
      swapchain_info_.width == 0) {
    LERROR("invalid swapchain info");
    return;
  }
  resource_pipeline_states_.resize(physical_resource_dims_.size());

  {
    pass_submission_state_.resize(physical_passes_.size());
    for (auto& p : pass_submission_state_) {
      p.reset();
    }
    ZoneScopedN("setup barriers");
    for (u32 pass_i = 0; pass_i < physical_passes_.size(); pass_i++) {
      physical_pass_setup_barriers(pass_i);
    }
  }

  {
    ZoneScopedN("Record commands");
    for (u32 pass_i : pass_stack_) {
      auto& pass = passes_[pass_i];
      auto& submission_state = pass_submission_state_[pass_i];

      VkDependencyInfo info{.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
      info.bufferMemoryBarrierCount = submission_state.buffer_barriers.size();
      info.pBufferMemoryBarriers = submission_state.buffer_barriers.data();
      info.imageMemoryBarrierCount = submission_state.image_barriers.size();
      info.pImageMemoryBarriers = submission_state.image_barriers.data();
      LINFO("barriers: {}", pass.get_name());
      LINFO("buffers");
      for (auto& b : submission_state.buffer_barriers) {
        print_barrier(b);
      }
      LINFO("images");
      for (auto& b : submission_state.image_barriers) {
        print_barrier(b);
      }
      LINFO("");
      vkCmdPipelineBarrier2KHR(cmd.cmd(), &info);

      pass.execute_(cmd);
    }
    // exit(1);
  }

  // blit to swapchain
  {
    ZoneScopedN("blit to swapchain");
    {
      // make swapchain img writeable in blit stage
      VkImageMemoryBarrier2 img_barriers[] = {
          VkImageMemoryBarrier2{
              .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
              .srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT |
                              VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
              .srcAccessMask = 0,
              .dstStageMask = VK_PIPELINE_STAGE_2_BLIT_BIT,
              .dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
              .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
              .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
              .image = swapchain_info_.curr_img,
              .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
          },
      };
      VkDependencyInfo info{.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                            .imageMemoryBarrierCount = COUNTOF(img_barriers),
                            .pImageMemoryBarriers = img_barriers};
      vkCmdPipelineBarrier2KHR(cmd.cmd(), &info);
    }

    for (u32 pass_i : swapchain_writer_passes_) {
      auto& pass = passes_[pass_i];
      bool pass_done = false;
      for (const auto& output : pass.get_resources()) {
        if (pass_done) {
          break;
        }
        // NOTE: assumes swapchain writers only have one color output
        if (output.access & VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT) {
          auto* resource = get_resource(output.idx);
          auto* tex = get_texture(output.idx);
          assert(resource && tex && resource->physical_idx < resource_pipeline_states_.size());
          if (!resource || !tex || resource->physical_idx >= resource_pipeline_states_.size()) {
            continue;
          }
          pass_done = true;

          auto& state = resource_pipeline_states_[resource->physical_idx];
          VkImageMemoryBarrier2 img_barriers[] = {
              VkImageMemoryBarrier2{
                  .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                  .srcStageMask = state.pipeline_barrier_src_stages,
                  .srcAccessMask = state.to_flush_access,
                  .dstStageMask = VK_PIPELINE_STAGE_2_BLIT_BIT,
                  .dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT,
                  .oldLayout = state.layout,
                  .newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                  .image = tex->image(),
                  .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
              },
          };
          // print_barrier(img_barriers[0]);
          VkDependencyInfo info{.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                                .imageMemoryBarrierCount = COUNTOF(img_barriers),
                                .pImageMemoryBarriers = img_barriers};
          vkCmdPipelineBarrier2KHR(cmd.cmd(), &info);

          VkExtent3D dims{glm::min(tex->create_info().extent.width, swapchain_info_.width),
                          glm::min(tex->create_info().extent.height, swapchain_info_.height), 1};
          vk2::blit_img(cmd.cmd(), tex->image(), swapchain_info_.curr_img, dims,
                        VK_IMAGE_ASPECT_COLOR_BIT);

          // write after read barrier
          state.layout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
          state.to_flush_access = 0;
          for (auto& e : state.invalidated_in_stage) e = 0;
          state.invalidated_in_stage[util::trailing_zeros(VK_PIPELINE_STAGE_2_BLIT_BIT)] =
              VK_ACCESS_2_TRANSFER_READ_BIT;
          state.pipeline_barrier_src_stages = VK_PIPELINE_STAGE_2_BLIT_BIT;
        }
      }
    }
    {
      VkImageMemoryBarrier2 img_barriers[] = {
          VkImageMemoryBarrier2{
              .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
              .srcStageMask = VK_PIPELINE_STAGE_2_BLIT_BIT,
              .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
              .dstStageMask = VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT,
              .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT,
              .oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
              .newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
              .image = swapchain_info_.curr_img,
              .subresourceRange = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                                   .levelCount = 1,
                                   .layerCount = 1},
          },
      };
      VkDependencyInfo info{.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                            .imageMemoryBarrierCount = COUNTOF(img_barriers),
                            .pImageMemoryBarriers = img_barriers};
      vkCmdPipelineBarrier2KHR(cmd.cmd(), &info);
    }
  }
}

// TODO: not recursive
VoidResult RenderGraph::traverse_dependencies_recursive(uint32_t pass_i, uint32_t stack_size) {
  if (stack_size > passes_.size()) {
    return std::unexpected("cycle detected");
  }

  auto& pass = passes_[pass_i];
  for (const uint32_t& read_i : pass.resource_read_indices_) {
    auto& resource_read = pass.resources_[read_i];
    std::span<const uint16_t> passes_writing_to_resource{};
    auto* input = get_resource(resource_read.idx);
    assert(input);
    passes_writing_to_resource = input->get_written_passes();

    // find outputs that write to this input
    if (passes_writing_to_resource.empty() && resource_read.usage != ResourceUsage::BufferInput) {
      return std::unexpected("no pass exists which writes to resource");
    }

    if (stack_size > passes_.size()) {
      return std::unexpected("cycle detected");
    }
    stack_size++;
    for (uint32_t pass_writing_to_resource : passes_writing_to_resource) {
      if (pass_writing_to_resource == pass.get_idx()) {
        // TODO: maybe bad!
        continue;
        return std::unexpected("pass depends on itself");
      }
      pass_stack_.push_back(pass_writing_to_resource);
      pass_dependencies_[pass.get_idx()].insert(pass_writing_to_resource);
      if (auto res = traverse_dependencies_recursive(pass_writing_to_resource, stack_size); !res) {
        return res;
      }
    }
  }
  return {};
}

void RenderGraph::prune_duplicates(std::vector<uint32_t>& data) {
  ZoneScoped;
  dup_prune_set_.clear();
  if (data.size() <= 1) return;
  auto write_it = data.begin();
  for (uint32_t el : data) {
    if (!dup_prune_set_.contains(el)) {
      *write_it = el;
      dup_prune_set_.insert(el);
      write_it++;
    }
  }
  data.resize(std::distance(data.begin(), write_it));
}

uint32_t RenderGraph::get_or_add_buffer_resource(const std::string& name) {
  auto it = resource_to_idx_map_.find(name);
  if (it != resource_to_idx_map_.end()) {
    uint32_t idx = it->second;
    if (resources_[idx].get_type() != ResourceType::Buffer) {
      LERROR("resource already exists and is not a buffer: {}", name);
      exit(1);
    }
    return it->second;
  }

  uint32_t idx = resources_.size();
  resource_to_idx_map_.emplace(name, idx);
  resources_.emplace_back(ResourceType::Buffer, idx);
  resources_[idx].name = name;
  return idx;
}

uint32_t RenderGraph::get_or_add_texture_resource(const std::string& name) {
  auto it = resource_to_idx_map_.find(name);
  if (it != resource_to_idx_map_.end()) {
    uint32_t idx = it->second;
    assert(idx < resources_.size());
    assert(resources_[idx].get_type() == ResourceType::Texture);
    assert(resources_[idx].name == name);
    return it->second;
  }

  uint32_t idx = resources_.size();
  resource_to_idx_map_.emplace(name, idx);
  resources_.emplace_back(ResourceType::Texture, idx);
  resources_[idx].name = name;
  return idx;
}

RenderResource* RenderGraph::get_resource(uint32_t idx) {
  assert(idx < resources_.size());
  if (idx >= resources_.size()) {
    return nullptr;
  }
  return &resources_[idx];
}

bool is_texture_usage(ResourceUsage usage) {
  switch (usage) {
    case ResourceUsage::ColorOutput:
    case ResourceUsage::StorageImageInput:
    // case ResourceUsage::ColorInput:
    case ResourceUsage::TextureInput:
    case ResourceUsage::DepthStencilOutput:
    case ResourceUsage::DepthStencilInput:
      return true;
    case gfx::ResourceUsage::None:
    case gfx::ResourceUsage::BufferInput:
    case gfx::ResourceUsage::BufferOutput:
      return false;
  }
  return false;
}

ResourceDimensions RenderGraph::get_resource_dims(const RenderResource& resource) const {
  ResourceDimensions dims{};
  if (resource.get_type() == ResourceType::Buffer) {
    assert(resource.buffer_info.size > 0);
    dims.buffer_info = resource.buffer_info;
  } else {
    assert(swapchain_info_.width > 0 && swapchain_info_.height > 0);
    dims = {.format = resource.info.format,
            .size_class = resource.info.size_class,
            .depth = 1,
            .layers = 1,
            .levels = 1,
            .samples = 1,
            .image_usage_flags = resource.image_usage};
    if (resource.info.size_class == SizeClass::SwapchainRelative) {
      dims.width = swapchain_info_.width;
      dims.height = swapchain_info_.height;
    } else if (resource.info.size_class == SizeClass::Absolute) {
      dims.width = (uint32_t)resource.info.size_x;
      dims.height = (uint32_t)resource.info.size_y;
    } else {  // InputRelative
      assert(0 && "nope");
    }
  }
  return dims;
}

void RenderGraph::set_swapchain_info(const RenderGraphSwapchainInfo& info) {
  swapchain_info_ = info;
}

void RenderGraph::build_physical_resource_reqs() {
  ZoneScoped;
  physical_resource_dims_.clear();
  physical_resource_dims_.reserve(resources_.size());
  // build physical resources
  for (uint32_t pass_i : pass_stack_) {
    auto& pass = passes_[pass_i];

    for (const auto& resource_usage : pass.get_resources()) {
      auto* res = get_resource(resource_usage.idx);
      if (res->physical_idx == RenderResource::unused) {
        assert(res);
        res->physical_idx = physical_resource_dims_.size();
        if (is_texture_usage(resource_usage.usage) && res->name == backbuffer_img_) {
          res->image_usage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        }
        physical_resource_dims_.emplace_back(get_resource_dims(*res));
      } else {
        if (is_texture_usage(resource_usage.usage)) {
          physical_resource_dims_[res->physical_idx].image_usage_flags |= res->image_usage;
        } else {
          assert(physical_resource_dims_[res->physical_idx].buffer_info.size ==
                 res->buffer_info.size);
          physical_resource_dims_[res->physical_idx].buffer_info.buffer_usage_flags |=
              res->buffer_info.buffer_usage_flags;
        }
      }
    }
  }
}

void RenderGraph::clear_physical_passes() {
  while (physical_passes_.size()) {
    physical_pass_unused_pool_.emplace_back(std::move(physical_passes_.back()));
    physical_passes_.pop_back();
  }
}

RenderGraph::PhysicalPass RenderGraph::get_physical_pass() {
  if (physical_pass_unused_pool_.empty()) {
    return PhysicalPass{};
  }
  auto pass = std::move(physical_pass_unused_pool_.back());
  pass.reset();
  physical_pass_unused_pool_.pop_back();
  return pass;
}

void RenderGraph::PhysicalPass::reset() {
  name.clear();
  invalidate_barriers.clear();
  flush_barriers.clear();
  physical_color_attachments.clear();
  physical_depth_stencil = RenderResource::unused;
}

VoidResult RenderGraph::output_graphvis(const std::filesystem::path& path) {
  if (physical_passes_.empty() || passes_.empty()) {
    return std::unexpected("no passes");
  }

  std::ofstream ofs(path);
  if (!ofs.is_open()) {
    return std::unexpected("failed to open file");
  }
  ofs << "digraph G {\nsize =\"4,4\";\n";
  auto s = pass_stack_;
  auto print_link = [&](const std::string& source, const std::string& sink,
                        const std::string& label = "") {
    if (label.size()) {
      std::println(ofs, "{} -> {} [label=\"{}\"];\n", source, sink, label);
    } else {
      std::println(ofs, "{} -> {};\n", source, sink);
    }
  };
  std::ranges::reverse(s);
  // for (auto p : s) {
  //   auto& pass = passes_[p];
  //   for (const auto& writer : pass_dependencies_[p]) {
  //     for (const auto& resource : passes_[writer].get_resources()) {
  //       assert(resource.idx < resources_.size());
  //       auto& output_res = resources_[resource.idx];
  //       for (const auto& input : pass.get_resources()) {
  //         if (output_res.name == resources_[input.idx].name) {
  //           print_link(passes_[writer].get_name(), pass.get_name(), output_res.name);
  //         }
  //       }
  //     }
  //   }
  // }

  for (uint32_t writer : swapchain_writer_passes_) {
    for (const auto& output : passes_[writer].get_resources()) {
      auto& res = resources_[output.idx];
      if (res.name == backbuffer_img_) {
        print_link(passes_[writer].get_name(), "swapchain", res.name);
      }
    }
  }
  ofs << "}\n";

  return {};
}

void RenderGraph::setup_attachments() {
  // only make attachments if they don't match prev frame
  // check if buffer/img in each slot works, otherwise make a new one
  physical_image_attachments_.resize(physical_resource_dims_.size());
  physical_buffers_.resize(physical_resource_dims_.size());
  resource_pipeline_states_.resize(physical_resource_dims_.size());

  for (size_t i = 0; i < physical_resource_dims_.size(); i++) {
    auto& dims = physical_resource_dims_[i];
    if (dims.is_image()) {
      auto* img = vk2::get_device().get_image(physical_image_attachments_[i].handle);
      bool is_reusable_img = false;

      if (!img) {
        physical_buffers_[i] = dims.buffer_info.handle;
      } else {
        const auto& cinfo = img->create_info();
        bool valid_extent = false;
        if (dims.size_class == SizeClass::SwapchainRelative) {
          valid_extent = cinfo.extent.width == swapchain_info_.width &&
                         cinfo.extent.height == swapchain_info_.height;
          dims.width = swapchain_info_.width;
          dims.height = swapchain_info_.height;
        } else {
          valid_extent = cinfo.extent.width == dims.width && cinfo.extent.width == dims.height &&
                         cinfo.extent.depth == dims.depth;
        }
        if (valid_extent && cinfo.array_layers == dims.layers && cinfo.mip_levels == dims.levels &&
            cinfo.samples == dims.samples && cinfo.format == vk2::to_vkformat(dims.format) &&
            cinfo.override_usage_flags == dims.image_usage_flags) {
          is_reusable_img = true;
        }
      }

      if (!is_reusable_img) {
        resource_pipeline_states_[i] = {};
        vk2::ImageCreateInfo info{};
        if (dims.depth == 1) {
          info.view_type = VK_IMAGE_VIEW_TYPE_2D;
        } else {
          info.view_type = VK_IMAGE_VIEW_TYPE_3D;
        }
        info.extent = VkExtent3D{dims.width, dims.height, dims.depth};
        info.array_layers = dims.layers;
        info.mip_levels = dims.levels;
        info.samples = (VkSampleCountFlagBits)(1 << (dims.samples - 1));
        info.format = vk2::to_vkformat(dims.format);
        info.override_usage_flags = dims.image_usage_flags;
        physical_image_attachments_[i] = vk2::get_device().create_image_holder(info);
        img = vk2::get_device().get_image(physical_image_attachments_[i].handle);
      }
    }
  }
}

void RenderGraph::build_barrier_infos() {
  ZoneScoped;
  auto get_access = [](std::vector<Barrier>& barriers, uint32_t idx) -> Barrier& {
    auto it = std::ranges::find_if(
        barriers, [idx](const Barrier& barrier) { return barrier.resource_idx == idx; });
    if (it != barriers.end()) {
      return *it;
    }
    return barriers.emplace_back(Barrier{
        .resource_idx = idx, .layout = VK_IMAGE_LAYOUT_UNDEFINED, .access = 0, .stages = 0});
  };

  for (const auto& pass_i : pass_stack_) {
    auto& pass = passes_[pass_i];
    auto& phys_pass = physical_passes_[pass_i];
    auto get_invalidate_access = [&get_access, &phys_pass](uint32_t idx) -> Barrier& {
      return get_access(phys_pass.invalidate_barriers, idx);
    };
    auto get_flush_access = [&get_access, &phys_pass](uint32_t idx) -> Barrier& {
      return get_access(phys_pass.flush_barriers, idx);
    };

    for (const auto& resource : pass.get_resources()) {
      auto& res = resources_[resource.idx];
      if (resource.usage == ResourceUsage::BufferInput) {
        auto& barrier = get_invalidate_access(res.physical_idx);
        barrier.access |= resource.access;
        barrier.stages |= resource.stages | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
      } else if (resource.usage == ResourceUsage::BufferOutput) {
        auto& barrier = get_flush_access(res.physical_idx);
        // TODO: better stage
        barrier.access |= resource.access;
        barrier.stages |= resource.stages | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
      } else if (resource.usage == ResourceUsage::DepthStencilOutput) {
        auto& barrier = get_flush_access(res.physical_idx);
        barrier.stages |= VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT |
                          VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
        barrier.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        barrier.access |= VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
      } else if (resource.usage == ResourceUsage::ColorOutput) {
        auto& barrier = get_flush_access(res.physical_idx);
        barrier.stages |= VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
        barrier.access |= VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
        barrier.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
      } else if (resource.usage == ResourceUsage::TextureInput ||
                 resource.usage == ResourceUsage::StorageImageInput) {
        auto& barrier = get_invalidate_access(res.physical_idx);
        barrier.stages |= resource.stages;
        barrier.layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        barrier.access = resource.access;
      } else {
        assert(0);
      }
    }
  }

  // for (const auto& pass_i : pass_stack_) {
  //   auto& pass = passes_[pass_i];
  //   auto& phys_pass = physical_passes_[pass_i];
  //   LINFO("\npass: {}", pass.get_name());
  //   LINFO("\nflush barriers (writes)");
  //   for (const auto& barrier : phys_pass.flush_barriers) {
  //     LINFO("access: {}, layout: {}, idx: {}, stages: {}", string_VkAccessFlags2(barrier.access),
  //           string_VkImageLayout(barrier.layout), barrier.resource_idx,
  //           string_VkPipelineStageFlags2(barrier.stages));
  //   }
  // }
  // TODO: aliasing to reuse images
}
bool ResourceDimensions::is_storage_image() const {
  return image_usage_flags & VK_IMAGE_USAGE_STORAGE_BIT;
}
void RenderGraph::PassSubmissionState::reset() {
  image_barriers.clear();
  buffer_barriers.clear();
}
bool ResourceDimensions::is_image() const { return buffer_info.size == 0; }

// Are there any access types in the barrier that haven’t been invalidated in any of the relevant
// stages?
bool RenderGraph::needs_invalidate(const Barrier& barrier, const ResourceState& state) {
  bool needs_invalidate = false;
  util::for_each_bit(barrier.stages, [&](u32 bit) {
    if (barrier.access & ~state.invalidated_in_stage[bit]) {
      needs_invalidate = true;
    }
  });
  return needs_invalidate;
}

void RenderGraph::physical_pass_setup_barriers(u32 pass_i) {
  ZoneScoped;
  auto& state = pass_submission_state_[pass_i];
  auto& pass = physical_passes_[pass_i];

  // place barriers
  for (const auto& barrier : pass.invalidate_barriers) {
    bool layout_change = false;
    bool need_pipeline_barrier = false;
    const auto& phys_dims = physical_resource_dims_[barrier.resource_idx];
    auto& resource_state = resource_pipeline_states_[barrier.resource_idx];
    if (phys_dims.is_image()) {
      // queue families ignored
      const auto* image =
          vk2::get_device().get_image(physical_image_attachments_[barrier.resource_idx].handle);
      assert(image);
      if (!image) continue;

      VkImageMemoryBarrier2 b{.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
      b.oldLayout = resource_state.layout;
      b.newLayout = barrier.layout;
      b.srcAccessMask = resource_state.to_flush_access;
      b.dstAccessMask = barrier.access;
      b.dstStageMask = barrier.stages;
      b.image = image->image();
      b.subresourceRange.aspectMask = vk2::format_to_aspect_flags(image->create_info().format);
      b.subresourceRange.layerCount = image->create_info().array_layers;
      b.subresourceRange.levelCount = image->create_info().mip_levels;

      layout_change = b.oldLayout != b.newLayout;
      bool needs_sync = layout_change || needs_invalidate(barrier, resource_state);

      if (needs_sync) {
        // LINFO("pass: {}", pass_i);
        // print_barrier(b);
        if (resource_state.pipeline_barrier_src_stages) {
          b.srcStageMask = resource_state.pipeline_barrier_src_stages;
          need_pipeline_barrier = true;
        } else {
          b.srcStageMask = VK_PIPELINE_STAGE_NONE;
          b.srcAccessMask = 0;
          assert(b.oldLayout == VK_IMAGE_LAYOUT_UNDEFINED);
        }
        state.image_barriers.push_back(b);
      }

      resource_state.layout = barrier.layout;

    } else {
      VkBufferMemoryBarrier2 b{.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2};
      auto& resource_state = resource_pipeline_states_[barrier.resource_idx];
      const auto* buffer = vk2::get_device().get_buffer(physical_buffers_[barrier.resource_idx]);
      assert(buffer);
      if (!buffer) continue;
      b.buffer = buffer->buffer();
      b.srcAccessMask = resource_state.to_flush_access;
      b.dstAccessMask = barrier.access;
      b.srcStageMask = resource_state.pipeline_barrier_src_stages;
      b.dstStageMask = barrier.stages;
      b.size = buffer->size();
      b.offset = 0;
      state.buffer_barriers.push_back(b);
    }

    // if pending write or layout change, must invalidate caches
    if (resource_state.to_flush_access || layout_change) {
      for (auto& e : resource_state.invalidated_in_stage) e = 0;
    }
  }

  for (const auto& barrier : pass.flush_barriers) {
    auto& resource_state = resource_pipeline_states_[barrier.resource_idx];
    auto& res_dims = physical_resource_dims_[barrier.resource_idx];

    // set the image layout transition for the pass
    if (res_dims.is_image()) {
      const auto* image =
          vk2::get_device().get_image(physical_image_attachments_[barrier.resource_idx].handle);
      assert(image);
      if (!image) continue;
      resource_state.layout = barrier.layout;
    }
    // mark pending writes from this pass
    resource_state.to_flush_access = barrier.access;
    // set the src stages
    resource_state.pipeline_barrier_src_stages = barrier.stages;
  }
}

vk2::Image* RenderGraph::get_texture(uint32_t idx) { return get_texture(get_resource(idx)); }
vk2::Image* RenderGraph::get_texture(RenderResource* resource) {
  if (!resource) return nullptr;
  return vk2::get_device().get_image(physical_image_attachments_[resource->physical_idx].handle);
}

void RenderGraph::print_barrier(const VkImageMemoryBarrier2& barrier) const {
  LINFO(
      "oldLayout: {}, newLayout: {}\nsrcAccess: {}, dstAccess: {}\nsrcStage: {}, "
      "dstStage: {}",
      string_VkImageLayout(barrier.oldLayout), string_VkImageLayout(barrier.newLayout),
      string_VkAccessFlags2(barrier.srcAccessMask), string_VkAccessFlags2(barrier.dstAccessMask),
      string_VkPipelineStageFlags2(barrier.srcStageMask),
      string_VkPipelineStageFlags2(barrier.dstStageMask));
}

void RenderGraph::print_barrier(const VkBufferMemoryBarrier2& barrier) const {
  LINFO(
      "size: {}, \nsrcAccess: {}, dstAccess: {}\nsrcStage: {}, "
      "dstStage: {}",
      barrier.size, string_VkAccessFlags2(barrier.srcAccessMask),
      string_VkAccessFlags2(barrier.dstAccessMask),
      string_VkPipelineStageFlags2(barrier.srcStageMask),
      string_VkPipelineStageFlags2(barrier.dstStageMask));
}
}  // namespace gfx
