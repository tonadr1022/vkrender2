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
#include "Types.hpp"
#include "core/Logger.hpp"
#include "util/BitOps.hpp"
#include "vk2/Buffer.hpp"
#include "vk2/Device.hpp"
#include "vk2/Hash.hpp"
#include "vk2/Texture.hpp"
#include "vk2/VkTypes.hpp"

namespace gfx {
namespace {
void get_vk_stage_access(Access access, VkAccessFlags2& out_access,
                         VkPipelineStageFlags2& out_stages) {
  out_stages = {};
  out_access = {};
  if (access & Access::ComputeRead) {
    out_stages |= VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    out_access |= VK_ACCESS_2_SHADER_READ_BIT;
  }
  if (access & Access::ComputeSample) {
    out_stages |= VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    out_access |= VK_ACCESS_2_SHADER_READ_BIT;
  }
  if (access & Access::ComputeWrite) {
    out_stages |= VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    out_access |= VK_ACCESS_2_SHADER_WRITE_BIT;
  }
  if (access & Access::IndirectRead) {
    out_stages |= VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT;
    out_access |= VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT;
  }
  if (access & Access::VertexRead) {
    out_stages |= VK_PIPELINE_STAGE_2_VERTEX_INPUT_BIT;
    out_access |= VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT;
  }
  if (access & Access::IndexRead) {
    out_stages |= VK_PIPELINE_STAGE_2_INDEX_INPUT_BIT;
    out_access |= VK_ACCESS_2_INDEX_READ_BIT;
  }
  if (access & Access::DepthStencilWrite) {
    out_access |= VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    out_stages |=
        VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
  }
  if (access & Access::DepthStencilRead) {
    out_access |= VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT;
    out_stages |=
        VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
  }
  if (access & Access::ColorWrite) {
    out_stages |= VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
    out_access |= VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
  }
  if (access & Access::ColorRead) {
    out_stages |= VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
    out_access |= VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT;
  }
  if (access & Access::TransferWrite) {
    out_stages |= VK_PIPELINE_STAGE_2_TRANSFER_BIT | VK_PIPELINE_STAGE_2_COPY_BIT |
                  VK_PIPELINE_STAGE_2_CLEAR_BIT;
    out_access |= VK_ACCESS_2_TRANSFER_WRITE_BIT;
  }
  if (access & Access::FragmentRead) {
    out_stages |= VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
    out_access |= VK_ACCESS_2_SHADER_READ_BIT;
  }
  if (access & Access::TransferRead) {
    out_stages |= VK_PIPELINE_STAGE_2_TRANSFER_BIT | VK_PIPELINE_STAGE_2_COPY_BIT |
                  VK_PIPELINE_STAGE_2_CLEAR_BIT;
    out_access |= VK_ACCESS_2_TRANSFER_READ_BIT;
  }
}
constexpr auto read_flags = Access::ColorRead | Access::ComputeRead | Access::DepthStencilRead |
                            Access::VertexRead | Access::IndexRead | Access::IndirectRead |
                            Access::TransferRead | Access::FragmentRead;
constexpr auto write_flags =
    Access::ColorWrite | Access::ComputeWrite | Access::DepthStencilWrite | Access::TransferWrite;
bool is_read_access(Access access) { return access & read_flags; }
bool is_write_access(Access access) { return access & write_flags; }

VkImageUsageFlags get_image_usage(Access access) {
  VkImageUsageFlags usage{};
  if (access & (Access::ColorWrite | Access::ColorRead)) {
    usage |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
  }
  if (access & (Access::FragmentRead)) {
    usage |= VK_IMAGE_USAGE_SAMPLED_BIT;
  }
  if (access & (Access::ComputeSample)) {
    usage |= VK_IMAGE_USAGE_SAMPLED_BIT;
  }
  if (access & (Access::DepthStencilRead | Access::DepthStencilWrite)) {
    usage |= VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
  }
  // TODO: maybe SHADER_READ_ONLY_OPTIMAL?
  if (access & Access::ComputeRW) {
    usage |= VK_IMAGE_USAGE_STORAGE_BIT;
  }
  if (access & Access::TransferRead) {
    usage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
  }
  if (access & Access::TransferWrite) {
    usage |= VK_IMAGE_USAGE_TRANSFER_DST_BIT;
  }
  return usage;
}

}  // namespace

RenderGraphPass::UsageAndHandle RenderGraphPass::init_usage_and_handle(Access access,
                                                                       RGResourceHandle handle,
                                                                       RenderResource& res) {
  UsageAndHandle res_usage{.handle = handle, .access = access};
  get_vk_stage_access(access, res_usage.access_flags, res_usage.stages);
  if (is_read_access(access)) {
    resource_read_indices_.emplace_back(resources_.size());
    res.read_in_pass(idx_);
  }
  if (is_write_access(access)) {
    res.written_in_pass(idx_);
  }
  resources_.emplace_back(res_usage);
  return res_usage;
}

void RenderGraphPass::add(BufferHandle buf_handle, Access access) {
  auto resource_handle = graph_.get_or_add_buffer_resource(buf_handle);
  RenderResource& res = *graph_.get_resource(resource_handle);
  res.access = static_cast<Access>(res.access | access);

  auto* buf = get_device().get_buffer(buf_handle);
  assert(buf);
  if (!buf) {
    return;
  }
  res.buffer_info = {buf_handle, buf->size()};
  init_usage_and_handle(access, resource_handle, res);
}

RGResourceHandle RenderGraphPass::add(const std::string& name, const AttachmentInfo& info,
                                      Access access, const std::string&) {
  auto handle = graph_.get_or_add_texture_resource(name);
  RenderResource& res = *graph_.get_resource(handle);
  res.access = static_cast<Access>(res.access | access);
  res.info = info;

  if (name == graph_.get_backbuffer_img_name()) {
    swapchain_write_idx_ = resources_.size();
  }
  init_usage_and_handle(access, handle, res);
  return handle;
}

RGResourceHandle RenderGraphPass::add_image_access(const std::string& name, Access access) {
  if (!graph_.resource_to_idx_map_.contains(name)) {
    return {};
  }
  auto handle = graph_.get_or_add_texture_resource(name);
  RenderResource& res = *graph_.get_resource(handle);
  res.access = static_cast<Access>(res.access | access);

  if (name == graph_.get_backbuffer_img_name()) {
    swapchain_write_idx_ = resources_.size();
  }
  init_usage_and_handle(access, handle, res);
  return handle;
}

RenderGraphPass::RenderGraphPass(std::string name, RenderGraph& graph, uint32_t idx, Type type)
    : name_(std::move(name)), graph_(graph), idx_(idx), type_(type) {}

RenderGraph::RenderGraph(std::string name) : name_(std::move(name)) { log_ = false; }

RenderGraphPass& RenderGraph::add_pass(const std::string& name, RenderGraphPass::Type type) {
  auto idx = passes_.size();
  passes_.emplace_back(name, *this, idx, type);
  return passes_.back();
}

VoidResult RenderGraph::validate() { return VoidResult{}; }

VoidResult RenderGraph::bake() {
  ZoneScoped;
  swapchain_img_ = get_device().acquire_next_image();
  desc_ = get_device().get_swapchain_info();
  if (auto ok = validate(); !ok) {
    return ok;
  }

  if (log_) {
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
    pass_dependencies_.clear();
    pass_dependencies_.resize(passes_.size());
    for (uint32_t pass_i = 0; pass_i < passes_.size(); pass_i++) {
      auto& pass = passes_[pass_i];
      for (const auto& usage : pass.get_resources()) {
        if (get_resource(usage.handle)->name == backbuffer_img_) {
          // TODO: move this to validation phase
          // if (!(usage.access_flags & VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT)) {
          //   return std::unexpected("backbuffer output is not of usage ColorOutput");
          // }
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

    if (log_) {
      LINFO("pass order: ");
      for (const auto& s : pass_stack_) {
        LINFO("{}", passes_[s].get_name());
      }
    }
  }

  build_physical_resource_reqs();

  if (log_) {
    for (auto& res : resources_) {
      if (res.get_type() == RenderResource::Type::Texture) {
        LINFO("{} {}", res.name, res.physical_idx);
        if (res.physical_idx < physical_resource_dims_.size()) {
          auto& dims = physical_resource_dims_[res.physical_idx];
          // LINFO("{} {}", res.physical_idx, string_VkImageUsageFlags(dims.image_usage_flags));
        }
      }
    }
  }

  {
    ZoneScopedN("build physical passes");
    physical_passes_.resize(passes_.size());
    for (auto& p : physical_passes_) {
      p.reset();
    }
    for (const auto& pass_i : pass_stack_) {
      const auto& pass = passes_[pass_i];
      PhysicalPass& phys_pass = physical_passes_[pass_i];
      phys_pass.name = pass.get_name();

      for (const auto& output : pass.get_resources()) {
        auto* res = get_resource(output.handle);
        if (res->physical_idx != RenderResource::unused) {
          if (output.access & Access::ColorWrite) {
            phys_pass.physical_color_attachments.emplace_back(res->physical_idx);
          } else if (output.access & Access::DepthStencilWrite) {
            phys_pass.physical_depth_stencil = res->physical_idx;
          }
        }
      }
    }
  }

  if (log_) {
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
    ZoneScoped;
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

    resource_states_.clear();
    resource_states_.reserve(physical_resource_dims_.size());
    for (auto pass_i : pass_stack_) {
      // auto& pass = passes_[pass_i];
      auto& phys_pass = physical_passes_[pass_i];
      resource_states_.clear();
      resource_states_.resize(physical_resource_dims_.size());
      for (auto& invalidate : phys_pass.invalidate_barriers) {
        auto& res = resource_states_[invalidate.resource_idx];
        res.invalidated_accesses |= invalidate.access;
        res.invalidated_stages |= invalidate.stages;
        res.initial_layout = invalidate.layout;
        res.final_layout = invalidate.layout;
        // // pending flushes have been invalidated already
        // res.flushed_stages = 0;
        // res.flushed_accesses = 0;
      }

      for (auto& flush : phys_pass.flush_barriers) {
        auto& res = resource_states_[flush.resource_idx];
        res.flushed_stages |= flush.stages;
        res.flushed_accesses |= flush.access;

        res.final_layout = flush.layout;

        if (res.initial_layout == VK_IMAGE_LAYOUT_UNDEFINED) {
          res.initial_layout = flush.layout;
          res.invalidated_stages |= flush_stage_to_invalidate(flush.stages);
          res.invalidated_accesses |= flush_access_to_invalidate(flush.access);
          phys_pass.discard_resources.emplace_back(flush.resource_idx);
        }
      }

      for (u32 resource_i = 0; resource_i < resource_states_.size(); resource_i++) {
        auto& res = resource_states_[resource_i];
        if (res.final_layout == VK_IMAGE_LAYOUT_UNDEFINED &&
            res.initial_layout == VK_IMAGE_LAYOUT_UNDEFINED) {
          continue;
        }

        // check if an invalidate already exists
        auto existing_invalidate_it = std::ranges::find_if(
            phys_pass.invalidate_barriers,
            [resource_i](const Barrier& barrier) { return barrier.resource_idx == resource_i; });
        bool existing_flush{}, existing_invalidate{};
        if (existing_invalidate_it != phys_pass.invalidate_barriers.end()) {
          existing_invalidate_it->access |= res.invalidated_accesses;
          existing_invalidate_it->stages |= res.invalidated_stages;
          existing_invalidate_it->layout = res.initial_layout;
          existing_invalidate = true;
        }
        auto existing_flush_it = std::ranges::find_if(
            phys_pass.flush_barriers,
            [resource_i](const Barrier& barrier) { return barrier.resource_idx == resource_i; });
        if (existing_flush_it != phys_pass.flush_barriers.end()) {
          existing_flush_it->access |= res.flushed_accesses;
          existing_flush_it->stages |= res.flushed_stages;
          existing_flush_it->layout = res.final_layout;
          existing_flush = true;
        }

        assert(res.final_layout != VK_IMAGE_LAYOUT_UNDEFINED);
        if (!existing_invalidate) {
          phys_pass.invalidate_barriers.emplace_back(Barrier{.resource_idx = resource_i,
                                                             .layout = res.initial_layout,
                                                             .access = res.invalidated_accesses,
                                                             .stages = res.invalidated_stages});
        }
        if (res.flushed_accesses && !existing_flush) {
          phys_pass.flush_barriers.emplace_back(Barrier{.resource_idx = resource_i,
                                                        .layout = res.final_layout,
                                                        .access = res.flushed_accesses,
                                                        .stages = res.flushed_stages});
        } else if (res.invalidated_accesses && !existing_flush) {
          // if pass read something that needs protection before writing, need flush with no
          // access sets the last pass which the resource was used as a stage
          phys_pass.flush_barriers.emplace_back(Barrier{.resource_idx = resource_i,
                                                        .layout = res.final_layout,
                                                        .access = 0,
                                                        .stages = res.invalidated_stages});
        }
      }
    }

    if (log_) {
      for (auto& phys_pass : physical_passes_) {
        for (auto& flush : phys_pass.flush_barriers) {
          LINFO("flush barrier {} {} {}", string_VkAccessFlags2(flush.access),
                string_VkPipelineStageFlags2(flush.stages), phys_pass.name);
        }
        for (auto& invalidate : phys_pass.invalidate_barriers) {
          LINFO("invalidate barrier {} {} {}", string_VkAccessFlags2(invalidate.access),
                string_VkPipelineStageFlags2(invalidate.stages), phys_pass.name);
        }
      }
    }
  }

  return {};
}

void RenderGraph::execute(CmdEncoder& cmd) {
  ZoneScoped;
  if (swapchain_img_ == nullptr || desc_.dims.x == 0 || desc_.dims.y == 0) {
    LERROR("invalid swapchain info");
    return;
  }

  {
    pass_submission_state_.resize(physical_passes_.size());
    for (auto& p : pass_submission_state_) {
      p.reset();
    }
    ZoneScopedN("setup barriers");
    for (auto pass_i : pass_stack_) {
      physical_pass_setup_barriers(pass_i);
    }
  }

  {
    ZoneScopedN("Record commands");
    for (u32 pass_i : pass_stack_) {
      ZoneScopedN("Record command");
      auto& pass = passes_[pass_i];
      auto& submission_state = pass_submission_state_[pass_i];

      VkDependencyInfo info{.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
      info.bufferMemoryBarrierCount = submission_state.buffer_barriers.size();
      info.pBufferMemoryBarriers = submission_state.buffer_barriers.data();
      info.imageMemoryBarrierCount = submission_state.image_barriers.size();
      info.pImageMemoryBarriers = submission_state.image_barriers.data();

      if (log_) {
        LINFO("barriers: {}", pass.get_name());
        LINFO("buffers");
        for (auto& b : submission_state.buffer_barriers) {
          print_barrier(b);
        }
        LINFO("images");
        for (auto& b : submission_state.image_barriers) {
          u32 idx = UINT32_MAX;
          for (u32 k = 0; k < physical_image_attachments_.size(); k++) {
            auto& img = physical_image_attachments_[k];
            if (!img.is_valid()) continue;
            auto* i = get_device().get_image(img);
            if (i && i->image() == b.image) {
              idx = k;
              break;
            }
          }
          for (auto& resource : resources_) {
            if (resource.get_type() == RenderResource::Type::Texture &&
                resource.physical_idx == idx) {
              std::print("resource barrier: {}\t", resource.name);
            }
          }
          print_barrier(b);
        }
        LINFO("");
      }

      vkCmdPipelineBarrier2KHR(cmd.cmd(), &info);

      // LINFO("pre executed");
      pass.execute_(cmd);
      // LINFO("executed");
    }
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
              .image = swapchain_img_,
              .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
          },
      };
      VkDependencyInfo info{.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                            .imageMemoryBarrierCount = COUNTOF(img_barriers),
                            .pImageMemoryBarriers = img_barriers};
      vkCmdPipelineBarrier2KHR(cmd.cmd(), &info);
    }

    // only write each image to the swapchain once
    util::fixed_vector<u32, 20> swapchain_write_indices;
    for (u32 pass_i : swapchain_writer_passes_) {
      auto& pass = passes_[pass_i];
      if (const auto* output = pass.get_swapchain_write_usage(); output != nullptr) {
        auto* resource = get_resource(output->handle);
        auto* tex = get_texture(output->handle);

        if (std::ranges::find(swapchain_write_indices, output->handle.idx) !=
            swapchain_write_indices.end()) {
          continue;
        }

        swapchain_write_indices.push_back(output->handle.idx);

        assert(resource && tex);
        if (!resource || !tex) {
          continue;
        }
        auto* pstate = get_resource_pipeline_state(resource->physical_idx);
        assert(pstate);
        auto& state = *pstate;
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
        VkDependencyInfo info{.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                              .imageMemoryBarrierCount = COUNTOF(img_barriers),
                              .pImageMemoryBarriers = img_barriers};
        vkCmdPipelineBarrier2KHR(cmd.cmd(), &info);

        VkExtent3D dims{glm::min(tex->create_info().extent.width, desc_.dims.x),
                        glm::min(tex->create_info().extent.height, desc_.dims.y), 1};
        blit_img(cmd.cmd(), tex->image(), swapchain_img_, dims, VK_IMAGE_ASPECT_COLOR_BIT);

        // write after read barrier
        state.layout = img_barriers[0].newLayout;
        state.to_flush_access = 0;
        for (auto& e : state.invalidated_in_stage) e = 0;
        state.invalidated_in_stage[util::trailing_zeros(VK_PIPELINE_STAGE_2_BLIT_BIT)] =
            VK_ACCESS_2_TRANSFER_READ_BIT;
        state.pipeline_barrier_src_stages = VK_PIPELINE_STAGE_2_BLIT_BIT;
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
              .image = swapchain_img_,
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
  if (stack_size > passes_.size() * 100) {
    LERROR("cycle");
    return std::unexpected("cycle detected");
  }

  auto& pass = passes_[pass_i];
  for (const uint32_t& read_i : pass.resource_read_indices_) {
    auto& resource_read = pass.resources_[read_i];
    std::span<const uint16_t> passes_writing_to_resource{};
    auto* input = get_resource(resource_read.handle);
    assert(input);
    passes_writing_to_resource = input->get_written_passes();

    // find outputs that write to this input
    if (passes_writing_to_resource.empty()) {
      LINFO("resource: {}", resources_[read_i].name);
      return std::unexpected("no pass exists which writes to resource");
    }

    if (stack_size > passes_.size() * 100) {
      LERROR("cycle");
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

RGResourceHandle RenderGraph::get_or_add_buffer_resource(BufferHandle handle) {
  auto it = buffer_to_idx_map_.find(handle);
  if (it != buffer_to_idx_map_.end()) {
    return it->second;
  }
  uint32_t idx = resources_.size();
  RGResourceHandle out_handle{idx, RenderResource::Type::Buffer};
  buffer_to_idx_map_.emplace(handle, out_handle);
  resources_.emplace_back(RenderResource::Type::Buffer, idx);
  return out_handle;
}

RGResourceHandle RenderGraph::get_or_add_texture_resource(const std::string& name) {
  auto it = resource_to_idx_map_.find(name);
  if (it != resource_to_idx_map_.end()) {
    return it->second;
  }

  uint32_t idx = resources_.size();
  RGResourceHandle handle{idx, RenderResource::Type::Texture};
  resource_to_idx_map_.emplace(name, handle);
  resources_.emplace_back(RenderResource::Type::Texture, idx);
  resources_[idx].name = name;
  return handle;
}

RenderResource* RenderGraph::get_resource(RGResourceHandle handle) {
  assert(handle.idx < resources_.size());
  if (handle.idx >= resources_.size()) {
    return nullptr;
  }
  return &resources_[handle.idx];
}

ResourceDimensions RenderGraph::get_resource_dims(const RenderResource& resource) const {
  ResourceDimensions dims{};
  if (resource.get_type() == RenderResource::Type::Buffer) {
    dims.buffer_info = resource.buffer_info;
  } else {
    assert(desc_.dims.x > 0 && desc_.dims.y > 0);
    dims = {.format = resource.info.format,
            .size_class = resource.info.size_class,
            .depth = 1,
            .layers = resource.info.layers,
            .levels = resource.info.levels,
            .samples = 1,
            .access_usage = resource.access};
    if (resource.info.size_class == SizeClass::SwapchainRelative) {
      dims.width = desc_.dims.x;
      dims.height = desc_.dims.y;
    } else if (resource.info.size_class == SizeClass::Absolute) {
      dims.width = (uint32_t)resource.info.dims.x;
      dims.height = (uint32_t)resource.info.dims.y;
    } else {  // InputRelative
      assert(0 && "nope");
    }
  }
  return dims;
}

void RenderGraph::build_physical_resource_reqs() {
  ZoneScoped;
  physical_resource_dims_.clear();
  physical_resource_dims_.reserve(resources_.size());
  for (uint32_t pass_i : pass_stack_) {
    auto& pass = passes_[pass_i];
    for (const auto& r : pass.get_resources()) {
      get_resource(r.handle)->physical_idx = RenderResource::unused;
    }
  }
  // build physical resources
  for (uint32_t pass_i : pass_stack_) {
    auto& pass = passes_[pass_i];
    for (const auto& resource_usage : pass.get_resources()) {
      auto* res = get_resource(resource_usage.handle);
      if (res->physical_idx == RenderResource::unused) {
        assert(res);
        res->physical_idx = physical_resource_dims_.size();
        if (res->name == backbuffer_img_) {
          res->access = static_cast<Access>(res->access | Access::TransferRead);
        }
        physical_resource_dims_.emplace_back(get_resource_dims(*res));
      } else {
        if (physical_resource_dims_[res->physical_idx].is_image()) {
          physical_resource_dims_[res->physical_idx].access_usage = static_cast<Access>(
              physical_resource_dims_[res->physical_idx].access_usage | res->access);
        }
      }
    }
  }
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
      auto& res = *get_resource(output.handle);
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

  for (auto& [key, val] : img_cache_used_) {
    img_cache_.emplace(key, std::move(val));
  }
  img_cache_used_.clear();

  for (size_t i = 0; i < physical_resource_dims_.size(); i++) {
    auto& dims = physical_resource_dims_[i];
    if (dims.is_image()) {
      ImageCreateInfo info{};
      info.extent = VkExtent3D{dims.width, dims.height, dims.depth};
      info.array_layers = dims.layers;
      info.mip_levels = dims.levels;
      info.samples = (VkSampleCountFlagBits)(1 << (dims.samples - 1));
      info.format = vk2::to_vkformat(dims.format);
      info.override_usage_flags |= get_image_usage(dims.access_usage);
      if (dims.depth == 1) {
        info.view_type = VK_IMAGE_VIEW_TYPE_2D;
        if (info.array_layers > 1) {
          info.view_type = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
        }
      } else {
        info.view_type = VK_IMAGE_VIEW_TYPE_3D;
      }
      assert(i < physical_image_attachments_.size());
      {
        auto it = img_cache_.find(dims);
        bool need_new_img{true};
        if (it != img_cache_.end()) {
          bool valid_extent{false};
          auto* img = get_device().get_image(it->second);
          if (img) {
            const auto& cinfo = img->create_info();
            if (dims.size_class == SizeClass::SwapchainRelative) {
              valid_extent =
                  cinfo.extent.width == desc_.dims.x && cinfo.extent.height == desc_.dims.y;
            } else {
              valid_extent = cinfo.extent.width == dims.width &&
                             cinfo.extent.height == dims.height && cinfo.extent.depth == dims.depth;
            }
            if (valid_extent && cinfo.array_layers == dims.layers &&
                cinfo.mip_levels == dims.levels && cinfo.samples == dims.samples &&
                cinfo.format == vk2::to_vkformat(dims.format) &&
                cinfo.override_usage_flags == get_image_usage(dims.access_usage)) {
              physical_image_attachments_[i] = it->second.handle;
              need_new_img = false;
            } else {
              LINFO("need new image: {} {} {}\ncinfo : {} {}", i, dims.width, dims.height,
                    cinfo.extent.width, cinfo.extent.height);
            }
          }
          if (!need_new_img) {
            img_cache_used_.emplace_back(it->first, std::move(it->second));
          }
          img_cache_.erase(it);
        }
        if (need_new_img) {
          image_pipeline_states_.erase(physical_image_attachments_[i]);
          LINFO("making new image: {}", i);
          img_cache_used_.emplace_back(dims, get_device().create_image_holder(info));
          // if (!inserted) {
          //   it->second = vk2::get_device().create_image_holder(info);
          // }
          physical_image_attachments_[i] = img_cache_used_.back().second.handle;
        }
      }
    } else {
      physical_buffers_[i] = dims.buffer_info.handle;
    }
  }
}
namespace {
VkImageLayout get_image_layout(Access access) {
  if (access & Access::ColorRW) {
    return VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
  }
  if (access & Access::FragmentRead) {
    return VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
  }
  if (access & Access::DepthStencilRW) {
    return VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL_KHR;
  }
  if (access & (Access::ComputeRW)) {
    return VK_IMAGE_LAYOUT_GENERAL;
  }
  return VK_IMAGE_LAYOUT_UNDEFINED;
}
}  // namespace

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

  // TODO: refactor
  for (const auto& pass_i : pass_stack_) {
    auto& pass = passes_[pass_i];
    auto& phys_pass = physical_passes_[pass_i];
    auto get_invalidate_access = [&get_access, &phys_pass](uint32_t idx) -> Barrier& {
      return get_access(phys_pass.invalidate_barriers, idx);
    };
    auto get_flush_access = [&get_access, &phys_pass](uint32_t idx) -> Barrier& {
      return get_access(phys_pass.flush_barriers, idx);
    };

    for (const auto& pass_resource_usage : pass.get_resources()) {
      auto resource_physical_idx = get_resource(pass_resource_usage.handle)->physical_idx;
      if (resource_physical_idx == RenderResource::unused) continue;
      if (is_read_access(pass_resource_usage.access)) {
        auto& barrier = get_invalidate_access(resource_physical_idx);
        barrier.layout = get_image_layout(pass_resource_usage.access);
        barrier.access |= pass_resource_usage.access_flags;
        barrier.stages |= pass_resource_usage.stages;
      }
      if (is_write_access(pass_resource_usage.access)) {
        auto& barrier = get_flush_access(resource_physical_idx);
        barrier.layout = get_image_layout(pass_resource_usage.access);
        barrier.access |= pass_resource_usage.access_flags;
        barrier.stages |= pass_resource_usage.stages;
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
    assert(barrier.resource_idx < physical_resource_dims_.size());
    assert(barrier.resource_idx < physical_image_attachments_.size());
    assert(barrier.resource_idx < physical_buffers_.size());
    const auto& phys_dims = physical_resource_dims_[barrier.resource_idx];
    auto* pstate = get_resource_pipeline_state(barrier.resource_idx);
    assert(pstate);
    auto& resource_state = *pstate;

    if (phys_dims.is_image()) {
      // queue families ignored
      const auto* image = get_device().get_image(physical_image_attachments_[barrier.resource_idx]);
      assert(image);
      if (!image) {
        LERROR("no image");
        exit(1);
      }
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
        if (resource_state.pipeline_barrier_src_stages) {
          b.srcStageMask = resource_state.pipeline_barrier_src_stages;
        } else {
          b.srcStageMask = VK_PIPELINE_STAGE_NONE;
          b.srcAccessMask = 0;
          // assert(b.oldLayout == VK_IMAGE_LAYOUT_UNDEFINED);
        }
        state.image_barriers.push_back(b);
      }

      resource_state.layout = b.newLayout;

    } else {
      VkBufferMemoryBarrier2 b{.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2};

      auto* pstate = get_resource_pipeline_state(barrier.resource_idx);
      assert(pstate);
      auto& resource_state = *pstate;
      const auto* buffer = get_device().get_buffer(physical_buffers_[barrier.resource_idx]);
      if (!buffer) {
        LERROR("no buffer");
        exit(1);
      }
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
    auto& resource_state = *get_resource_pipeline_state(barrier.resource_idx);
    auto& res_dims = physical_resource_dims_[barrier.resource_idx];

    // set the image layout transition for the pass
    if (res_dims.is_image()) {
      const auto* image = get_device().get_image(physical_image_attachments_[barrier.resource_idx]);
      assert(image);
      if (!image) continue;
      assert(resource_state.layout == barrier.layout);
      resource_state.layout = barrier.layout;
    }
    // mark pending writes from this pass
    resource_state.to_flush_access = barrier.access;
    // set the src stages
    resource_state.pipeline_barrier_src_stages = barrier.stages;
  }
}

ImageHandle RenderGraph::get_texture_handle(RenderResource* resource) {
  if (!resource) return {};
  return physical_image_attachments_[resource->physical_idx];
}

ImageHandle RenderGraph::get_texture_handle(RGResourceHandle resource) {
  return get_texture_handle(get_resource(resource));
}
Image* RenderGraph::get_texture(RGResourceHandle handle) {
  return get_texture(get_resource(handle));
}

Image* RenderGraph::get_texture(RenderResource* resource) {
  if (!resource) return nullptr;
  return get_device().get_image(physical_image_attachments_[resource->physical_idx]);
}

void RenderGraph::print_barrier(const VkImageMemoryBarrier2& barrier) const {
  LINFO(
      "oldLayout: {}, newLayout: {}, aspect {}\nsrcAccess: {}, dstAccess: {}\nsrcStage: {}, "
      "dstStage: {}",
      string_VkImageLayout(barrier.oldLayout), string_VkImageLayout(barrier.newLayout),
      string_VkImageAspectFlags(barrier.subresourceRange.aspectMask),
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

const RenderGraphPass::UsageAndHandle* RenderGraphPass::get_swapchain_write_usage() const {
  if (swapchain_write_idx_ == RenderResource::unused) {
    return nullptr;
  }
  return &resources_[swapchain_write_idx_];
}

void RenderGraph::reset() {
  passes_.clear();
  buffer_to_idx_map_.clear();
  physical_resource_dims_.clear();
  resources_.clear();
  resource_to_idx_map_.clear();
  pass_dependencies_.clear();
  swapchain_writer_passes_.clear();
  dup_prune_set_.clear();
  physical_image_attachments_.clear();
  physical_buffers_.clear();
}

std::size_t ResourceDimensionsHasher::operator()(const ResourceDimensions& dims) const {
  if (dims.size_class == SizeClass::SwapchainRelative) {
    auto h = std::make_tuple(dims.format, dims.levels, dims.layers, dims.access_usage,
                             dims.size_class, dims.samples);
    return vk2::detail::hashing::hash<decltype(h)>{}(h);
  }
  auto h = std::make_tuple(dims.width, dims.height, dims.format, dims.levels, dims.layers,
                           dims.access_usage, dims.size_class, dims.depth, dims.samples);
  return vk2::detail::hashing::hash<decltype(h)>{}(h);
}

RenderGraph::ResourceState* RenderGraph::get_resource_pipeline_state(u32 idx) {
  // TODO: remove stale pipeline states
  // TODO: imgui menu

  assert(idx < physical_resource_dims_.size());
  if (idx >= physical_resource_dims_.size()) {
    return nullptr;
  }
  auto& dims = physical_resource_dims_[idx];
  if (dims.is_image()) {
    return &image_pipeline_states_[physical_image_attachments_[idx]];
  }
  return &buffer_pipeline_states_[physical_buffers_[idx]];
}

}  // namespace gfx
