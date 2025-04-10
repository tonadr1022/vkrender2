#include "RenderGraph.hpp"

#include <vulkan/vk_enum_string_helper.h>
#include <vulkan/vulkan_core.h>

#include <algorithm>
#include <cstdint>
#include <expected>
#include <fstream>
#include <tracy/Tracy.hpp>
#include <utility>

#include "Logger.hpp"
#include "vk2/Device.hpp"
#include "vk2/Texture.hpp"
#include "vk2/VkTypes.hpp"

namespace gfx {

RenderResourceHandle RenderGraphPass::add_color_output(const std::string& name,
                                                       const AttachmentInfo& info,
                                                       const std::string& input) {
  // TODO: queue
  uint32_t handle = graph_.get_or_add_texture_resource(name);
  RenderTextureResource* res = graph_.get_texture_resource(handle);
  res->image_usage |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
  res->written_in_pass(idx_);
  res->info = info;
  if (input.size()) {
    assert(0 && "unimplemented");
  }
  UsageAndHandle usage{.idx = handle, .usage = ResourceUsage::ColorOutput};
  resource_outputs_.emplace_back(usage);
  return handle;
}

RenderResourceHandle RenderGraphPass::add_storage_image_input(const std::string& name) {
  uint32_t handle = graph_.get_or_add_texture_resource(name);
  RenderTextureResource* res = graph_.get_texture_resource(handle);
  res->read_in_pass(idx_);
  res->image_usage |= VK_IMAGE_USAGE_STORAGE_BIT;
  UsageAndHandle usage{.idx = handle, .usage = ResourceUsage::StorageImageInput};
  usage.access = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
  usage.stages = type_ == Type::Graphics ? VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT
                                         : VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
  resource_inputs_.emplace_back(usage);

  return handle;
}
RenderResourceHandle RenderGraphPass::add_texture_input(const std::string& name) {
  uint32_t handle = graph_.get_or_add_texture_resource(name);
  RenderTextureResource* res = graph_.get_texture_resource(handle);
  res->read_in_pass(idx_);
  res->image_usage |= VK_IMAGE_USAGE_SAMPLED_BIT;
  UsageAndHandle usage{.idx = handle, .usage = ResourceUsage::TextureInput};
  usage.access = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
  usage.stages = type_ == Type::Graphics ? VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT
                                         : VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;

  resource_inputs_.emplace_back(usage);

  return handle;
}

RenderResourceHandle RenderGraphPass::set_depth_stencil_input(const std::string& name) {
  uint32_t handle = graph_.get_or_add_texture_resource(name);
  RenderTextureResource* res = graph_.get_texture_resource(handle);
  res->read_in_pass(idx_);
  res->image_usage |= VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
  UsageAndHandle usage{.idx = handle, .usage = ResourceUsage::DepthStencilInput};
  resource_inputs_.emplace_back(usage);
  return handle;
}

RenderResourceHandle RenderGraphPass::set_depth_stencil_output(const std::string& name,
                                                               const AttachmentInfo& info) {
  uint32_t handle = graph_.get_or_add_texture_resource(name);
  RenderTextureResource* res = graph_.get_texture_resource(handle);
  res->written_in_pass(idx_);
  res->image_usage |= VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
  res->info = info;
  UsageAndHandle usage{.idx = handle, .usage = ResourceUsage::DepthStencilOutput};
  resource_outputs_.emplace_back(usage);
  return handle;
}

RenderGraphPass::RenderGraphPass(std::string name, ExecuteFn fn, RenderGraph& graph, uint32_t idx,
                                 Type type)
    : name_(std::move(name)), execute_(std::move(fn)), graph_(graph), idx_(idx), type_(type) {}

RenderGraph::RenderGraph(std::string name) : name_(std::move(name)) {}

RenderGraphPass& RenderGraph::add_pass(const std::string& name, ExecuteFn execute,
                                       RenderGraphPass::Type type) {
  auto idx = passes_.size();
  passes_.emplace_back(name, std::move(execute), *this, idx, type);
  return passes_.back();
}

VoidResult RenderGraph::bake() {
  ZoneScoped;
  bool log = true;
  // validate
  // go through each input and make sure it's an output of a previous pass if it needs to read from
  // it
  // for (auto& pass : passes_) {
  //   // for (auto& tex_resource : pass.)
  // }

  if (log) {
    for (const auto& resource : resources_) {
      for (const auto& b : resource->get_read_passes()) {
        LINFO("{}: read in {}", resource->name, passes_[b].get_name());
      }
      for (const auto& b : resource->get_written_passes()) {
        LINFO("{}: written in {}", resource->name, passes_[b].get_name());
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
      for (const auto& usage : pass.get_resource_outputs()) {
        LINFO("usage: {} {}", usage.idx, resources_[usage.idx]->name);
        if (resources_[usage.idx]->name == backbuffer_img_) {
          // TODO: move this to validation phase
          if (usage.usage != ResourceUsage::ColorOutput) {
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
      if (res->get_type() == ResourceType::Texture) {
        auto* tex = (RenderTextureResource*)res.get();
        assert(tex->physical_idx != RenderResource::unused);
        LINFO("{} {}", tex->name, tex->physical_idx);
        assert(tex->physical_idx < physical_resource_dims_.size());
        auto& dims = physical_resource_dims_[tex->physical_idx];
        LINFO("{} {}", tex->physical_idx, string_VkImageUsageFlags(dims.image_usage_flags));
      }
    }
  }

  clear_physical_passes();

  {
    ZoneScopedN("build physical passes");
    physical_passes_.reserve(passes_.size());
    for (const auto& pass_i : pass_stack_) {
      const auto& pass = passes_[pass_i];
      PhysicalPass phys_pass = get_physical_pass();
      // TODO: don't allocate string here
      phys_pass.name = pass.get_name();

      for (const auto& output : pass.get_resource_outputs()) {
        if (is_texture_usage(output.usage)) {
          auto* tex = get_texture_resource(output.idx);
          if (output.usage == ResourceUsage::ColorOutput) {
            phys_pass.physical_color_attachments.emplace_back(tex->physical_idx);
          } else if (output.usage == ResourceUsage::DepthStencilOutput) {
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

  {
    ZoneScopedN("build barriers");
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

      auto process_resources = [&](const std::vector<RenderGraphPass::UsageAndHandle>& resources) {
        for (const auto& input : resources) {
          auto* res = resources_[input.idx].get();
          if (input.usage == ResourceUsage::DepthStencilOutput) {
            auto& barrier = get_flush_access(res->physical_idx);
            barrier.stages |=
                VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_EXT;
            barrier.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
            barrier.access |= VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
          } else if (input.usage == ResourceUsage::ColorOutput) {
            auto& barrier = get_flush_access(res->physical_idx);
            barrier.stages |= VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
            barrier.access |= VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
            barrier.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
          } else if (input.usage == ResourceUsage::TextureInput ||
                     input.usage == ResourceUsage::StorageImageInput) {
            auto& barrier = get_invalidate_access(res->physical_idx);
            barrier.stages |= input.stages;
            barrier.layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            barrier.access = input.access;
            // barrier.stages |= VK_PIPELINE_STAGE_2_
          } else {
            assert(0);
          }
        }
      };

      process_resources(pass.get_resource_inputs());
      process_resources(pass.get_resource_outputs());
    }

    for (const auto& pass_i : pass_stack_) {
      auto& pass = passes_[pass_i];
      auto& phys_pass = physical_passes_[pass_i];
      LINFO("\npass: {}", pass.get_name());
      LINFO("\nflush barriers (writes)");
      for (const auto& barrier : phys_pass.flush_barriers) {
        LINFO("access: {}, layout: {}, idx: {}, stages: {}", string_VkAccessFlags2(barrier.access),
              string_VkImageLayout(barrier.layout), barrier.resource_idx,
              string_VkPipelineStageFlags2(barrier.stages));
      }
    }
    // TODO: aliasing to reuse images
  }

  return {};
}

void RenderGraph::execute() {
  // for (auto pass_i : pass_stack_) {
  //   auto& phys_pass = physical_passes_[pass_i];
  //   util::fixed_vector<VkImageMemoryBarrier2, 8> img_barriers;
  //   for (auto& b : phys_pass.invalidate_barriers) {
  //     // auto& resource = physical_res
  //     VkImageMemoryBarrier2 img_barrier{.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
  //     // img_barrier.
  //   }
  //
  //   // emit pre pass barriers
  // }
}

// TODO: not recursive
VoidResult RenderGraph::traverse_dependencies_recursive(uint32_t pass_i, uint32_t stack_size) {
  if (stack_size > passes_.size()) {
    return std::unexpected("cycle detected");
  }

  auto& pass = passes_[pass_i];
  for (const auto& resource_read : pass.get_resource_inputs()) {
    std::span<const uint16_t> passes_writing_to_resource{};
    if (is_texture_usage(resource_read.usage)) {
      auto* input = get_texture_resource(resource_read.idx);
      assert(input);
      passes_writing_to_resource = input->get_written_passes();
    }

    for (auto r : passes_writing_to_resource) {
      LINFO("pass {} writes to {}", passes_[pass.get_idx()].get_name(), resources_[r]->name);
    }

    // find outputs that write to this input
    if (passes_writing_to_resource.empty()) {
      return std::unexpected("no pass exists which writes to resource");
    }
    if (stack_size > passes_.size()) {
      return std::unexpected("cycle detected");
    }
    stack_size++;
    for (uint32_t pass_writing_to_resource : passes_writing_to_resource) {
      if (pass_writing_to_resource == pass.get_idx()) {
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

uint32_t RenderGraph::get_or_add_texture_resource(const std::string& name) {
  auto it = resource_to_idx_map_.find(name);
  if (it != resource_to_idx_map_.end()) {
    uint32_t idx = it->second;
    assert(idx < resources_.size());
    assert(resources_[idx]->get_type() == ResourceType::Texture);
    assert(resources_[idx]->name == name);
    return it->second;
  }

  uint32_t idx = resources_.size();
  resource_to_idx_map_.emplace(name, idx);
  resources_.emplace_back(std::make_unique<RenderTextureResource>(idx));
  resources_[idx]->name = name;
  return idx;
}

RenderTextureResource* RenderGraph::get_texture_resource(uint32_t idx) {
  assert(idx < resources_.size());
  assert(resources_[idx]->get_type() == ResourceType::Texture);
  return (RenderTextureResource*)resources_[idx].get();
}

bool is_texture_usage(ResourceUsage usage) {
  switch (usage) {
    case ResourceUsage::ColorOutput:
    case ResourceUsage::StorageImageInput:
    case ResourceUsage::ColorInput:
    case ResourceUsage::TextureInput:
    case ResourceUsage::DepthStencilOutput:
    case ResourceUsage::DepthStencilInput:
      return true;
    case gfx::ResourceUsage::None:
      return false;
  }
  return false;
}

ResourceDimensions RenderGraph::get_resource_dims(const RenderTextureResource& resource) const {
  assert(swapchain_info_.width > 0 && swapchain_info_.height > 0);
  ResourceDimensions dims{.format = resource.info.format,
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
    for (const auto& resource_usage : pass.get_resource_inputs()) {
      if (is_texture_usage(resource_usage.usage)) {
        auto* tex = get_texture_resource(resource_usage.idx);
        if (tex->physical_idx == RenderResource::unused) {
          assert(tex);
          tex->physical_idx = physical_resource_dims_.size();
          physical_resource_dims_.emplace_back(get_resource_dims(*tex));
        } else {
          physical_resource_dims_[tex->physical_idx].image_usage_flags |= tex->image_usage;
        }
      } else {  // buffer
        assert(0);
      }
    }
    // TODO: lambda
    for (const auto& resource_usage : pass.get_resource_outputs()) {
      if (is_texture_usage(resource_usage.usage)) {
        auto* tex = get_texture_resource(resource_usage.idx);
        if (tex->physical_idx == RenderResource::unused) {
          assert(tex);
          tex->physical_idx = physical_resource_dims_.size();
          physical_resource_dims_.emplace_back(get_resource_dims(*tex));
        } else {
          physical_resource_dims_[tex->physical_idx].image_usage_flags |= tex->image_usage;
        }
      } else {  // buffer
        assert(0);
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
  for (auto p : s) {
    auto& pass = passes_[p];
    for (const auto& writer : pass_dependencies_[p]) {
      for (const auto& output : passes_[writer].get_resource_outputs()) {
        auto* output_res = resources_[output.idx].get();
        assert(output_res);
        for (const auto& input : pass.get_resource_inputs()) {
          if (output_res->name == resources_[input.idx]->name) {
            print_link(passes_[writer].get_name(), pass.get_name(), output_res->name);
          }
        }
      }
    }
  }

  for (uint32_t writer : swapchain_writer_passes_) {
    for (const auto& output : passes_[writer].get_resource_outputs()) {
      auto* res = resources_[output.idx].get();
      assert(res);
      if (res->name == backbuffer_img_) {
        print_link(passes_[writer].get_name(), "swapchain", res->name);
      }
    }
  }
  ofs << "}\n";

  return {};
}

void RenderGraph::setup_attachments() {
  // only make attachments if they don't match prev frame
  physical_image_attachments_.resize(physical_resource_dims_.size());
  for (size_t i = 0; i < physical_resource_dims_.size(); i++) {
    auto& dims = physical_resource_dims_[i];
    if (!dims.buffer_info.size) {
      auto* img = vk2::get_device().get_image(physical_image_attachments_[i]);
      if (img) {
        const auto& cinfo = img->create_info();
        if (cinfo.extent.width == dims.width && cinfo.extent.height == dims.height &&
            cinfo.extent.depth == dims.depth && cinfo.array_layers == dims.layers &&
            cinfo.mip_levels == dims.levels && cinfo.samples == dims.samples &&
            cinfo.format == vk2::to_vkformat(dims.format) &&
            cinfo.override_usage_flags == dims.image_usage_flags) {
          // use existing
        } else {
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
          // auto handle = vk2::get_device().create_image(info);
        }
      }

      // make a new image if needed
    }
  }
}

}  // namespace gfx
