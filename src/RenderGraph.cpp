#include "RenderGraph.hpp"

#include <vulkan/vulkan_core.h>

#include <algorithm>
#include <cstdint>
#include <expected>
#include <utility>

#include "Logger.hpp"

namespace gfx {

RenderResourceHandle Pass::add_color_output(const std::string& name, const AttachmentInfo& info,
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
  resource_writes_.emplace_back(handle, ResourceUsage::ColorOutput);
  return handle;
}

RenderResourceHandle Pass::add_texture_input(const std::string& name) {
  uint32_t handle = graph_.get_or_add_texture_resource(name);
  RenderTextureResource* res = graph_.get_texture_resource(handle);
  res->read_in_pass(idx_);
  res->image_usage |= VK_IMAGE_USAGE_SAMPLED_BIT;
  resource_reads_.emplace_back(handle, ResourceUsage::TextureInput);
  return handle;
}

RenderResourceHandle Pass::set_depth_stencil_input(const std::string& name) {
  uint32_t handle = graph_.get_or_add_texture_resource(name);
  RenderTextureResource* res = graph_.get_texture_resource(handle);
  res->read_in_pass(idx_);
  res->image_usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
  resource_reads_.emplace_back(handle, ResourceUsage::DepthStencilInput);
  return handle;
}

RenderResourceHandle Pass::set_depth_stencil_output(const std::string& name,
                                                    const AttachmentInfo& info) {
  uint32_t handle = graph_.get_or_add_texture_resource(name);
  RenderTextureResource* res = graph_.get_texture_resource(handle);
  res->written_in_pass(idx_);
  res->image_usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
  res->info = info;
  resource_writes_.emplace_back(handle, ResourceUsage::DepthStencilOutput);
  return handle;
}

Pass::Pass(std::string name, ExecuteFn fn, RenderGraph& graph, uint32_t idx)
    : name_(std::move(name)), execute_(std::move(fn)), graph_(graph), idx_(idx) {}

RenderGraph::RenderGraph(std::string name) : name_(std::move(name)) {}

Pass& RenderGraph::add_pass(const std::string& name, ExecuteFn execute) {
  auto idx = passes_.size();
  passes_.emplace_back(name, std::move(execute), *this, idx);
  return passes_.back();
}

VoidResult RenderGraph::bake() {
  // validate
  // go through each input and make sure it's an output of a previous pass if it needs to read from
  // it
  // for (auto& pass : passes_) {
  //   // for (auto& tex_resource : pass.)
  // }

  for (const auto& resource : resources_) {
    for (const auto& b : resource->get_read_passes()) {
      LINFO("{}: read in {}", resource->name, passes_[b].get_name());
    }
    for (const auto& b : resource->get_written_passes()) {
      LINFO("{}: written in {}", resource->name, passes_[b].get_name());
    }
  }

  // TODO: validate that backbuffer img has color write

  // find sinks
  pass_stack_.clear();
  pass_dependencies_.resize(passes_.size());
  for (uint32_t pass_i = 0; pass_i < passes_.size(); pass_i++) {
    auto& pass = passes_[pass_i];
    for (auto& usage : pass.get_resource_writes()) {
      if (resources_[usage.idx]->name == backbuffer_img_) {
        // TODO: move this to validation phase
        if (usage.usage != ResourceUsage::ColorOutput) {
          return std::unexpected("backbuffer output is not of usage ColorOutput");
        }
        pass_stack_.emplace_back(pass_i);
      }
    }
  }

  auto sink_cnt = pass_stack_.size();
  if (sink_cnt == 0) {
    return std::unexpected("no backbuffer writes found");
  }

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

  LINFO("pass order: ");
  for (const auto& s : pass_stack_) {
    LINFO("{}", passes_[s].get_name());
  }

  // build physical resources
  // for (uint32_t pass_i : pass_stack_) {
  // }
  return {};
}

void RenderGraph::log() {}

void RenderGraph::execute() {}

// TODO: not recursive
VoidResult RenderGraph::traverse_dependencies_recursive(uint32_t pass_i, uint32_t stack_size) {
  if (stack_size > passes_.size()) {
    return std::unexpected("cycle detected");
  }

  auto& pass = passes_[pass_i];
  for (const auto& resource_read : pass.get_resource_reads()) {
    std::span<const uint32_t> passes_writing_to_resource{};
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
    case ResourceUsage::TextureInput:
    case ResourceUsage::DepthStencilOutput:
    case ResourceUsage::DepthStencilInput:
      return true;
  }
  return false;
}
}  // namespace gfx
