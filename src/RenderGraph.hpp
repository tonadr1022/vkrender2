#pragma once

#include <vulkan/vulkan_core.h>

#include <expected>
#include <filesystem>
#include <functional>
#include <span>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "Types.hpp"
#include "core/FixedVector.hpp"
#include "vk2/Pool.hpp"

namespace gfx {

using VoidResult = std::expected<void, const char*>;

struct CmdEncoder;
using ExecuteFn = std::function<void(CmdEncoder& cmd)>;

struct BufferInfo {
  BufferHandle handle;
  size_t size{};
};

struct ResourceDimensions {
  Format format{};
  BufferInfo buffer_info;
  ImageHandle external_img_handle;
  SizeClass size_class{SizeClass::SwapchainRelative};
  uint32_t width{}, height{}, depth{}, layers{1}, levels{1}, samples{1};
  bool scaled{true};
  bool is_swapchain{};
  Access access_usage{};
  // VkImageUsageFlags image_usage_flags{};
  // TODO: queues
  [[nodiscard]] bool is_image() const;
  friend bool operator==(const ResourceDimensions& a, const ResourceDimensions& b) {
    bool valid_extent = false;
    if (a.size_class == SizeClass::SwapchainRelative && a.size_class == b.size_class) {
      valid_extent = true;
    } else {
      valid_extent = a.size_class == b.size_class && a.width == b.width && a.height == b.height &&
                     a.depth == b.depth;
    }
    return valid_extent && a.format == b.format && a.layers == b.layers && a.levels == b.levels &&
           a.access_usage == b.access_usage;
  }
};

struct ResourceDimensionsHasher {
  std::size_t operator()(const ResourceDimensions& dims) const;
};

struct TextureUsage {
  std::string name;
};

struct PassCreateInfo {};

struct RenderGraph;

struct RenderResource {
  enum class Type : uint8_t { Buffer, Texture };
  static constexpr uint32_t unused = UINT32_MAX;
  RenderResource(Type type, uint32_t idx) : type_(type), idx_(idx) {}
  RenderResource(std::string name, uint32_t phyisical_idx, Type type, uint32_t idx)
      : name(std::move(name)), physical_idx(phyisical_idx), type_(type), idx_(idx) {}

  [[nodiscard]] Type get_type() const { return type_; }
  [[nodiscard]] uint32_t get_idx() const { return idx_; }

  void read_in_pass(uint32_t pass) { read_passes_.emplace_back(pass); }
  [[nodiscard]] std::span<const uint16_t> get_read_passes() const { return read_passes_; }

  void written_in_pass(uint32_t pass) { written_passes_.emplace_back(pass); }
  [[nodiscard]] std::span<const uint16_t> get_written_passes() const { return written_passes_; }

  std::string name;
  uint32_t physical_idx{unused};
  AttachmentInfo info;
  // VkImageUsageFlags image_usage{};
  Access access{};
  BufferInfo buffer_info{};
  ImageHandle img_handle;

 private:
  Type type_;
  uint32_t idx_{unused};
  // TODO: flat set
  util::fixed_vector<uint16_t, 20> written_passes_;
  util::fixed_vector<uint16_t, 20> read_passes_;
};

// enum class ResourceUsage : uint8_t {
//   None = 0,
//   // ColorInput,
//   ColorOutput,
//   TextureInput,
//   StorageImageInput,
//   DepthStencilOutput,
//   DepthStencilInput,
//   BufferInput,
//   BufferOutput,
// };

// TODO: better handle type
using RenderResourceHandle = uint32_t;

struct RGResourceHandle {
  RGResourceHandle(u32 idx, RenderResource::Type type) : idx(idx), type(type) {}
  RGResourceHandle() = default;
  u32 idx{UINT32_MAX};
  RenderResource::Type type;
  [[nodiscard]] bool is_valid() const { return idx != UINT32_MAX; }
  bool operator==(RGResourceHandle& other) const { return idx == other.idx && type == other.type; }
};

struct RenderGraphPass {
  enum class Type : uint8_t { Compute, Graphics };
  explicit RenderGraphPass(std::string name, RenderGraph& graph, uint32_t idx, Type type);
  RGResourceHandle add_image_access(const std::string& name, Access access);
  void add(BufferHandle buf_handle, Access access);
  RGResourceHandle add(const std::string& name, const AttachmentInfo& info, Access access,
                       const std::string& input = "");
  void add(ImageHandle image, Access access);

  template <typename F>
  void set_execute_fn(F&& fn)
    requires std::invocable<F&, CmdEncoder&>
  {
    execute_ = std::forward<F>(fn);
  }

  // TODO: assign queue to the pass
  [[nodiscard]] const std::string& get_name() const { return name_; }
  [[nodiscard]] uint32_t get_idx() const { return idx_; }

  struct UsageAndHandle {
    RGResourceHandle handle;
    VkAccessFlags2 access_flags{};
    VkPipelineStageFlags2 stages{};
    Access access{};
  };

  // [[nodiscard]] const std::vector<UsageAndHandle>& get_resource_inputs() const {
  //   return resource_inputs_;
  // }
  // [[nodiscard]] const std::vector<UsageAndHandle>& get_resource_outputs() const {
  //   return resource_outputs_;
  // }
  [[nodiscard]] const UsageAndHandle* get_swapchain_write_usage() const;
  [[nodiscard]] const std::vector<UsageAndHandle>& get_resources() const { return resources_; }

 private:
  friend struct RenderGraph;

  u32 swapchain_write_idx_{RenderResource::unused};
  std::vector<UsageAndHandle> resources_;
  std::vector<u32> resource_read_indices_;
  UsageAndHandle init_usage_and_handle(Access access, RGResourceHandle handle, RenderResource& res);
  // std::vector<UsageAndHandle> resource_inputs_;
  // std::vector<UsageAndHandle> resource_outputs_;

  const std::string name_;
  ExecuteFn execute_;
  RenderGraph& graph_;
  const uint32_t idx_;

  // [[nodiscard]] bool contains_input(const std::string& name) const;
};

struct RenderGraph {
  explicit RenderGraph(std::string name = "RenderGraph");
  RenderGraphPass& add_pass(const std::string& name,
                            RenderGraphPass::Type type = RenderGraphPass::Type::Graphics);
  void set_backbuffer_img(const std::string& name) { backbuffer_img_ = name; }
  void set_render_scale(float render_scale) { render_scale_ = render_scale; }
  [[nodiscard]] const std::string& get_backbuffer_img_name() const { return backbuffer_img_; }
  void reset();
  VoidResult bake();
  VoidResult output_graphvis(const std::filesystem::path& path);
  void setup_attachments();
  void execute(CmdEncoder& cmd);

  RGResourceHandle get_or_add_buffer_resource(BufferHandle handle);
  RGResourceHandle get_or_add_texture_resource(ImageHandle handle);
  RGResourceHandle get_or_add_texture_resource(const std::string& name);
  RenderResource* get_resource(RGResourceHandle handle);
  Image* get_texture(RGResourceHandle handle);
  Image* get_texture(RenderResource* resource);
  ImageHandle get_texture_handle(RenderResource* resource);
  ImageHandle get_texture_handle(RGResourceHandle resource);

  [[nodiscard]] const AttachmentInfo& get_swapchain_info() const { return desc_; }
  void print_pass_order();

 private:
  // TODO: integrate swapchain more closely?
  friend struct RenderGraphPass;

  VkImage swapchain_img_{};
  std::string name_;
  std::vector<RenderGraphPass> passes_;
  std::string backbuffer_img_;

  struct Barrier {
    uint32_t resource_idx;
    VkImageLayout layout;
    VkAccessFlags2 access;
    VkPipelineStageFlags2 stages;
  };

  struct PassSubmissionState {
    void reset();
    std::vector<VkImageMemoryBarrier2> image_barriers;
    std::vector<VkBufferMemoryBarrier2> buffer_barriers;
  };
  std::vector<PassSubmissionState> pass_submission_state_;

  struct PhysicalPass {
    std::string name;
    std::vector<u32> discard_resources;
    std::vector<Barrier> flush_barriers;
    std::vector<Barrier> invalidate_barriers;
    std::vector<uint32_t> physical_color_attachments;
    uint32_t physical_depth_stencil{RenderResource::unused};
    void reset();
  };

  struct ResourceState {
    VkAccessFlags2 invalidated_in_stage[64] = {};
    VkAccessFlags2 to_flush_access{};
    VkPipelineStageFlags2 pipeline_barrier_src_stages{};
    VkImageLayout layout{VK_IMAGE_LAYOUT_UNDEFINED};
  };

  std::vector<PhysicalPass> physical_passes_;
  bool needs_invalidate(const Barrier& barrier, const ResourceState& state);

  void prune_duplicates(std::vector<uint32_t>& data);

  // TODO: pool
  std::vector<RenderResource> resources_;
  std::vector<ResourceDimensions> physical_resource_dims_;
  std::unordered_map<ImageHandle, ResourceState> image_pipeline_states_;
  std::unordered_map<BufferHandle, ResourceState> buffer_pipeline_states_;
  ResourceState* get_resource_pipeline_state(u32 physical_idx);

  std::unordered_map<std::string, RGResourceHandle> resource_to_idx_map_;
  std::unordered_map<BufferHandle, RGResourceHandle> buffer_to_idx_map_;
  std::unordered_map<ImageHandle, RGResourceHandle> image_to_idx_map_;

  // TODO: bitset
  std::vector<std::unordered_set<uint32_t>> pass_dependencies_;
  std::vector<uint32_t> pass_stack_;
  std::vector<uint32_t> swapchain_writer_passes_;
  VoidResult traverse_dependencies_recursive(uint32_t pass_i, uint32_t stack_size);
  std::unordered_set<u32> visited_;

  std::unordered_set<uint32_t> dup_prune_set_;

  std::unordered_multimap<ResourceDimensions, Holder<ImageHandle>, ResourceDimensionsHasher>
      img_cache_;
  std::vector<std::pair<ResourceDimensions, Holder<ImageHandle>>> img_cache_used_;
  std::vector<ImageHandle> physical_image_attachments_;
  std::vector<BufferHandle> physical_buffers_;

  [[nodiscard]] ResourceDimensions get_resource_dims(const RenderResource& resource) const;
  void build_physical_resource_reqs();
  void build_barrier_infos();
  void build_resource_aliases();
  void physical_pass_setup_barriers(u32 pass_i);
  void print_barrier(const VkImageMemoryBarrier2& barrier) const;
  void print_barrier(const VkBufferMemoryBarrier2& barrier) const;
  VoidResult validate();

  std::unordered_map<u64, BufferHandle> buffer_bindings_;

  struct ResourceState2 {
    VkImageLayout initial_layout{};
    VkImageLayout final_layout{};
    VkAccessFlags2 invalidated_accesses{};
    VkAccessFlags2 flushed_accesses{};
    VkPipelineStageFlags2 invalidated_stages{};
    VkPipelineStageFlags2 flushed_stages{};
  };
  std::vector<ResourceState2> resource_states_;
  AttachmentInfo desc_;
  bool log_ = true;
  float render_scale_{1};
};

/*
dims1 dims2 dims3
dims3 == dims1

pass1 pass2 pass3
use1  use2  use3

in pass 3, we know use1 is never used again, so it can be reused


*/

}  // namespace gfx
