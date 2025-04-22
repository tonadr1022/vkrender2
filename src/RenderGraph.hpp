#pragma once

#include <vulkan/vulkan_core.h>

#include <expected>
#include <filesystem>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "FixedVector.hpp"
#include "Types.hpp"
#include "vk2/Device.hpp"

namespace gfx {

using VoidResult = std::expected<void, const char*>;

struct CmdEncoder;
using ExecuteFn = std::function<void(CmdEncoder& cmd)>;

enum class AttachmentType : uint8_t { Color, Depth, Stencil };

enum Access : uint16_t {
  None = 1ULL << 0,
  ColorWrite = 1ULL << 1,
  ColorRead = 1ULL << 2,
  ColorRW = ColorRead | ColorWrite,
  DepthStencilRead = 1ULL << 3,
  DepthStencilWrite = 1ULL << 4,
  DepthStencilRW = DepthStencilRead | DepthStencilWrite,
  VertexRead = 1ULL << 5,
  IndexRead = 1ULL << 6,
  IndirectRead = 1ULL << 7,
  ComputeRead = 1ULL << 8,
  ComputeWrite = 1ULL << 9,
  ComputeRW = ComputeRead | ComputeWrite,
  TransferRead = 1ULL << 10,
  TransferWrite = 1ULL << 11,
  FragmentRead = 1ULL << 12,
};

using AccessFlags = uint32_t;

enum class SizeClass : uint8_t { Absolute, SwapchainRelative, InputRelative };

using AttachmentInfoFlags = uint8_t;

struct AttachmentInfo {
  SizeClass size_class{SizeClass::SwapchainRelative};
  uvec3 dims{1};
  ImageUsageFlags aux_flags{};
  Format format{};
  u32 layers{1};
  u32 levels{1};
};

struct BufferInfo {
  vk2::BufferHandle handle;
  size_t size{};
  u64 proxy_hash{};
};

struct ResourceDimensions {
  Format format{};
  BufferInfo buffer_info;
  SizeClass size_class{SizeClass::SwapchainRelative};
  uint32_t width{}, height{}, depth{}, layers{1}, levels{1}, samples{1};
  VkImageUsageFlags image_usage_flags{};
  // TODO: queues
  [[nodiscard]] bool is_storage_image() const;
  [[nodiscard]] bool is_image() const;
};

struct BufferUsage {
  std::string name;
  size_t size{};
  BufferUsageFlags usage{};
};

struct TextureUsage {
  std::string name;
};

struct PassCreateInfo {};

enum class ResourceType : uint8_t { Buffer, Texture };
struct RenderGraph;

struct RenderResource {
  static constexpr uint32_t unused = UINT32_MAX;
  RenderResource(ResourceType type, uint32_t idx) : type_(type), idx_(idx) {}
  RenderResource(std::string name, uint32_t phyisical_idx, ResourceType type, uint32_t idx)
      : name(std::move(name)), physical_idx(phyisical_idx), type_(type), idx_(idx) {}

  [[nodiscard]] ResourceType get_type() const { return type_; }
  [[nodiscard]] uint32_t get_idx() const { return idx_; }

  void read_in_pass(uint32_t pass) { read_passes_.emplace_back(pass); }
  [[nodiscard]] std::span<const uint16_t> get_read_passes() const { return read_passes_; }

  void written_in_pass(uint32_t pass) { written_passes_.emplace_back(pass); }
  [[nodiscard]] std::span<const uint16_t> get_written_passes() const { return written_passes_; }

  std::string name;
  uint32_t physical_idx{unused};
  AttachmentInfo info;
  VkImageUsageFlags image_usage{};
  Access access{};
  BufferInfo buffer_info{};
  vk2::ImageHandle img_handle;

 private:
  ResourceType type_;
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

struct RenderGraphPass {
  enum class Type : uint8_t { Compute, Graphics };
  explicit RenderGraphPass(std::string name, RenderGraph& graph, uint32_t idx, Type type);
  void add(const std::string& name, vk2::BufferHandle buffer, Access access);
  void add(const std::string& name, const vk2::Holder<vk2::BufferHandle>& buffer, Access access);
  // void add(const std::string& name, vk2::ImageHandle image, Access access);
  void add_proxy(const std::string& name, Access access);
  RenderResourceHandle add(const std::string& name, const AttachmentInfo& info, Access access);

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
    uint32_t idx{};
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
  UsageAndHandle init_usage_and_handle(Access access, RenderResourceHandle handle,
                                       RenderResource& res);
  // std::vector<UsageAndHandle> resource_inputs_;
  // std::vector<UsageAndHandle> resource_outputs_;

  const std::string name_;
  ExecuteFn execute_;
  RenderGraph& graph_;
  const uint32_t idx_;
  const Type type_{Type::Graphics};

  // [[nodiscard]] bool contains_input(const std::string& name) const;
};

struct RenderGraphSwapchainInfo {
  VkImage curr_img{};
  uint32_t width{}, height{};
};

struct RenderGraph {
  explicit RenderGraph(std::string name = "RenderGraph");
  void set_swapchain_info(const RenderGraphSwapchainInfo& info);
  RenderGraphPass& add_pass(const std::string& name,
                            RenderGraphPass::Type type = RenderGraphPass::Type::Graphics);
  void set_backbuffer_img(const std::string& name) { backbuffer_img_ = name; }
  const std::string& get_backbuffer_img_name() const { return backbuffer_img_; }
  void reset();
  VoidResult bake();
  VoidResult output_graphvis(const std::filesystem::path& path);
  void setup_attachments();
  void execute(CmdEncoder& cmd);

  uint32_t get_or_add_buffer_resource(const std::string& name);
  uint32_t get_or_add_texture_resource(const std::string& name);
  RenderResource* get_resource(RenderResourceHandle idx);
  vk2::Image* get_texture(RenderResourceHandle idx);
  vk2::Image* get_texture(RenderResource* resource);
  vk2::ImageHandle get_texture_handle(RenderResource* resource);
  vk2::ImageHandle get_texture_handle(RenderResourceHandle resource);
  void set_resource(const std::string& name, vk2::BufferHandle handle);

 private:
  // TODO: integrate swapchain more closely?
  RenderGraphSwapchainInfo swapchain_info_{};
  std::string name_;
  std::vector<RenderGraphPass> passes_;
  std::string backbuffer_img_;

  struct Barrier {
    uint32_t resource_idx;
    VkImageLayout layout;
    VkAccessFlags2 access;
    VkPipelineStageFlags2 stages;
  };

  // invalidate barriers are when gpu cache needs to be invalidated (read)
  // flush barriers are when gpu cache needs to be flushed (write)

  // std::vector<Barrier> pass_flush_barriers_;
  // std::vector<Barrier> pass_invalidate_barriers_;

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
  std::vector<ResourceState> resource_pipeline_states_;
  std::unordered_map<std::string, uint32_t> resource_to_idx_map_;

  // TODO: bitset
  std::vector<std::unordered_set<uint32_t>> pass_dependencies_;
  std::vector<uint32_t> pass_stack_;
  std::vector<uint32_t> swapchain_writer_passes_;
  VoidResult traverse_dependencies_recursive(uint32_t pass_i, uint32_t stack_size);

  std::unordered_set<uint32_t> dup_prune_set_;

  std::vector<vk2::Holder<vk2::ImageHandle>> physical_image_attachments_;
  std::vector<vk2::BufferHandle> physical_buffers_;

  [[nodiscard]] ResourceDimensions get_resource_dims(const RenderResource& resource) const;
  void build_physical_resource_reqs();
  void build_barrier_infos();
  void physical_pass_setup_barriers(u32 pass_i);
  void print_barrier(const VkImageMemoryBarrier2& barrier) const;
  void print_barrier(const VkBufferMemoryBarrier2& barrier) const;
  VoidResult validate();

  std::unordered_map<u64, vk2::BufferHandle> buffer_bindings_;

  struct ResourceState2 {
    VkImageLayout initial_layout{};
    VkImageLayout final_layout{};
    VkAccessFlags2 invalidated_accesses{};
    VkAccessFlags2 flushed_accesses{};
    VkPipelineStageFlags2 invalidated_stages{};
    VkPipelineStageFlags2 flushed_stages{};
  };
  std::vector<ResourceState2> resource_states_;
};

}  // namespace gfx
