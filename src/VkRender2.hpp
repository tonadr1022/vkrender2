#pragma once

#include <vulkan/vulkan_core.h>

#include <filesystem>
#include <optional>
#include <queue>
#include <vector>

#include "AABB.hpp"
#include "BaseRenderer.hpp"
#include "CommandEncoder.hpp"
#include "IBL.hpp"
#include "RenderGraph.hpp"
#include "Scene.hpp"
#include "SceneLoader.hpp"
#include "StateTracker.hpp"
#include "Types.hpp"
#include "shaders/common.h.glsl"
#include "techniques/CSM.hpp"
#include "util/IndexAllocator.hpp"
#include "vk2/Buffer.hpp"
#include "vk2/DeletionQueue.hpp"
#include "vk2/Device.hpp"
#include "vk2/PipelineManager.hpp"
#include "vk2/SamplerCache.hpp"
#include "vk2/Texture.hpp"

namespace gfx {

struct LinearAllocator {
  explicit LinearAllocator(u64 size) : size(size) {}
  u64 size;
  u64 curr_offset{};

  u64 alloc(u64 size) {
    u64 offset = curr_offset;
    curr_offset += size;
    return offset;
  }

  void free() {
    size = 0;
    curr_offset = 0;
  }
};

struct MaterialData {
  u32 albedo_tex;
};

template <typename T>
struct InFlightResource {
  T data;
  VkFence fence{};
};

struct LinearStagingBuffer {
  explicit LinearStagingBuffer(vk2::Buffer* buffer) : buffer_(buffer) {}

  u64 copy(const void* data, u64 size) {
    memcpy((char*)buffer_->mapped_data() + offset_, data, size);
    offset_ += size;
    return offset_ - size;
  }
  [[nodiscard]] vk2::Buffer* get_buffer() const { return buffer_; }

 private:
  u64 offset_{};
  vk2::Buffer* buffer_{};
};

struct VkRender2 final : public BaseRenderer {
  static VkRender2& get();
  static void init(const InitInfo& info);
  static void shutdown();
  explicit VkRender2(const InitInfo& info);
  ~VkRender2() override;

  SceneHandle load_scene(const std::filesystem::path& path, bool dynamic = false,
                         const mat4& transform = mat4{1});
  void submit_static(SceneHandle scene, mat4 transform = mat4{1});

  // TODO: private
  void immediate_submit(std::function<void(VkCommandBuffer cmd)>&& function);
  void immediate_submit(std::function<void(CmdEncoder& cmd)>&& function);
  void transfer_submit(std::function<void(VkCommandBuffer cmd, VkFence fence,
                                          std::queue<InFlightResource<vk2::Buffer*>>&)>&& function);
  [[nodiscard]] const QueueFamilies& get_queues() const { return queues_; }
  void set_env_map(const std::filesystem::path& path);
  void bind_bindless_descriptors(CmdEncoder& ctx);

 private:
  void on_draw(const SceneDrawInfo& info) override;
  void on_gui() override;
  void on_resize() override;

  // TODO: refactor
  struct SceneGPUResources {
    vk2::Buffer vertex_buffer;
    vk2::Buffer index_buffer;
    vk2::Buffer materials_buffer;
    vk2::Buffer draw_indirect_buffer;
    vk2::Buffer material_indices;
    vk2::Buffer instance_buffer;
    std::vector<vk2::Image> textures;
    u32 draw_cnt{};
  };

  struct LoadedScene {
    SceneLoadData scene_graph_data;
    std::unique_ptr<SceneGPUResources> resources;
  };

  // TODO: move ownership elsewhere. renderer shuld only
  // own gpu resources
 public:
  std::vector<LoadedScene> loaded_dynamic_scenes_;

 private:
  struct FrameData {
    std::optional<vk2::Buffer> scene_uniform_buf;
    // vk2::Holder<vk2::BufferHandle> draw_cnt_buf;
    // vk2::Holder<vk2::BufferHandle> final_draw_cmd_buf;
  };
  std::vector<FrameData> per_frame_data_2_;
  FrameData& curr_frame_2() { return per_frame_data_2_[curr_frame_num() % 2]; }
  void init_pipelines();
  void init_indirect_drawing();
  static constexpr u32 max_draws{100'000};

  struct SceneUniforms {
    mat4 view_proj;
    mat4 view;
    mat4 proj;
    uvec4 debug_flags;
    vec3 view_pos;
    float _p1;
    vec3 light_dir;
    float _p2;
    vec3 light_color;
    float ambient_intensity;
  };
  SceneUniforms scene_uniform_cpu_data_;

  VkCommandPool imm_cmd_pool_;
  VkCommandBuffer imm_cmd_buf_;

  // TODO: tune or make adjustable
  struct LinearBuffer {
    explicit LinearBuffer(const vk2::BufferCreateInfo& info) : buffer(info), allocator(info.size) {}
    explicit LinearBuffer(vk2::Buffer buffer)
        : buffer(std::move(buffer)), allocator(buffer.size()) {}
    vk2::Buffer buffer;
    LinearAllocator allocator;
    u64 alloc(u64 size) { return allocator.alloc(size); }
  };

  template <typename T>
  struct SlotBuffer {
    explicit SlotBuffer(const vk2::BufferCreateInfo& info)
        : buffer(info), allocator(info.size / sizeof(T)) {}

    explicit SlotBuffer(vk2::Buffer buffer)
        : buffer(std::move(buffer)), allocator(this->buffer.size() / sizeof(T)) {}

    vk2::Buffer buffer;
    util::SlotAllocator<T> allocator;
    u64 alloc() { return allocator.alloc(); }
  };

  u64 draw_cnt_{};
  u64 obj_data_cnt_{};
  struct DrawStats {
    u64 total_vertices;
    u64 total_indices;
    u32 vertices;
    u32 indices;
    u32 draw_cmds;
    u32 textures;
    u32 materials;
  };

  struct StaticSceneGPUResources {
    SceneLoadData scene_graph_data;
    std::vector<gfx::Material> materials;
    std::vector<gfx::PrimitiveDrawInfo> mesh_draw_infos;
    std::vector<util::SlotAllocator<gfx::ObjectData>::Slot> object_data_slots;
    u64 first_vertex;
    u64 first_index;
    u64 num_vertices;
    u64 num_indices;
    u64 materials_idx_offset;
  };
  struct GPUDrawInfo {
    u32 index_cnt;
    u32 first_index;
    u32 vertex_offset;
    u32 pad;
  };

  struct ObjectDraw {
    std::vector<util::SlotAllocator<GPUDrawInfo>::Slot> draw_cmd_slots;
    std::vector<util::SlotAllocator<gfx::ObjectData>::Slot> obj_data_slots;
  };

  std::unordered_map<SceneHandle, ObjectDraw> active_static_draws_;
  std::unordered_map<std::string, StaticSceneGPUResources> static_scenes_;

  gfx::RenderGraph rg_;
  DrawStats static_draw_stats_{};

  struct StaticGPUDrawData {
    vk2::Holder<vk2::BufferHandle> output_draw_cmd_buf[max_frames_in_flight];
    vk2::Holder<vk2::BufferHandle> alpha_mask_output_draw_cmd_buf[max_frames_in_flight];
  };

  struct StaticMeshDrawManager {
    explicit StaticMeshDrawManager(size_t initial_max_draw_cnt);
    StaticMeshDrawManager(const StaticMeshDrawManager&) = delete;
    StaticMeshDrawManager(StaticMeshDrawManager&&) = delete;
    StaticMeshDrawManager& operator=(const StaticMeshDrawManager&) = delete;
    StaticMeshDrawManager& operator=(StaticMeshDrawManager&&) = delete;

    struct Alloc {
      util::FreeListAllocator::Slot draw_cmd_slot;
    };
    using RenderObjectHandle = GenerationalHandle<struct Alloc>;
    // RenderObjectHandle add_draws(std::span<GPUDrawInfo> draws);
    RenderObjectHandle add_draws(VkCommandBuffer cmd, size_t size, size_t staging_offset,
                                 vk2::Buffer& staging);
    void remove_scene(RenderObjectHandle handle);

    Pool<RenderObjectHandle, Alloc> allocs_;
    util::FreeListAllocator draw_cmds_buf_allocator;
    vk2::Holder<vk2::BufferHandle> draw_cmds_buf_handle;  // GPUDrawInfo

    vk2::Holder<vk2::BufferHandle>
        out_draw_cmds_buf[max_frames_in_flight];  // [draw cnt][DrawCmd[]]
  };

  std::optional<StaticMeshDrawManager> opaque_draw_mgr_;
  std::optional<StaticMeshDrawManager> opaque_alpha_mask_draw_mgr_;
  std::optional<StaticMeshDrawManager> transparent_draw_mgr_;

  StaticGPUDrawData unculled_draw_data_;
  StaticGPUDrawData main_culled_draw_data_;

  void draw_opaque(CmdEncoder& cmd, const StaticGPUDrawData& draw_buf);

 public:
  // TODO: remove this this is awful
  std::optional<LinearBuffer> static_vertex_buf_;

 private:
  std::optional<LinearBuffer> static_index_buf_;
  std::optional<LinearBuffer> static_materials_buf_;
  std::optional<LinearBuffer> static_instance_data_buf_;

  AABB scene_aabb_{};
  std::optional<SlotBuffer<GPUDrawInfo>> static_draw_info_buf_;
  std::optional<SlotBuffer<gfx::ObjectData>> static_object_data_buf_;
  std::vector<vk2::Image> static_textures_;

  StateTracker state_;
  StateTracker transfer_q_state_;

  std::optional<vk2::Sampler> linear_sampler_;
  std::optional<vk2::Sampler> nearest_sampler_;
  std::optional<vk2::Sampler> linear_sampler_clamp_to_edge_;

  struct DefaultData {
    std::optional<vk2::Image> white_img;
  } default_data_;
  gfx::DefaultMaterialData default_mat_data_;

  struct InstanceData {
    u32 material_id;
    u32 instance_id;
  };

  std::filesystem::path default_env_map_path_;
  std::unique_ptr<CSM> csm_;
  vk2::Sampler shadow_sampler_;
  std::optional<IBL> ibl_;
  vk2::DeletionQueue main_del_q_;
  vk2::PipelineHandle img_pipeline_;
  vk2::PipelineHandle draw_pipeline_;
  vk2::PipelineHandle cull_objs_pipeline_;
  vk2::PipelineHandle skybox_pipeline_;
  vk2::PipelineHandle postprocess_pipeline_;
  vk2::PipelineHandle gbuffer_pipeline_;
  vk2::PipelineHandle deferred_shade_pipeline_;
  Format gbuffer_a_format_{Format::R32G32B32A32Sfloat};
  Format gbuffer_b_format_{Format::R32G32B32A32Sfloat};
  Format gbuffer_c_format_{Format::R32G32B32A32Sfloat};
  Format draw_img_format_{Format::R32G32B32A32Sfloat};
  Format depth_img_format_{Format::D32Sfloat};
  // TODO: make more robust settings
  bool deferred_enabled_{true};
  bool frustum_cull_enabled_{true};
  VkPipelineLayout default_pipeline_layout_{};
  std::queue<InFlightResource<vk2::Buffer*>> pending_buffer_transfers_;

  std::vector<vk2::Buffer> free_staging_buffers_;

  std::unordered_map<u64, VkDescriptorSet> imgui_desc_sets_;
  VkDescriptorSet get_imgui_set(VkSampler sampler, VkImageView view);
  // non owning
  VkDescriptorSet main_set_{};
  VkDescriptorSet main_set2_{};
  VmaAllocator allocator_;
  // end non owning

  u32 debug_mode_{DEBUG_MODE_NONE};
  const char* debug_mode_to_string(u32 mode);
  std::filesystem::path env_tex_path_;
  i32 prefilter_mip_skybox_level_{};
  // TODO: enum loser
  bool render_prefilter_mip_skybox_{};
  void make_cubemap_views_all_mips(const vk2::Image& texture,
                                   std::vector<std::optional<vk2::ImageView>>& views);
  void generate_mipmaps(StateTracker& state, VkCommandBuffer cmd, vk2::Image& tex);
  u64 cube_vertices_gpu_offset_{};
  // u64 cube_indices_gpu_offset_{};

  void add_basic_forward_pass(RenderGraph& rg);
  AttachmentInfo swapchain_att_info_;
// TODO: fix
#ifdef __APPLE__
  bool portable_{true};
#else
  bool portable_{false};
#endif
  u32 tonemap_type_{1};
  const char* tonemap_type_names_[2] = {"Optimized Filmic", "ACES Film"};

 public:
  std::optional<vk2::Image> load_hdr_img(CmdEncoder& ctx, const std::filesystem::path& path,
                                         bool flip = false);
  void draw_cube(VkCommandBuffer cmd) const;
  void generate_mipmaps(CmdEncoder& ctx, vk2::Image& tex);
  [[nodiscard]] const DefaultData& get_default_data() const { return default_data_; }
};

}  // namespace gfx
