#pragma once

#include <vulkan/vulkan_core.h>

#include <filesystem>
#include <optional>
#include <queue>
#include <vector>

#include "AABB.hpp"
#include "BaseRenderer.hpp"
#include "CommandEncoder.hpp"
#include "RenderGraph.hpp"
#include "Scene.hpp"
#include "SceneLoader.hpp"
#include "StateTracker.hpp"
#include "Types.hpp"
#include "shaders/common.h.glsl"
#include "techniques/CSM.hpp"
#include "techniques/IBL.hpp"
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

struct FreeListBuffer {
  Holder<BufferHandle> buffer;
  util::FreeListAllocator allocator;
  [[nodiscard]] vk2::Buffer* get_buffer() const { return vk2::get_device().get_buffer(buffer); }
};

struct VkRender2 final : public BaseRenderer {
  struct Line {
    vec3 p1, p2;
    vec4 color;
  };
  static VkRender2& get();
  static void init(const InitInfo& info);
  static void shutdown();
  explicit VkRender2(const InitInfo& info);
  ~VkRender2() override;

  SceneHandle load_model(const std::filesystem::path& path, bool dynamic = false,
                         const mat4& transform = mat4{1});

  // TODO: private
  void immediate_submit(std::function<void(VkCommandBuffer cmd)>&& function);
  void immediate_submit(std::function<void(CmdEncoder& cmd)>&& function);
  void transfer_submit(std::function<void(VkCommandBuffer cmd, VkFence fence,
                                          std::queue<InFlightResource<vk2::Buffer*>>&)>&& function);
  void enqueue_transfer();
  void set_env_map(const std::filesystem::path& path);
  void bind_bindless_descriptors(CmdEncoder& ctx);

 private:
  void on_draw(const SceneDrawInfo& info) override;
  void on_imgui() override;
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
    Holder<BufferHandle> line_draw_buf;
    // vk2::Holder<vk2::BufferHandle> draw_cnt_buf;
    // vk2::Holder<vk2::BufferHandle> final_draw_cmd_buf;
  };

  std::vector<FrameData> per_frame_data_2_;
  FrameData& curr_frame_2() { return per_frame_data_2_[curr_frame_num() % 2]; }
  void init_pipelines();
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

  struct DrawStats {
    u64 total_vertices;
    u64 total_indices;
    u32 vertices;
    u32 indices;
    u32 textures;
    u32 materials;
  };

  struct StaticMeshDrawManager {
    using Handle = GenerationalHandle<struct Alloc>;

    explicit StaticMeshDrawManager(std::string name, size_t initial_max_draw_cnt);
    StaticMeshDrawManager(const StaticMeshDrawManager&) = delete;
    StaticMeshDrawManager(StaticMeshDrawManager&&) = delete;
    StaticMeshDrawManager& operator=(const StaticMeshDrawManager&) = delete;
    StaticMeshDrawManager& operator=(StaticMeshDrawManager&&) = delete;

    struct Alloc {
      util::FreeListAllocator::Slot draw_cmd_slot;
      u32 num_double_sided_draws;
    };

    struct DrawPass {
      explicit DrawPass(std::string name, u32 num_single_sided_draws, u32 num_double_sided_draws,
                        u32 frames_in_flight);
      std::string name;
      // lol this is jank but i don't want a string alloc every time. need to make a special hashed
      // string that combines multiple strings at once
      std::string name_double_sided;

      std::array<std::vector<Holder<BufferHandle>>, 2> out_draw_cmds_bufs;
      [[nodiscard]] BufferHandle get_frame_out_draw_cmd_buf_handle(bool double_sided) const;
      [[nodiscard]] vk2::Buffer* get_frame_out_draw_cmd_buf(bool double_sided) const;
      bool enabled{true};
    };
    void add_draw_pass(const std::string& name);

    // TODO: this is a little jank
    Handle add_draws(StateTracker& state, VkCommandBuffer cmd, size_t size, size_t staging_offset,
                     vk2::Buffer& staging, u32 num_double_sided_draws);
    void remove_draws(StateTracker& state, VkCommandBuffer cmd, Handle handle);

    [[nodiscard]] const std::string& get_name() const { return name_; }
    [[nodiscard]] u32 get_num_draw_cmds(bool double_sided) const {
      return num_draw_cmds_[double_sided];
    }
    [[nodiscard]] BufferHandle get_draw_info_buf_handle() const;
    [[nodiscard]] vk2::Buffer* get_draw_info_buf() const;

    bool draw_enabled{true};
    [[nodiscard]] bool should_draw() const {
      return draw_enabled && (get_num_draw_cmds(false) > 0 || get_num_draw_cmds(true) > 0);
    }

    [[nodiscard]] const std::vector<DrawPass>& get_draw_passes() const { return draw_passes_; }

   private:
    std::vector<DrawPass> draw_passes_;
    std::string name_;
    Pool<Handle, Alloc> allocs_;
    FreeListBuffer draw_cmds_buf_;
    u32 num_draw_cmds_[2] = {};  // idx 1 is double sided
  };

  struct StaticModelGPUResources {
    StaticModelGPUResources() = default;
    StaticModelGPUResources(SceneLoadData scene_graph_data,
                            std::vector<gfx::PrimitiveDrawInfo>&& mesh_draw_infos,
                            util::FreeListAllocator::Slot&& materials_slot,
                            util::FreeListAllocator::Slot&& vertices_slot,
                            util::FreeListAllocator::Slot&& indices_slot,
                            std::vector<Holder<ImageHandle>>&& textures,
                            std::vector<Material>&& materials, u64 first_vertex, u64 first_index,
                            u64 num_vertices, u64 num_indices, std::string name, u32 ref_count)
        : scene_graph_data(std::move(scene_graph_data)),
          mesh_draw_infos(std::move(mesh_draw_infos)),
          materials_slot(std::move(materials_slot)),
          vertices_slot(std::move(vertices_slot)),
          indices_slot(std::move(indices_slot)),
          textures(std::move(textures)),
          materials(std::move(materials)),
          first_vertex(first_vertex),
          first_index(first_index),
          num_vertices(num_vertices),
          num_indices(num_indices),
          name(std::move(name)),
          ref_count(ref_count) {}
    StaticModelGPUResources(const StaticModelGPUResources&) = delete;
    StaticModelGPUResources(StaticModelGPUResources&&) = default;
    StaticModelGPUResources& operator=(const StaticModelGPUResources&) = delete;
    StaticModelGPUResources& operator=(StaticModelGPUResources&&) = default;
    ~StaticModelGPUResources();

    SceneLoadData scene_graph_data;
    std::vector<gfx::PrimitiveDrawInfo> mesh_draw_infos;
    util::FreeListAllocator::Slot materials_slot;
    util::FreeListAllocator::Slot vertices_slot;
    util::FreeListAllocator::Slot indices_slot;
    std::vector<Holder<ImageHandle>> textures;
    std::vector<Material> materials;
    u64 first_vertex;
    u64 first_index;
    u64 num_vertices;
    u64 num_indices;
    std::string name;
    u32 ref_count;
  };

  using StaticModelGPUResourcesHandle = GenerationalHandle<StaticModelGPUResources>;
  Pool<StaticModelGPUResourcesHandle, StaticModelGPUResources> static_models_pool_;
  std::unordered_map<std::string, StaticModelGPUResourcesHandle> static_model_name_to_handle_;

  struct StaticModelInstanceResources {
    std::vector<ObjectData> object_datas;
    StaticMeshDrawManager::Handle opaque_draws_handle;
    StaticMeshDrawManager::Handle opaque_alpha_draws_handle;
    StaticMeshDrawManager::Handle transparent_draws_handle;
    util::FreeListAllocator::Slot instance_data_slot;
    util::FreeListAllocator::Slot object_data_slot;
    StaticModelGPUResourcesHandle model_resources_handle;
    // owned by gpu resource
    const char* name;
  };

  std::vector<StaticModelInstanceResources> to_delete_static_model_instances_;
  using StaticModelInstanceResourcesHandle = GenerationalHandle<StaticModelInstanceResources>;
  Pool<StaticModelInstanceResourcesHandle, StaticModelInstanceResources>
      static_model_instance_pool_;
  std::vector<StaticModelInstanceResourcesHandle> loaded_model_instance_resources_;
  void free(StaticModelInstanceResources& instance);
  void free(CmdEncoder& cmd, StaticModelInstanceResources& instance);

  enum GPUDrawInfoFlags : u8 { GPUDrawInfoFlags_DoubleSided = (1 << 0) };

  struct GPUDrawInfo {
    u32 index_cnt;
    u32 first_index;
    u32 vertex_offset;
    u32 instance_id;
    u32 flags;
  };

  gfx::RenderGraph rg_;
  DrawStats static_draw_stats_{};

  std::optional<StaticMeshDrawManager> static_opaque_draw_mgr_;
  std::optional<StaticMeshDrawManager> static_opaque_alpha_mask_draw_mgr_;
  std::optional<StaticMeshDrawManager> static_transparent_draw_mgr_;
  std::vector<mat4> cull_vp_matrices_;
  std::array<StaticMeshDrawManager*, 3> draw_managers_;
  u32 main_mesh_pass_idx_{0};

  bool should_draw(const StaticMeshDrawManager& mgr) const;

  void execute_static_geo_draws(CmdEncoder& cmd);
  void execute_draw(CmdEncoder& cmd, const vk2::Buffer& buffer, u32 draw_count) const;

 public:
  // TODO: refactor IBL so it doesn't need this to draw a cube (needs this for push constants)
  FreeListBuffer static_vertex_buf_;

 private:
  FreeListBuffer static_index_buf_;
  FreeListBuffer static_instance_data_buf_;
  FreeListBuffer static_object_data_buf_;
  FreeListBuffer static_materials_buf_;

  AABB scene_aabb_{};

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
  vk2::PipelineHandle gbuffer_alpha_mask_pipeline_;
  vk2::PipelineHandle deferred_shade_pipeline_;
  vk2::PipelineHandle line_draw_pipeline_;
  Format gbuffer_a_format_{Format::R16G16B16A16Sfloat};
  Format gbuffer_b_format_{Format::R16G16B16A16Sfloat};
  Format gbuffer_c_format_{Format::R16G16B16A16Sfloat};
  Format draw_img_format_{Format::R16G16B16A16Sfloat};
  Format depth_img_format_{Format::D32Sfloat};
  // TODO: make more robust settings
  bool deferred_enabled_{true};
  bool draw_debug_aabbs_{false};
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

  std::vector<std::optional<LoadedSceneData>> loaded_scenes_;
  u32 debug_mode_{DEBUG_MODE_NONE};
  const char* debug_mode_to_string(u32 mode);
  std::filesystem::path env_tex_path_;
  i32 prefilter_mip_skybox_level_{};
  // TODO: enum loser
  bool render_prefilter_mip_skybox_{};
  void make_cubemap_views_all_mips(const vk2::Image& texture,
                                   std::vector<std::optional<vk2::ImageView>>& views);
  void generate_mipmaps(StateTracker& state, VkCommandBuffer cmd, vk2::Image& tex);
  util::FreeListAllocator::Slot cube_vertices_slot_;

  void add_rendering_passes(RenderGraph& rg);
  void copy_buffer(VkCommandBuffer cmd, BufferHandle src, BufferHandle dst, size_t src_offset,
                   size_t dst_offset, size_t size);
  void copy_buffer(VkCommandBuffer cmd, vk2::Buffer* src, vk2::Buffer* dst, size_t src_offset,
                   size_t dst_offset, size_t size);
  // AttachmentInfo swapchain_att_info_;
// TODO: fix
#ifdef __APPLE__
  bool portable_{true};
#else
  bool portable_{false};
#endif
  u32 tonemap_type_{1};
  const char* tonemap_type_names_[2] = {"Optimized Filmic", "ACES Film"};
  struct FrustumCullSettings {
    bool enabled{true};
    bool paused{false};
  } frustum_cull_settings_;
  vec2 near_far_z_{.1, 1000.f};

  struct LineVertex {
    vec4 pos;
    vec4 color;
  };
  std::vector<LineVertex> line_draw_vertices_;
  void draw_skybox(CmdEncoder& cmd);

 public:
  void draw_cube(VkCommandBuffer cmd) const;
  std::optional<vk2::Image> load_hdr_img(CmdEncoder& ctx, const std::filesystem::path& path,
                                         bool flip = false);
  void generate_mipmaps(CmdEncoder& ctx, vk2::Image& tex);

  // TODO: make resource manager load this data on startup
  [[nodiscard]] const DefaultData& get_default_data() const { return default_data_; }

  void draw_line(const vec3& p1, const vec3& p2, const vec4& color);

  /**
   * @brief draws a plane
   *
   * @param o  origin
   * @param v1 first vector the plane lies on
   * @param v2 second vector the plane lies on
   * @param s1 half length horizontal
   * @param s2 half length vertical
   * @param n1 num internal lines horizontal
   * @param n2 num internal lines vertical
   * @param color
   * @param outline_color [TODO:parameter]
   */
  void draw_plane(const vec3& o, const vec3& v1, const vec3& v2, float s1, float s2, u32 n1 = 1,
                  u32 n2 = 1, const vec4& color = vec4{1.f}, const vec4& outline_color = vec4{1.f});

  void draw_box(const mat4& model, const vec3& size, const vec4& color = vec4{1.f});
  void draw_box(const mat4& model, const AABB& aabb, const vec4& color = vec4{1.f});
};

}  // namespace gfx
