#pragma once

#include <vulkan/vulkan_core.h>

#include <filesystem>
#include <optional>
#include <vector>

#include "AABB.hpp"
#include "CommandEncoder.hpp"
#include "RenderGraph.hpp"
#include "Scene.hpp"
#include "SceneLoader.hpp"
#include "SceneResources.hpp"
#include "StateTracker.hpp"
#include "Types.hpp"
#include "shaders/common.h.glsl"
#include "techniques/CSM.hpp"
#include "techniques/IBL.hpp"
#include "util/IndexAllocator.hpp"
#include "vk2/Buffer.hpp"
#include "vk2/Device.hpp"
#include "vk2/PipelineManager.hpp"

struct GLFWwindow;
namespace tracy {
struct VkCtx;
}

struct LoadedInstanceData;
struct LoadedModelData;
struct BlendTreeNode;

namespace gfx {

struct AnimationState;

inline constexpr u32 max_frames_in_flight = 2;
struct ModelGPUResources {
  std::vector<Holder<ImageHandle>> textures;
  std::vector<PrimitiveDrawInfo> mesh_draw_infos;
  util::FreeListAllocator::Slot materials_slot;
  util::FreeListAllocator::Slot vertices_slot;
  util::FreeListAllocator::Slot indices_slot;
  util::FreeListAllocator::Slot animated_vertices_gpu_slot;
  std::vector<Material> materials;
  u64 first_vertex;
  u64 first_index;
  u64 num_vertices;
  u64 num_indices;
  std::string name;
  u32 ref_count;
};

enum MeshPass : u8 {
  MeshPass_Opaque,
  MeshPass_OpaqueDoubleSided,
  MeshPass_OpaqueAlphaMask,
  MeshPass_OpaqueAlphaMaskDoubleSided,
  MeshPass_Transparent,
  MeshPass_TransparentDoubleSided,
  MeshPass_Count
};

CullMode get_cull_mode(MeshPass pass);
MeshPass get_double_sided_pass(MeshPass pass);

std::string to_string(MeshPass p);

struct GPUInstanceData {
  u32 material_id;
  u32 instance_id;
  u32 flags;
};
struct StaticModelInstanceResources {
  std::vector<ObjectData> object_datas;
  std::vector<GPUInstanceData> instance_datas;
  std::vector<int> node_to_instance_and_obj;
  std::array<u32, MeshPass_Count> mesh_pass_draw_handles{UINT32_MAX};
  util::FreeListAllocator2::Slot instance_data_slot;
  util::FreeListAllocator2::Slot object_data_slot;
  util::FreeListAllocator2::Slot animated_vertex_buf_slot;
  util::FreeListAllocator2::Slot global_bone_mat_slot;
  util::FreeListAllocator2::Slot skin_commands_slot;
  ModelHandle model_handle;
  const char* name;  // owned by gpu resource
  bool is_animated{};
};

using ModelGPUResourceHandle = GenerationalHandle<ModelGPUResources>;

struct LinearCopyer {
  explicit LinearCopyer(void* data, u64 capacity) : capacity_(capacity), data_(data) {}
  LinearCopyer() = default;

  [[nodiscard]] u64 copy(const void* data, u64 size);
  void reset() { offset_ = 0; }

 private:
  u64 capacity_{};
  u64 offset_{};
  void* data_{};
};

struct LinearStagingBuffer {
  explicit LinearStagingBuffer(Buffer* buffer) : buffer_(buffer) {}

  u64 copy(const void* data, u64 size) {
    memcpy((char*)buffer_->mapped_data() + offset_, data, size);
    offset_ += size;
    return offset_ - size;
  }
  [[nodiscard]] Buffer* get_buffer() const { return buffer_; }

 private:
  u64 offset_{};
  Buffer* buffer_{};
};

struct FreeListBuffer2 {
  Holder<BufferHandle> buffer;
  util::FreeListAllocator2 allocator;
  [[nodiscard]] Buffer* get_buffer() const { return get_device().get_buffer(buffer); }
};

struct FreeListBuffer {
  Holder<BufferHandle> buffer;
  util::FreeListAllocator allocator;
  [[nodiscard]] Buffer* get_buffer() const;
};

template <u32 N>
struct FreeListNBuffers {
  std::array<Holder<BufferHandle>, N> buffers;
  util::FreeListAllocator2 allocator;
};

struct SceneDrawInfo {
  mat4 view;
  float fov_degees;
  vec3 view_pos;
  vec3 light_dir;
  vec3 light_color;
  float ambient_intensity{0.1};
  float fov_degrees{70.f};
};

AABB transform_aabb(const glm::mat4& model, const AABB& aabb);

class VkRender2 final {
 public:
  struct InitInfo {
    GLFWwindow* window;
    Device* device;
    std::filesystem::path resource_dir;
    const char* name = "App";
    bool vsync{true};
  };
  static VkRender2& get();
  static void init(const InitInfo& info, bool& success);
  static void shutdown();
  explicit VkRender2(const InitInfo& info, bool& succes);
  ~VkRender2();

  bool load_model2(const std::filesystem::path& path, LoadedModelData& result);
  StaticModelInstanceResourcesHandle add_instance(ModelHandle model_handle);
  void update_transforms(LoadedInstanceData& instance, std::vector<i32>& changed_nodes);
  void update_animation(LoadedInstanceData& instance, float dt);
  void draw_joints(LoadedInstanceData& instance);
  bool update_skins(LoadedInstanceData& instance);
  void remove_instance(StaticModelInstanceResourcesHandle handle);
  void mark_dirty(InstanceHandle handle);

  Pool<ModelGPUResourceHandle, ModelGPUResources> model_gpu_resources_pool_;

  // TODO: private
  void immediate_submit(std::function<void(CmdEncoder& cmd)>&& function);
  void enqueue_transfer();
  void set_env_map(const std::filesystem::path& path);
  void draw(const SceneDrawInfo& info);
  void new_frame();
  void set_imgui_enabled(bool imgui_enabled) { draw_imgui_ = imgui_enabled; }
  [[nodiscard]] bool get_imgui_enabled() const { return draw_imgui_; }
  ImageHandle load_hdr_img(const std::filesystem::path& path, bool flip = false);
  void generate_mipmaps(CmdEncoder& ctx, ImageHandle handle);
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
  void draw_sphere(const glm::vec3& center, float radius, const glm::vec4& color,
                   int segments = 12);

 private:
  Device* device_;
  GLFWwindow* window_;
  std::filesystem::path resource_dir_;

  struct PerFrameData {
    // VkCommandPool cmd_pool;
    // VkCommandBuffer main_cmd_buffer;
    Holder<BufferHandle> scene_uniform_buf;
    Holder<BufferHandle> line_draw_buf;
  };
  std::vector<PerFrameData> per_frame_data_;
  [[nodiscard]] u64 curr_frame_in_flight_num() const;
  void on_imgui();
  static constexpr u32 max_draws{100'000};

  struct SceneUniforms {
    mat4 view_proj;
    mat4 view;
    mat4 proj;
    mat4 inverse_view_proj;
    mat4 inverse_proj;
    uvec4 debug_flags;
    vec3 view_pos;
    float _p1;
    vec3 light_dir;
    float _p2;
    vec3 light_color;
    float ambient_intensity;
  };
  SceneUniforms scene_uniform_cpu_data_;

  // animated models:
  // animated vertices: all one buffer
  // output vertex buffers: per frame in flight
  // bone matrix buffers: per frame in flight
  FreeListBuffer animated_vertex_buf_;
  // per frame in flight
  FreeListNBuffers<max_frames_in_flight> animated_vertex_output_bufs_;
  // per frame in flight
  std::vector<Holder<BufferHandle>> bone_matrix_bufs_;
  FreeListBuffer animated_instance_data_buf_;
  FreeListBuffer animated_object_data_buf_;

  struct SkinCommand {
    u32 skin_vtx_i;
    u32 out_vtx_i;
    u32 bone_mat_start_i;
  };
  FreeListBuffer2 skin_instance_datas_;

  util::FreeListAllocator2 global_skin_mat_allocator_;
  std::vector<mat4> global_skin_matrices_;

  FreeListBuffer static_vertex_buf_;
  FreeListBuffer static_index_buf_;
  FreeListBuffer2 static_instance_data_buf_;
  FreeListBuffer2 static_object_data_buf_;
  FreeListBuffer static_materials_buf_;

  struct DrawStats {
    u64 total_vertices;
    u64 total_indices;
    u32 vertices;
    u32 animated_vertices;
    u32 indices;
    u32 textures;
    u32 materials;
  };

  std::array<u32, 4> opaque_mesh_pass_idxs_{MeshPass_Opaque, MeshPass_OpaqueDoubleSided,
                                            MeshPass_OpaqueAlphaMask,
                                            MeshPass_OpaqueAlphaMaskDoubleSided};

  struct StaticMeshDrawManager {
    static constexpr u32 null_handle{UINT32_MAX};
    StaticMeshDrawManager() = default;
    void init(MeshPass type, size_t initial_max_draw_cnt, Device* device, const std::string& name);
    StaticMeshDrawManager(const StaticMeshDrawManager&) = delete;
    StaticMeshDrawManager(StaticMeshDrawManager&&) = delete;
    StaticMeshDrawManager& operator=(const StaticMeshDrawManager&) = delete;
    StaticMeshDrawManager& operator=(StaticMeshDrawManager&&) = delete;

    struct Alloc {
      util::FreeListAllocator2::Slot draw_cmd_slot;
    };

    struct DrawPass {
      explicit DrawPass(u32 num_draws, u32 frames_in_flight, Device* device);
      std::vector<Holder<BufferHandle>> out_draw_cmds_bufs;
      [[nodiscard]] BufferHandle get_frame_out_draw_cmd_buf_handle() const;
      [[nodiscard]] Buffer* get_frame_out_draw_cmd_buf() const;
      Device* device_;
      bool enabled{true};
    };
    [[nodiscard]] u32 add_draw_pass();

    // TODO: this is a little jank
    u32 add_draws(StateTracker& state, size_t size, size_t staging_offset);
    void remove_draws(StateTracker& state, CmdEncoder& cmd, u32 handle);

    [[nodiscard]] const std::string& get_name() const { return name_; }
    [[nodiscard]] u32 get_num_draw_cmds() const { return num_draw_cmds_; }
    [[nodiscard]] BufferHandle get_draw_info_buf_handle() const;
    [[nodiscard]] Buffer* get_draw_info_buf() const;

    bool draw_enabled{true};
    [[nodiscard]] bool should_draw() const { return draw_enabled && get_num_draw_cmds() > 0; }

    [[nodiscard]] const std::vector<DrawPass>& get_draw_passes() const { return draw_passes_; }
    [[nodiscard]] const DrawPass& get_draw_pass(u32 idx) const { return draw_passes_[idx]; }
    [[nodiscard]] MeshPass get_mesh_pass() const { return mesh_pass_; }

   private:
    std::vector<DrawPass> draw_passes_;
    std::string name_;
    std::vector<Alloc> allocs_;
    std::vector<u32> free_alloc_indices_;
    FreeListBuffer2 draw_cmds_buf_;
    FreeListBuffer2 animated_draw_cmds_buf_;
    u32 num_draw_cmds_{};
    Device* device_{};
    MeshPass mesh_pass_{MeshPass_Count};
  };

  std::array<u32, MeshPass_Count> main_view_mesh_pass_indices_;
  std::vector<std::array<u32, MeshPass_Count>> shadow_mesh_pass_indices_;
  std::vector<StaticModelInstanceResourcesHandle> to_delete_static_model_instances_;

 public:
  Pool<StaticModelInstanceResourcesHandle, StaticModelInstanceResources>
      static_model_instance_pool_;

 private:
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

  DrawStats draw_stats_{};

  std::array<StaticMeshDrawManager, MeshPass_Count> static_draw_mgrs_;
  std::array<StaticMeshDrawManager, MeshPass_Count> animated_draw_mgrs_;
  StaticMeshDrawManager& get_mgr(MeshPass pass, bool is_animated) {
    return is_animated ? animated_draw_mgrs_[pass] : static_draw_mgrs_[pass];
  }

  std::vector<mat4> cull_vp_matrices_;

  [[nodiscard]] bool should_draw(const StaticMeshDrawManager& mgr) const;
  void execute_static_geo_draws(CmdEncoder& cmd, MeshPass pass, bool is_animated = false);
  void execute_draw(CmdEncoder& cmd, BufferHandle buffer, u32 draw_count) const;

  AABB scene_aabb_{};

  StateTracker state_;

  SamplerHandle linear_sampler_;
  SamplerHandle nearest_sampler_;
  SamplerHandle linear_sampler_clamp_to_edge_;

  struct DefaultData {
    Holder<ImageHandle> white_img;
  } default_data_;
  gfx::DefaultMaterialData default_mat_data_;

  BufferCopyer object_data_buffer_copier_;
  std::vector<InstanceHandle> dirty_instances_;
  std::vector<ObjectData> object_datas_to_copy_;
  std::vector<Holder<BufferHandle>> frame_staging_buffers_;
  std::vector<LinearCopyer> staging_buffer_copiers_;
  LinearCopyer& get_staging_copyer() {
    return staging_buffer_copiers_[device_->curr_frame_in_flight()];
  }

  PipelineTask make_pipeline_task(const ComputePipelineCreateInfo& info,
                                  PipelineHandle* out_handle);
  PipelineTask make_pipeline_task(const GraphicsPipelineCreateInfo& info,
                                  PipelineHandle* out_handle);
  std::filesystem::path default_env_map_path_;
  std::unique_ptr<CSM> csm_;
  SamplerHandle shadow_sampler_;
  std::optional<IBL> ibl_;
  RenderGraph rg_;
  PipelineHandle img_pipeline_;
  PipelineHandle draw_pipeline_;
  PipelineHandle cull_objs_pipeline_;
  PipelineHandle skybox_pipeline_;
  PipelineHandle postprocess_pipeline_;
  PipelineHandle gbuffer_pipeline_;
  PipelineHandle gbuffer_alpha_mask_pipeline_;
  PipelineHandle deferred_shade_pipeline_;
  PipelineHandle line_draw_pipeline_;
  PipelineHandle transparent_oit_pipeline_;
  PipelineHandle oit_comp_pipeline_;
  PipelineHandle skinning_comp_pipeline_;
  PipelineHandle ssao_1_pipeline_;
  PipelineHandle ssao_blur_pipeline_;
  PipelineHandle fxaa_pipeline_;
  PipelineHandle up_down_sample_pipeline_;
  Format gbuffer_a_format_{Format::R8G8B8A8Unorm};
  Format gbuffer_b_format_{Format::R8G8B8A8Unorm};
  Format gbuffer_c_format_{Format::R8G8B8A8Unorm};
  Format ssao_format_{Format::R32Sfloat};
  Format draw_img_format_{Format::R16G16B16A16Sfloat};
  Format depth_img_format_{Format::D32Sfloat};
  Holder<BufferHandle> ssao_noise_buf_;
  Holder<BufferHandle> ssao_kernel_buf_;

  std::vector<Buffer> free_staging_buffers_;

  std::vector<CmdEncoder*> frame_imm_submits_;
  std::vector<std::optional<LoadedSceneData>> loaded_scenes_;
  u32 debug_mode_{DEBUG_MODE_NONE};
  const char* debug_mode_to_string(u32 mode);
  std::filesystem::path env_tex_path_;
  i32 prefilter_mip_skybox_render_mip_level_{1};
  void generate_mipmaps(StateTracker& state, CmdEncoder& cmd, ImageHandle handle);
  Holder<BufferHandle> cube_vertex_buf_;
  void add_rendering_passes(RenderGraph& rg);
  void init_ssao();

  u32 tonemap_type_{1};
  const char* tonemap_type_names_[2] = {"Optimized Filmic", "ACES Film"};
  struct FrustumCullSettings {
    bool enabled{false};
    bool paused{false};
  } frustum_cull_settings_;
  vec2 near_far_z_{.1, 10000.f};

  struct LineVertex {
    vec4 pos;
    vec4 color;
  };
  std::vector<LineVertex> line_draw_vertices_;
  void draw_skybox(CmdEncoder& cmd);

  [[nodiscard]] float aspect_ratio() const;
  [[nodiscard]] uvec2 window_dims() const;
  PerFrameData& curr_frame();

  bool render_prefilter_mip_skybox_{true};
  bool draw_imgui_{true};
  bool draw_debug_aabbs_{false};

  // OIT things
  u32 max_oit_fragments_{};
  Holder<BufferHandle> oit_fragment_buffer_;
  Holder<BufferHandle> oit_atomic_counter_buf_;
  Holder<ImageHandle> oit_heads_tex_;
  bool oit_enabled_{true};
  bool oit_debug_heatmap_{false};
  float oit_opacity_boost_{0.0};

  // TODO: settings config file
  bool ssao_enabled_{true};
  bool ssao_blur_enabled_{true};
  bool fxaa_enabled_{true};
  struct FxaaSettings {
    float fixed_threshold{fixed_thresh_range.x};
    float relative_threshold{rel_thresh_range.x};
    float subpixel_blending{0.75f};
    constexpr static float default_fixed_threshold{0.0833f};
    constexpr static float default_relative_threshold{0.125f};
    constexpr static vec2 rel_thresh_range{0.063f, 0.333f};
    constexpr static vec2 fixed_thresh_range{0.0312f, 0.0833f};
  } fxaa_settings_;
  float render_scale_{1.0};
};

}  // namespace gfx
