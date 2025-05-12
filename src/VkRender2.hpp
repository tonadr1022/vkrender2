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
#include "StateTracker.hpp"
#include "Types.hpp"
#include "shaders/common.h.glsl"
#include "techniques/CSM.hpp"
#include "techniques/IBL.hpp"
#include "util/IndexAllocator.hpp"
#include "vk2/Buffer.hpp"
#include "vk2/PipelineManager.hpp"
#include "vk2/Texture.hpp"

struct GLFWwindow;
namespace tracy {
struct VkCtx;
}

namespace gfx {

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

struct FreeListBuffer {
  Holder<BufferHandle> buffer;
  util::FreeListAllocator allocator;
  [[nodiscard]] Buffer* get_buffer() const;
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

  ModelHandle load_model(const std::filesystem::path& path, bool dynamic = false,
                         const mat4& transform = mat4{1});

  // TODO: private
  void immediate_submit(std::function<void(VkCommandBuffer cmd)>&& function);
  void immediate_submit(std::function<void(CmdEncoder& cmd)>&& function);
  void enqueue_transfer();
  void set_env_map(const std::filesystem::path& path);
  void draw(const SceneDrawInfo& info);
  void new_frame();
  void set_imgui_enabled(bool imgui_enabled) { draw_imgui_ = imgui_enabled; }
  [[nodiscard]] bool get_imgui_enabled() const { return draw_imgui_; }
  ImageHandle load_hdr_img(CmdEncoder& ctx, const std::filesystem::path& path, bool flip = false);
  void generate_mipmaps(CmdEncoder& ctx, Image& tex);
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

 private:
  Device* device_;
  GLFWwindow* window_;
  std::filesystem::path resource_dir_;

  struct PerFrameData {
    VkCommandPool cmd_pool;
    VkCommandBuffer main_cmd_buffer;
    tracy::VkCtx* tracy_vk_ctx{};
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

  FreeListBuffer static_vertex_buf_;
  FreeListBuffer static_index_buf_;
  FreeListBuffer static_instance_data_buf_;
  FreeListBuffer static_object_data_buf_;
  FreeListBuffer static_materials_buf_;

  struct DrawStats {
    u64 total_vertices;
    u64 total_indices;
    u32 vertices;
    u32 indices;
    u32 textures;
    u32 materials;
  };

  enum MeshPass : u8 {
    MeshPass_Opaque,
    MeshPass_OpaqueAlphaMask,
    MeshPass_Transparent,
    MeshPass_Count
  };
  std::array<u32, 2> opaque_mesh_pass_idxs_{MeshPass_Opaque, MeshPass_OpaqueAlphaMask};

  struct StaticMeshDrawManager {
    using Handle = GenerationalHandle<struct Alloc>;
    StaticMeshDrawManager() = default;
    void init(MeshPass type, size_t initial_max_draw_cnt, Device* device);
    StaticMeshDrawManager(const StaticMeshDrawManager&) = delete;
    StaticMeshDrawManager(StaticMeshDrawManager&&) = delete;
    StaticMeshDrawManager& operator=(const StaticMeshDrawManager&) = delete;
    StaticMeshDrawManager& operator=(StaticMeshDrawManager&&) = delete;

    struct Alloc {
      util::FreeListAllocator::Slot draw_cmd_slot;
      u32 num_double_sided_draws;
    };

    struct DrawPass {
      explicit DrawPass(u32 num_single_sided_draws, u32 num_double_sided_draws,
                        u32 frames_in_flight, Device* device);

      std::array<std::vector<Holder<BufferHandle>>, 2> out_draw_cmds_bufs;
      [[nodiscard]] BufferHandle get_frame_out_draw_cmd_buf_handle(bool double_sided) const;
      [[nodiscard]] Buffer* get_frame_out_draw_cmd_buf(bool double_sided) const;
      Device* device_;
      bool enabled{true};
    };
    [[nodiscard]] u32 add_draw_pass();

    // TODO: this is a little jank
    Handle add_draws(StateTracker& state, CmdEncoder& cmd, size_t size, size_t staging_offset,
                     Buffer& staging, u32 num_double_sided_draws);
    void remove_draws(StateTracker& state, VkCommandBuffer cmd, Handle handle);

    [[nodiscard]] const std::string& get_name() const { return name_; }
    [[nodiscard]] u32 get_num_draw_cmds(bool double_sided) const {
      return num_draw_cmds_[double_sided];
    }
    [[nodiscard]] BufferHandle get_draw_info_buf_handle() const;
    [[nodiscard]] Buffer* get_draw_info_buf() const;

    bool draw_enabled{true};
    [[nodiscard]] bool should_draw() const {
      return draw_enabled && (get_num_draw_cmds(false) > 0 || get_num_draw_cmds(true) > 0);
    }

    [[nodiscard]] const std::vector<DrawPass>& get_draw_passes() const { return draw_passes_; }
    [[nodiscard]] const DrawPass& get_draw_pass(u32 idx) const { return draw_passes_[idx]; }
    [[nodiscard]] MeshPass get_mesh_pass() const { return mesh_pass_; }

   private:
    std::vector<DrawPass> draw_passes_;
    std::string name_;
    Pool<Handle, Alloc> allocs_;
    FreeListBuffer draw_cmds_buf_;
    u32 num_draw_cmds_[2] = {};  // idx 1 is double sided
    Device* device_{};
    MeshPass mesh_pass_{MeshPass_Count};
  };

  std::array<u32, MeshPass_Count> main_view_mesh_pass_indices_;
  std::vector<std::array<u32, MeshPass_Count>> shadow_mesh_pass_indices_;

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
    std::array<StaticMeshDrawManager::Handle, MeshPass_Count> mesh_pass_draw_handles;
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

  DrawStats static_draw_stats_{};

  std::array<StaticMeshDrawManager, MeshPass_Count> static_draw_mgrs_;
  std::vector<mat4> cull_vp_matrices_;

  [[nodiscard]] bool should_draw(const StaticMeshDrawManager& mgr) const;
  void execute_static_geo_draws(CmdEncoder& cmd, bool double_sided, MeshPass pass);
  void execute_draw(CmdEncoder& cmd, const Buffer& buffer, u32 draw_count) const;

  AABB scene_aabb_{};

  StateTracker state_;

  SamplerHandle linear_sampler_;
  SamplerHandle nearest_sampler_;
  SamplerHandle linear_sampler_clamp_to_edge_;

  struct DefaultData {
    Holder<ImageHandle> white_img;
  } default_data_;
  gfx::DefaultMaterialData default_mat_data_;

  struct GPUInstanceData {
    u32 material_id;
    u32 instance_id;
  };

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
  Format gbuffer_a_format_{Format::R8G8B8A8Unorm};
  Format gbuffer_b_format_{Format::R8G8B8A8Unorm};
  Format gbuffer_c_format_{Format::R8G8B8A8Unorm};
  Format draw_img_format_{Format::R16G16B16A16Sfloat};
  Format depth_img_format_{Format::D32Sfloat};

  std::vector<Buffer> free_staging_buffers_;

  std::unordered_map<u64, VkDescriptorSet> imgui_desc_sets_;
  VkDescriptorSet get_imgui_set(VkSampler sampler, VkImageView view);
  // non owning
  // VkDescriptorSet main_set_{};
  // VkDescriptorSet main_set2_{};
  // end non owning

  std::vector<std::optional<LoadedSceneData>> loaded_scenes_;
  u32 debug_mode_{DEBUG_MODE_NONE};
  const char* debug_mode_to_string(u32 mode);
  std::filesystem::path env_tex_path_;
  i32 prefilter_mip_skybox_render_mip_level_{1};
  void generate_mipmaps(StateTracker& state, VkCommandBuffer cmd, Image& tex);
  Holder<BufferHandle> cube_vertex_buf_;
  void add_rendering_passes(RenderGraph& rg);

  // AttachmentInfo swapchain_att_info_;

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
  void render_imgui(CmdEncoder& cmd, uvec2 draw_extent, ImageViewHandle target_img_view);

  [[nodiscard]] float aspect_ratio() const;
  [[nodiscard]] uvec2 window_dims() const;
  PerFrameData& curr_frame();

  bool render_prefilter_mip_skybox_{true};
  bool draw_imgui_{true};
  bool deferred_enabled_{true};
  bool draw_debug_aabbs_{false};
};

}  // namespace gfx
