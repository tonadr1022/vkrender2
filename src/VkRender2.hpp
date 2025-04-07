#pragma once

#include <vulkan/vulkan_core.h>

#include <filesystem>
#include <optional>
#include <queue>
#include <vector>

#include "BaseRenderer.hpp"
#include "Scene.hpp"
#include "SceneLoader.hpp"
#include "StateTracker.hpp"
#include "shaders/common.h.glsl"
#include "techniques/CSM.hpp"
#include "util/IndexAllocator.hpp"
#include "vk2/Buffer.hpp"
#include "vk2/DeletionQueue.hpp"
#include "vk2/PipelineManager.hpp"
#include "vk2/SamplerCache.hpp"
#include "vk2/Texture.hpp"

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
struct CmdEncoder {
  explicit CmdEncoder(VkCommandBuffer cmd) : cmd_(cmd) {}
  void dispatch(u32 work_groups_x, u32 work_groups_y, u32 work_groups_z);
  void bind_compute_pipeline(VkPipeline pipeline);
  void bind_descriptor_set(VkPipelineBindPoint bind_point, VkPipelineLayout layout,
                           VkDescriptorSet* set, u32 idx);
  void push_constants(VkPipelineLayout layout, u32 size, void* data);

  [[nodiscard]] VkCommandBuffer cmd() const { return cmd_; }

 private:
  VkCommandBuffer cmd_;
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
  void transfer_submit(std::function<void(VkCommandBuffer cmd, VkFence fence,
                                          std::queue<InFlightResource<vk2::Buffer*>>&)>&& function);
  [[nodiscard]] const QueueFamilies& get_queues() const { return queues_; }

 private:
  void on_draw(const SceneDrawInfo& info) override;
  void on_gui() override;
  void on_resize() override;
  void create_attachment_imgs();

  // TODO: refactor
  struct SceneGPUResources {
    vk2::Buffer vertex_buffer;
    vk2::Buffer index_buffer;
    vk2::Buffer materials_buffer;
    vk2::Buffer draw_indirect_buffer;
    vk2::Buffer material_indices;
    vk2::Buffer instance_buffer;
    std::vector<vk2::Texture> textures;
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
  };
  std::vector<FrameData> per_frame_data_2_;
  FrameData& curr_frame_2() { return per_frame_data_2_[curr_frame_num() % 2]; }
  void init_pipelines();
  void init_indirect_drawing();
  void init_ibl();
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
    std::vector<gfx::PrimitiveDrawInfo> mesh_draw_infos;
    std::vector<util::SlotAllocator<gfx::ObjectData>::Slot> object_data_slots;
    u64 first_vertex;
    u64 first_index;
    u64 num_vertices;
    u64 num_indices;
    u64 materials_idx_offset;
  };
  struct DrawInfo {
    u32 index_cnt;
    u32 first_index;
    u32 vertex_offset;
    u32 pad;
  };

  struct ObjectDraw {
    std::vector<util::SlotAllocator<DrawInfo>::Slot> draw_cmd_slots;
    std::vector<util::SlotAllocator<gfx::ObjectData>::Slot> obj_data_slots;
  };
  // std::vector<ObjectDraw> static_draws_;

  std::unordered_map<SceneHandle, ObjectDraw> active_static_draws_;
  std::unordered_map<std::string, StaticSceneGPUResources> static_scenes_;

  DrawStats static_draw_stats_{};
  std::optional<LinearBuffer> static_vertex_buf_;
  std::optional<LinearBuffer> static_index_buf_;
  std::optional<LinearBuffer> static_materials_buf_;
  std::optional<LinearBuffer> static_instance_data_buf_;

  std::optional<SlotBuffer<DrawInfo>> static_draw_info_buf_;
  std::optional<SlotBuffer<gfx::ObjectData>> static_object_data_buf_;
  std::vector<vk2::Texture> static_textures_;
  std::optional<vk2::Buffer> final_draw_cmd_buf_;
  std::optional<vk2::Buffer> draw_cnt_buf_;

  StateTracker state_;
  StateTracker transfer_q_state_;
  std::optional<vk2::Texture> depth_img_;
  std::optional<vk2::Texture> img_;
  std::optional<vk2::Texture> post_processed_img_;

  std::optional<vk2::Sampler> linear_sampler_;
  struct DefaultData {
    std::optional<vk2::Texture> white_img;
  } default_data_;
  gfx::DefaultMaterialData default_mat_data_;
  struct InstanceData {
    u32 material_id;
    u32 instance_id;
  };

  std::optional<CSM> csm_;
  vk2::DeletionQueue main_del_q_;
  vk2::PipelineHandle img_pipeline_;
  vk2::PipelineHandle draw_pipeline_;
  vk2::PipelineHandle cull_objs_pipeline_;
  vk2::PipelineHandle equirect_to_cube_pipeline_;
  vk2::PipelineHandle equirect_to_cube_pipeline2_;
  vk2::PipelineHandle skybox_pipeline_;
  vk2::PipelineHandle convolute_cube_pipeline_;
  vk2::PipelineHandle convolute_cube_raster_pipeline_;
  vk2::PipelineHandle postprocess_pipeline_;
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
  std::optional<vk2::Texture> load_hdr_img(const std::filesystem::path& path, bool flip = false);
  std::filesystem::path env_tex_path_;
  std::optional<vk2::Texture> env_equirect_tex_;
  std::optional<vk2::Texture> env_cubemap_tex_;
  std::optional<vk2::Texture> convoluted_cubemap_tex_;
  std::array<std::optional<vk2::TextureView>, 6> cubemap_tex_views_;
  std::array<std::optional<vk2::TextureView>, 6> convoluted_cubemap_tex_views_;
  u64 cube_vertices_gpu_offset_{};
  // u64 cube_indices_gpu_offset_{};
  // std::optional<vk2::Buffer> cube_vertex_buf_;
  // std::optional<vk2::Buffer> cube_index_buf_;

 public:
  [[nodiscard]] const DefaultData& get_default_data() const { return default_data_; }
};
