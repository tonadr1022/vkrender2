#pragma once

#include <vulkan/vulkan_core.h>

#include <filesystem>
#include <optional>
#include <queue>

#include "BaseRenderer.hpp"
#include "Scene.hpp"
#include "SceneLoader.hpp"
#include "StateTracker.hpp"
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

struct VkRender2 final : public BaseRenderer {
  static VkRender2& get();
  static void init(const InitInfo& info);
  static void shutdown();
  explicit VkRender2(const InitInfo& info);
  ~VkRender2() override;

  SceneHandle load_scene(const std::filesystem::path& path, bool dynamic = false);
  void submit_static(SceneHandle scene, mat4 transform = mat4{1});

  // TODO: private
  void immediate_submit(std::function<void(VkCommandBuffer cmd)>&& function);
  [[nodiscard]] const QueueFamilies& get_queues() const { return queues_; }

 private:
  template <typename T>
  struct InFlightResource {
    T data;
    VkFence fence{};
  };

  void on_draw(const SceneDrawInfo& info) override;
  void on_gui() override;
  void on_resize() override;
  void create_attachment_imgs();
  void set_viewport_and_scissor(VkCommandBuffer cmd, VkExtent2D extent);

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

  struct SceneUniforms {
    mat4 view_proj;
    uvec4 debug_flags;
    vec3 view_pos;
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

  u64 draw_cnt_{};
  std::optional<LinearBuffer> static_vertex_buf_;
  std::optional<LinearBuffer> static_index_buf_;
  std::optional<LinearBuffer> static_materials_buf_;
  std::optional<LinearBuffer> static_material_indices_buf_;
  std::optional<LinearBuffer> static_draw_cmds_buf_;
  std::optional<LinearBuffer> static_transforms_buf_;
  std::vector<vk2::Texture> static_textures_;

  StateTracker state_;
  StateTracker transfer_q_state_;
  std::optional<vk2::Texture> depth_img_;
  std::optional<vk2::Texture> img_;
  std::optional<vk2::Sampler> linear_sampler_;
  struct DefaultData {
    std::optional<vk2::Texture> white_img;
  } default_data_;
  gfx::DefaultMaterialData default_mat_data_;

  vk2::DeletionQueue main_del_q_;
  std::filesystem::path shader_dir_;
  [[nodiscard]] std::string get_shader_path(const std::string& path) const;
  vk2::PipelineHandle img_pipeline_;
  vk2::PipelineHandle draw_pipeline_;
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
 public:
  [[nodiscard]] const DefaultData& get_default_data() const { return default_data_; }
};
