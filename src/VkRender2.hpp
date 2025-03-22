#pragma once

#include <vulkan/vulkan_core.h>

#include <filesystem>
#include <optional>
#include <queue>

#include "BaseRenderer.hpp"
#include "Scene.hpp"
#include "StateTracker.hpp"
#include "vk2/Buffer.hpp"
#include "vk2/DeletionQueue.hpp"
#include "vk2/PipelineManager.hpp"
#include "vk2/SamplerCache.hpp"
#include "vk2/Texture.hpp"

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

VK2_DEFINE_HANDLE(Scene);

struct VkRender2 final : public BaseRenderer {
  static VkRender2& get();
  static void init(const InitInfo& info);
  static void shutdown();
  explicit VkRender2(const InitInfo& info);
  ~VkRender2() override;

  SceneHandle load_scene(const std::filesystem::path& path);
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

  struct InstanceData {
    mat4 transform;
    u32 material_id;
  };

  // TODO: refactor
  struct SceneGPUResources {
    vk2::Buffer vertex_buffer;
    vk2::Buffer index_buffer;
    vk2::Buffer materials_buffer;
    vk2::Buffer draw_indirect_buffer;
    vk2::Buffer instance_buffer;
    // std::vector<u32> material_indices;
    std::vector<vk2::Sampler> samplers;
    std::vector<vk2::Texture> textures;
    u32 draw_cnt{};
  };

  struct LoadedScene {
    SceneLoadData scene_graph_data;
    std::unique_ptr<SceneGPUResources> resources;
  };

  std::vector<LoadedScene> loaded_scenes_;

  VkCommandPool imm_cmd_pool_;
  VkCommandBuffer imm_cmd_buf_;
  // VkFence imm_fence_;

  StateTracker state_;
  StateTracker transfer_q_state_;
  std::optional<vk2::Texture> depth_img_;
  std::optional<vk2::Texture> img_;
  struct DefaultData {
    std::optional<vk2::Texture> white_img;
  } default_data_;

  vk2::DeletionQueue main_del_q_;
  std::filesystem::path shader_dir_;
  [[nodiscard]] std::string get_shader_path(const std::string& path) const;
  vk2::PipelineHandle img_pipeline_;
  vk2::PipelineHandle draw_pipeline_;
  VkPipelineLayout default_pipeline_layout_{};
  std::queue<InFlightResource<vk2::Buffer*>> pending_buffer_transfers_;

  std::vector<vk2::Buffer> free_staging_buffers_;

  // non owning
  VkDescriptorSet main_set_{};
  VkDescriptorSet main_set2_{};
  VmaAllocator allocator_;
  // end non owning
 public:
  [[nodiscard]] const DefaultData& get_default_data() const { return default_data_; }
};
