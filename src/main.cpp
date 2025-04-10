
#include <cstring>
#include <print>

#include "App.hpp"
#include "Logger.hpp"
#include "RenderGraph.hpp"
#include "Types.hpp"
#include "VkRender2.hpp"

#define CMP(arg, cmp) strcmp(arg, cmp) == 0

using vr = std::expected<void, const char*>;
int main(int argc, char* argv[]) {
  gfx::RenderGraph rg{};
  rg.set_backbuffer_img("final_out");
  rg.set_swapchain_info(gfx::RenderGraphSwapchainInfo{.width = 1600, .height = 900});

  auto& forward = rg.add_pass("forward", [](CmdEncoder&) { LINFO("executing forward"); });
  forward.add_color_output("forward_output", {.format = gfx::Format::R32G32B32A32Sfloat});
  forward.set_depth_stencil_output("depth", {.format = gfx::Format::D32Sfloat});

  auto& pp = rg.add_pass("postprocess", [](CmdEncoder&) { LINFO("executing postprocess"); });
  pp.add_color_output("postprocessout", {.format = gfx::Format::R8G8B8A8Unorm});
  pp.add_texture_input("forward_output");

  auto& pp2 = rg.add_pass("pp2", [](CmdEncoder&) {});
  pp2.add_texture_input("postprocessout");
  pp2.add_color_output("final_out", {});

  // auto& imgui = rg.add_pass("imgui", [](CmdEncoder&) {});
  // imgui.add_color_output("final_out", {});

  auto res = rg.bake();
  if (!res) {
    LERROR("bake error {}", res.error());
    exit(1);
  }
  std::filesystem::path graph_dir{"graphs"};
  if (!std::filesystem::exists(graph_dir)) {
    std::filesystem::create_directory(graph_dir);
  }
  auto out_dot_path = graph_dir / "graph.dot";
  auto out_svg_path = graph_dir / "graph.svg";
  res = rg.output_graphvis(out_dot_path);
  if (res) {
    std::system(
        std::format("dot -Tsvg {} -o {}", out_dot_path.string(), out_svg_path.string()).c_str());
  }

  exit(0);

  u32 w{800}, h{800};
  bool vsync{true}, maximize{false};
  for (int i = 1; i < argc; i++) {
    char* arg = argv[i];
    if (CMP(arg, "-w") && i < argc - 1) {
      w = std::stoi(argv[i + 1]);
    } else if (CMP(arg, "-h") && i < argc - 1) {
      h = std::stoi(argv[i + 1]);
    } else if (CMP(arg, "--no-vsync")) {
      vsync = false;
    } else if (CMP(arg, "--maximize") || CMP(arg, "-m")) {
      maximize = true;
    }
  }
  w = std::min(std::max(w, 100u), 5000u);
  h = std::min(std::max(h, 100u), 5000u);
  App app{{.name = "VkRender2", .width = w, .height = h, .maximize = maximize, .vsync = vsync}};
  app.run();
}
