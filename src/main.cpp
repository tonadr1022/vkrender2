
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

  auto& forward = rg.add_pass("forward", [](CmdEncoder&) { LINFO("executing forward"); });
  forward.add_color_output("forward_output", {.format = gfx::Format::R32G32B32A32Sfloat});
  forward.set_depth_stencil_output("depth", {.format = gfx::Format::D32Sfloat});
  auto& pp = rg.add_pass("postprocess", [](CmdEncoder&) { LINFO("executing postprocess"); });
  pp.add_color_output("final_out", {.format = gfx::Format::R8G8B8A8Unorm});

  pp.add_texture_input("forward_output");
  rg.set_backbuffer_img("final_out");

  auto res = rg.bake();
  if (!res) {
    LERROR("bake error {}", res.error());
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
