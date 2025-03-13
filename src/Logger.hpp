#pragma once

#include <print>

namespace lg {

#define LINFO(...) std::println("[info] " __VA_ARGS__)
#define LWARN(...) std::println("[warn] " __VA_ARGS__)
#define LERROR(...) std::println("[error] " __VA_ARGS__)
#define LCRITICAL(...) std::println("[critical] " __VA_ARGS__)

}  // namespace lg
