#pragma once

#include "StringUtil.hpp"

// inspired by https://github.com/vblanco20-1/vulkan-guide/blob/engine/extra-engine/cvars.h

#include <cstdint>

enum class CVarFlags : uint16_t {
  None = 0,
  NoEdit = 1 << 1,
  EditReadOnly = 1 << 2,
  Advanced = 1 << 3,

  EditCheckbox = 1 << 8,
  EditFloatDrag = 1 << 9,
};

class CVarParameter;
class CVarSystem {
 public:
  static CVarSystem& get();
  virtual CVarParameter* get_cvar(::util::string::Hash hash) = 0;
  virtual CVarParameter* create_float_cvar(const char* name, const char* description,
                                           double default_value, double current_value) = 0;
  virtual CVarParameter* create_int_cvar(const char* name, const char* description,
                                         int32_t default_value, int32_t current_value) = 0;
  virtual CVarParameter* create_string_cvar(const char* name, const char* description,
                                            const char* default_value,
                                            const char* current_value) = 0;
  virtual double* get_float_cvar(::util::string::Hash hash) = 0;
  virtual void set_float_cvar(::util::string::Hash hash, double value) = 0;
  virtual int32_t* get_int_cvar(::util::string::Hash hash) = 0;
  virtual void set_int_cvar(::util::string::Hash hash, int32_t value) = 0;
  virtual const char* get_string_cvar(::util::string::Hash hash) = 0;
  virtual void set_string_cvar(::util::string::Hash hash, const char* value) = 0;
  virtual void draw_imgui_editor() = 0;
};

template <typename T>
struct AutoCVar {
 protected:
  uint32_t idx_;
  using CVarType = T;
};

struct AutoCVarInt : AutoCVar<int32_t> {
  AutoCVarInt(const char* name, const char* desc, int initial_value,
              CVarFlags flags = CVarFlags::None);
  int32_t get();
  int32_t* get_ptr();
  void set(int32_t val);
};

struct AutoCVarFloat : AutoCVar<double> {
  AutoCVarFloat(const char* name, const char* description, double default_value,
                CVarFlags flags = CVarFlags::None);
  double get();
  double* get_ptr();
  float get_float();
  float* get_float_ptr();
  void set(double val);
};

struct AutoCVarString : AutoCVar<std::string> {
  AutoCVarString(const char* name, const char* description, const char* default_value,
                 CVarFlags flags = CVarFlags::None);
  const char* get();
  void set(std::string&& val);
};
