#include "CVar.hpp"

#include <algorithm>
#include <span>
#include <unordered_map>
#include <vector>

#include "imgui.h"
#include "imgui_stdlib.h"

enum class CVarType : char {
  Int,
  Float,
  String,
};

class CVarParameter {
 public:
  friend class CVarSystemImpl;

  u32 array_idx;

  CVarType type;
  CVarFlags flags;
  std::string name;
  std::string description;
};

template <typename T>
struct CVarStorage {
  T default_value;
  T current;
  CVarParameter* parameter;
};

template <typename T>
struct CVarArray {
  std::vector<CVarStorage<T>> cvars;
  u32 last_cvar{0};
  explicit CVarArray(size_t size) { cvars.reserve(size); }
  T get_current(u32 idx) { return cvars[idx].current; }
  T* get_current_ptr(u32 idx) { return &cvars[idx].current; }
  void set_current(const T& val, u32 idx) { cvars[idx].current = val; }

  u32 add(const T& default_value, const T& current_value, CVarParameter* param) {
    u32 idx = cvars.size();
    cvars.emplace_back(default_value, current_value, param);
    param->array_idx = idx;
    return idx;
  }
  u32 add(const T& value, CVarParameter* param) {
    u32 idx = cvars.size();
    cvars.emplace_back(value, value, param);
    param->array_idx = idx;
    return idx;
  }
};

class CVarSystemImpl : public CVarSystem {
 public:
  CVarParameter* get_cvar(util::string::Hash hash) final {
    auto it = saved_cvars_.find(hash);
    if (it != saved_cvars_.end()) {
      return &it->second;
    }
    return nullptr;
  }
  CVarParameter* create_float_cvar(const char* name, const char* description, double default_value,
                                   double current_value) final {
    auto* param = init_cvar(name, description);
    param->type = CVarType::Float;
    get_cvar_array<double>().add(default_value, current_value, param);
    return param;
  }

  CVarParameter* create_string_cvar(const char* name, const char* description,
                                    const char* default_value, const char* current_value) final {
    auto* param = init_cvar(name, description);
    param->type = CVarType::Float;
    get_cvar_array<std::string>().add(default_value, current_value, param);
    return param;
  }
  CVarParameter* create_int_cvar(const char* name, const char* description, int32_t default_value,
                                 int32_t current_value) final {
    auto* param = init_cvar(name, description);
    param->type = CVarType::Int;
    get_cvar_array<int32_t>().add(default_value, current_value, param);
    return param;
  }

  template <typename T>
  T* get_cvar_current(u32 hash) {
    CVarParameter* param = get_cvar(hash);
    if (!param) return nullptr;
    return get_cvar_array<T>().get_current_ptr(param->array_idx);
  }
  template <typename T>
  void set_cvar_current(u32 hash, const T& value) {
    CVarParameter* param = get_cvar(hash);
    if (param) {
      get_cvar_array<T>().set_current(value, param->array_idx);
    }
  }

  double* get_float_cvar(util::string::Hash hash) final { return get_cvar_current<double>(hash); }

  void set_float_cvar(util::string::Hash hash, double value) final {
    set_cvar_current<double>(hash, value);
  }

  int32_t* get_int_cvar(util::string::Hash hash) final { return get_cvar_current<int32_t>(hash); }
  void set_int_cvar(util::string::Hash hash, int32_t value) final {
    set_cvar_current<int32_t>(hash, value);
  }
  const char* get_string_cvar(util::string::Hash hash) final {
    return get_cvar_current<std::string>(hash)->c_str();
  }
  void set_string_cvar(util::string::Hash hash, const char* value) final {
    set_cvar_current<std::string>(hash, value);
  }

  template <typename T>
  CVarArray<T>& get_cvar_array();

  template <>
  CVarArray<i32>& get_cvar_array() {
    return int_cvars_;
  }

  template <>
  CVarArray<double>& get_cvar_array() {
    return float_cvars_;
  }

  template <>
  CVarArray<std::string>& get_cvar_array() {
    return string_cvars_;
  }

  static CVarSystemImpl& get() { return static_cast<CVarSystemImpl&>(CVarSystem::get()); }

  void im_gui_label(const char* label, float text_width) {
    constexpr const float left_pad = 50;
    constexpr const float editor_width = 100;
    const ImVec2 line_start = ImGui::GetCursorScreenPos();
    float full_width = text_width + left_pad;
    ImGui::Text("%s", label);
    ImGui::SameLine();
    ImGui::SetCursorScreenPos(ImVec2{line_start.x + full_width, line_start.y});
    ImGui::SetNextItemWidth(editor_width);
  }

  void draw_imgui_edit_cvar_parameter(CVarParameter* p, float text_width) {
    const bool is_read_only =
        static_cast<u32>(p->flags) & static_cast<u32>(CVarFlags::EditReadOnly);
    const bool is_checkbox = static_cast<u32>(p->flags) & static_cast<u32>(CVarFlags::EditCheckbox);
    const bool is_drag = static_cast<u32>(p->flags) & static_cast<u32>(CVarFlags::EditFloatDrag);

    switch (p->type) {
      case CVarType::Int:
        if (is_read_only) {
          std::string display_format = p->name + "= %i";
          ImGui::Text(display_format.c_str(), get_cvar_array<int32_t>().get_current(p->array_idx));
        } else {
          if (is_checkbox) {
            im_gui_label(p->name.c_str(), text_width);
            ImGui::PushID(p);
            bool is_checked = get_cvar_array<int32_t>().get_current(p->array_idx) != 0;
            if (ImGui::Checkbox("", &is_checked)) {
              get_cvar_array<int32_t>().set_current(static_cast<int32_t>(is_checked), p->array_idx);
            }
            ImGui::PopID();
          } else {
            im_gui_label(p->name.c_str(), text_width);
            ImGui::PushID(p);
            ImGui::InputInt("", get_cvar_array<int32_t>().get_current_ptr(p->array_idx));
            ImGui::PopID();
          }
        }
        break;
      case CVarType::Float:
        if (is_read_only) {
          std::string display_format = p->name + "= %f";
          ImGui::Text(display_format.c_str(), get_cvar_array<int32_t>().get_current(p->array_idx));
        } else {
          im_gui_label(p->name.c_str(), text_width);
          ImGui::PushID(p);
          if (is_drag) {
            float d = get_cvar_array<double>().get_current(p->array_idx);
            if (ImGui::DragFloat("", &d, .01)) {
              *get_cvar_array<double>().get_current_ptr(p->array_idx) = static_cast<double>(d);
            }
          } else {
            ImGui::InputDouble("", get_cvar_array<double>().get_current_ptr(p->array_idx));
          }
          ImGui::PopID();
        }
        break;
      case CVarType::String:
        if (is_read_only) {
          std::string display_format = p->name + "= %s";
          ImGui::Text(display_format.c_str(),
                      get_cvar_array<std::string>().get_current(p->array_idx).c_str());
        } else {
          im_gui_label(p->name.c_str(), text_width);
          ImGui::PushID(p);
          ImGui::InputText("", get_cvar_array<std::string>().get_current_ptr(p->array_idx));
          ImGui::PopID();
        }
    }
    if (ImGui::IsItemHovered()) {
      ImGui::SetTooltip("%s", p->description.c_str());
    }
  }

  bool show_advanced = true;
  std::string search_txt;
  std::unordered_map<std::string, std::vector<CVarParameter*>> categorized_params;
  void draw_imgui_editor() final {
    ImGui::InputText("Filter", &search_txt);
    ImGui::Checkbox("Advanced", &show_advanced);
    active_edit_parameters_.clear();
    auto add_to_edit_list = [&](CVarParameter* param) {
      bool hidden = static_cast<u32>(param->flags) & static_cast<u32>(CVarFlags::NoEdit);
      bool advanced = static_cast<u32>(param->flags) & static_cast<u32>(CVarFlags::Advanced);
      if (!hidden && (show_advanced || !advanced) &&
          param->name.find(search_txt) != std::string::npos) {
        active_edit_parameters_.emplace_back(param);
      }
    };

    for (auto& v : get_cvar_array<int32_t>().cvars) {
      add_to_edit_list(v.parameter);
    }
    for (auto& v : get_cvar_array<double>().cvars) {
      add_to_edit_list(v.parameter);
    }
    for (auto& v : get_cvar_array<std::string>().cvars) {
      add_to_edit_list(v.parameter);
    }
    auto edit_params = [this](std::span<CVarParameter*> params) {
      std::ranges::sort(params,
                        [](CVarParameter* a, CVarParameter* b) { return a->name < b->name; });
      float max_text_width = 0;
      for (CVarParameter* p : params) {
        max_text_width = std::max(max_text_width, ImGui::CalcTextSize(p->name.c_str()).x);
      }
      for (CVarParameter* p : params) {
        draw_imgui_edit_cvar_parameter(p, max_text_width);
      }
    };
    // categorize by "."
    if (active_edit_parameters_.size() > 10) {
      categorized_params.clear();
      for (CVarParameter* p : active_edit_parameters_) {
        size_t dot_pos = p->name.find_first_of('.');
        std::string category;
        if (dot_pos != std::string::npos) {
          category = p->name.substr(0, dot_pos);
        }
        categorized_params[category].emplace_back(p);
      }
      for (auto& [category, params] : categorized_params) {
        if (ImGui::BeginMenu(category.c_str())) {
          edit_params(params);
          ImGui::EndMenu();
        }
      }
    } else {
      edit_params(active_edit_parameters_);
    }
  }

 private:
  CVarParameter* init_cvar(const char* name, const char* description) {
    if (get_cvar(name)) return nullptr;
    u32 name_hash = util::string::Hash{name};
    auto r = saved_cvars_.emplace(name_hash, CVarParameter{});
    CVarParameter* new_param = &r.first->second;
    new_param->name = name;
    new_param->description = description;
    return new_param;
  }

  std::vector<CVarParameter*> active_edit_parameters_;
  std::unordered_map<u32, CVarParameter> saved_cvars_;
  CVarArray<i32> int_cvars_{200};
  CVarArray<double> float_cvars_{200};
  CVarArray<std::string> string_cvars_{200};
};

CVarSystem& CVarSystem::get() {
  static CVarSystemImpl impl{};
  return impl;
}

AutoCVarFloat::AutoCVarFloat(const char* name, const char* description, double default_value,
                             CVarFlags flags) {
  CVarParameter* param =
      CVarSystemImpl::get().create_float_cvar(name, description, default_value, default_value);
  param->flags = flags;
  idx_ = param->array_idx;
}

namespace {

template <typename T>
T get_cvar_current_by_index(u32 idx) {
  return CVarSystemImpl::get().get_cvar_array<T>().get_current(idx);
}
template <typename T>
T* get_cvar_current_ptr_by_index(u32 idx) {
  return CVarSystemImpl::get().get_cvar_array<T>().get_current_ptr(idx);
}

template <typename T>
void set_cvar_by_idx(u32 idx, const T& data) {
  CVarSystemImpl::get().get_cvar_array<T>().set_current(data, idx);
}

}  // namespace

double AutoCVarFloat::get() { return get_cvar_current_by_index<CVarType>(idx_); }

double* AutoCVarFloat::get_ptr() { return get_cvar_current_ptr_by_index<CVarType>(idx_); }

float AutoCVarFloat::get_float() { return static_cast<float>(get()); }

float* AutoCVarFloat::get_float_ptr() { return reinterpret_cast<float*>(get_ptr()); }

void AutoCVarFloat::set(double val) { set_cvar_by_idx(idx_, val); }

AutoCVarInt::AutoCVarInt(const char* name, const char* desc, int initial_value, CVarFlags flags) {
  CVarParameter* param =
      CVarSystemImpl::get().create_int_cvar(name, desc, initial_value, initial_value);
  param->flags = flags;
  idx_ = param->array_idx;
}

int32_t AutoCVarInt::get() { return get_cvar_current_by_index<CVarType>(idx_); }

int32_t* AutoCVarInt::get_ptr() { return get_cvar_current_ptr_by_index<CVarType>(idx_); }

void AutoCVarInt::set(int32_t val) { set_cvar_by_idx(idx_, val); }
AutoCVarString::AutoCVarString(const char* name, const char* description, const char* default_value,
                               CVarFlags flags) {
  CVarParameter* param =
      CVarSystemImpl::get().create_string_cvar(name, description, default_value, default_value);
  param->flags = flags;
  idx_ = param->array_idx;
}

const char* AutoCVarString::get() { return get_cvar_current_ptr_by_index<CVarType>(idx_)->c_str(); }

void AutoCVarString::set(std::string&& val) { set_cvar_by_idx<CVarType>(idx_, val); }
