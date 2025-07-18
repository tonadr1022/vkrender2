#pragma once

// TODO: remove this
#include <vulkan/vulkan_core.h>

#include <cstdint>
#include <tuple>

#include "Common.hpp"
#include "core/Flags.hpp"
#include "vk2/Handle.hpp"
#include "vk2/Hash.hpp"

template <typename HandleT>
struct GenerationalHandle {
  GenerationalHandle() = default;

  explicit GenerationalHandle(uint32_t idx, uint32_t gen) : idx_(idx), gen_(gen) {}

  [[nodiscard]] bool is_valid() const { return gen_ != 0; }

  [[nodiscard]] uint32_t get_gen() const { return gen_; }
  [[nodiscard]] uint32_t get_idx() const { return idx_; }
  friend bool operator!=(const GenerationalHandle& a, const GenerationalHandle& b) {
    return a.idx_ != b.idx_ || a.gen_ != b.gen_;
  }
  friend bool operator==(const GenerationalHandle& a, const GenerationalHandle& b) {
    return a.idx_ == b.idx_ && a.gen_ == b.gen_;
  }

  template <typename, typename>
  friend struct Pool;

 private:
  uint32_t idx_{};
  uint32_t gen_{};
};

namespace std {
template <typename HandleT>
struct hash<GenerationalHandle<HandleT>> {
  std::size_t operator()(const GenerationalHandle<HandleT>& handle) const noexcept {
    auto h = std::make_tuple(handle.get_idx(), handle.get_gen());
    return gfx::vk2::detail::hashing::hash<decltype(h)>{}(h);
  }
};
}  // namespace std

namespace gfx {

namespace constants {
inline constexpr u32 remaining_array_layers = ~0U;
inline constexpr u32 remaining_mip_layers = ~0U;
inline constexpr u64 whole_size = ~0UL;
}  // namespace constants
inline constexpr u32 frames_in_flight = 2;
enum class QueueType : u8 {
  Graphics,
  Compute,
  Transfer,
  Count,
};
enum ColorComponentFlagBits : u8 {
  ColorComponentRBit = 0x00000001,
  ColorComponentGBit = 0x00000002,
  ColorComponentBBit = 0x00000004,
  ColorComponentABit = 0x00000008,
};
using ColorComponentFlags = u32;

enum class StencilOp : u8 {
  Keep = 0,
  Zero,
  Replace,
  IncrementAndClamp,
  DecrementAndClamp,
  IncrementAndWrap,
  DecrementAndWrap,
};
enum class CompareOp : u8 {
  Never = 0,
  Less,
  Equal,
  LessOrEqual,
  Greater,
  NotEqual,
  GreaterOrEqual,
  Always
};

enum class CullMode : u8 {
  None = 0,
  Front,
  Back,
};
enum class PolygonMode : u8 {
  Fill = 0,
  Line,
  Point,
};

enum class PrimitiveTopology : u8 {
  PointList,
  LineList,
  LineStrip,
  TriangleList,
  TriangleStrip,
  TriangleFan,
  PatchList
};
enum class BlendFactor : u8 {
  Zero = 0,
  One = 1,
  SrcColor = 2,
  OneMinusSrcColor = 3,
  DstColor = 4,
  OneMinusDstColor = 5,
  SrcAlpha = 6,
  OneMinusSrcAlpha = 7,
  DstAlpha = 8,
  OneMinusDstAlpha = 9,
  ConstantColor = 10,
  OneMinusConstantColor = 11,
  ConstantAlpha = 12,
  OneMinusConstantAlpha = 13,
  SRC_ALPHA_SATURATE = 14,
  Src1Color = 15,
  OneMinusSrc1Color = 16,
  Src1Alpha = 17,
  OneMinusSrc1Alpha = 18,
};

enum class BlendOp : u32 {
  Add = 0,
  Subtract = 1,
  ReverseSubtract = 2,
  Min = 3,
  Max = 4,
  ZeroExt = 1000148000,
  SrcExt = 1000148001,
  DstExt = 1000148002,
  SrcOverExt = 1000148003,
  DstOverExt = 1000148004,
  SrcInExt = 1000148005,
  DstInExt = 1000148006,
  SrcOutExt = 1000148007,
  DstOutExt = 1000148008,
  SrcAtopExt = 1000148009,
  DstAtopExt = 1000148010,
  XorExt = 1000148011,
  MultiplyExt = 1000148012,
  ScreenExt = 1000148013,
  OverlayExt = 1000148014,
  DarkenExt = 1000148015,
  LightenExt = 1000148016,
  ColorDodgeExt = 1000148017,
  ColorBurnExt = 1000148018,
  HardLightExt = 1000148019,
  SoftLightExt = 1000148020,
  DifferenceExt = 1000148021,
  ExclusionExt = 1000148022,
  InvertExt = 1000148023,
  InvertRgbExt = 1000148024,
  LinearDodgeExt = 1000148025,
  LinearBurnExt = 1000148026,
  VividLightExt = 1000148027,
  LinearLightExt = 1000148028,
  PinLightExt = 1000148029,
  HardMixExt = 1000148030,
  HslHueExt = 1000148031,
  HslSaturationExt = 1000148032,
  HslColorExt = 1000148033,
  HslLuminosityExt = 1000148034,
  PlusExt = 1000148035,
  PlusClampedExt = 1000148036,
  PlusClampedAlphaExt = 1000148037,
  PlusDarkerExt = 1000148038,
  MinusExt = 1000148039,
  MinusClampedExt = 1000148040,
  ContrastExt = 1000148041,
  InvertOvgExt = 1000148042,
  RedExt = 1000148043,
  GreenExt = 1000148044,
  BlueExt = 1000148045,
  MaxEnum = 0x7FFFFFFF
};
enum LogicOp : u8 {
  LogicOpClear = 0,
  LogicOpAnd = 1,
  LogicOpAndReverse = 2,
  LogicOpCopy = 3,
  LogicOpAndInverted = 4,
  LogicOpNoOp = 5,
  LogicOpXor = 6,
  LogicOpOr = 7,
  LogicOpNor = 8,
  LogicOpEquivalent = 9,
  LogicOpInvert = 10,
  LogicOpOrReverse = 11,
  LogicOpCopyInverted = 12,
  LogicOpOrInverted = 13,
  LogicOpNand = 14,
  LogicOpSet = 15,
};

enum SampleCountFlagBits : u8 {
  SampleCount1Bit = 0x00000001,
  SampleCount2Bit = 0x00000002,
  SampleCount4Bit = 0x00000004,
  SampleCount8Bit = 0x00000008,
  SampleCount16Bit = 0x00000010,
  SampleCount32Bit = 0x00000020,
  SampleCount64Bit = 0x00000040,
};
using SampleCountFlags = u32;

enum class FrontFace : u8 { CounterClockwise = 0, Clockwise };

enum class ImageViewType : u8 {
  OneD,
  TwoD,
  ThreeD,
  Cube,
  OneDArray,
  TwoDArray,
  CubeArray,
};

enum class Format {
  Undefined = VK_FORMAT_UNDEFINED,
  R4G4UnormPack8 = VK_FORMAT_R4G4_UNORM_PACK8,
  R4G4B4A4UnormPack16 = VK_FORMAT_R4G4B4A4_UNORM_PACK16,
  B4G4R4A4UnormPack16 = VK_FORMAT_B4G4R4A4_UNORM_PACK16,
  R5G6B5UnormPack16 = VK_FORMAT_R5G6B5_UNORM_PACK16,
  B5G6R5UnormPack16 = VK_FORMAT_B5G6R5_UNORM_PACK16,
  R5G5B5A1UnormPack16 = VK_FORMAT_R5G5B5A1_UNORM_PACK16,
  B5G5R5A1UnormPack16 = VK_FORMAT_B5G5R5A1_UNORM_PACK16,
  A1R5G5B5UnormPack16 = VK_FORMAT_A1R5G5B5_UNORM_PACK16,
  R8Unorm = VK_FORMAT_R8_UNORM,
  R8Snorm = VK_FORMAT_R8_SNORM,
  R8Uscaled = VK_FORMAT_R8_USCALED,
  R8Sscaled = VK_FORMAT_R8_SSCALED,
  R8Uint = VK_FORMAT_R8_UINT,
  R8Sint = VK_FORMAT_R8_SINT,
  R8Srgb = VK_FORMAT_R8_SRGB,
  R8G8Unorm = VK_FORMAT_R8G8_UNORM,
  R8G8Snorm = VK_FORMAT_R8G8_SNORM,
  R8G8Uscaled = VK_FORMAT_R8G8_USCALED,
  R8G8Sscaled = VK_FORMAT_R8G8_SSCALED,
  R8G8Uint = VK_FORMAT_R8G8_UINT,
  R8G8Sint = VK_FORMAT_R8G8_SINT,
  R8G8Srgb = VK_FORMAT_R8G8_SRGB,
  R8G8B8Unorm = VK_FORMAT_R8G8B8_UNORM,
  R8G8B8Snorm = VK_FORMAT_R8G8B8_SNORM,
  R8G8B8Uscaled = VK_FORMAT_R8G8B8_USCALED,
  R8G8B8Sscaled = VK_FORMAT_R8G8B8_SSCALED,
  R8G8B8Uint = VK_FORMAT_R8G8B8_UINT,
  R8G8B8Sint = VK_FORMAT_R8G8B8_SINT,
  R8G8B8Srgb = VK_FORMAT_R8G8B8_SRGB,
  B8G8R8Unorm = VK_FORMAT_B8G8R8_UNORM,
  B8G8R8Snorm = VK_FORMAT_B8G8R8_SNORM,
  B8G8R8Uscaled = VK_FORMAT_B8G8R8_USCALED,
  B8G8R8Sscaled = VK_FORMAT_B8G8R8_SSCALED,
  B8G8R8Uint = VK_FORMAT_B8G8R8_UINT,
  B8G8R8Sint = VK_FORMAT_B8G8R8_SINT,
  B8G8R8Srgb = VK_FORMAT_B8G8R8_SRGB,
  R8G8B8A8Unorm = VK_FORMAT_R8G8B8A8_UNORM,
  R8G8B8A8Snorm = VK_FORMAT_R8G8B8A8_SNORM,
  R8G8B8A8Uscaled = VK_FORMAT_R8G8B8A8_USCALED,
  R8G8B8A8Sscaled = VK_FORMAT_R8G8B8A8_SSCALED,
  R8G8B8A8Uint = VK_FORMAT_R8G8B8A8_UINT,
  R8G8B8A8Sint = VK_FORMAT_R8G8B8A8_SINT,
  R8G8B8A8Srgb = VK_FORMAT_R8G8B8A8_SRGB,
  B8G8R8A8Unorm = VK_FORMAT_B8G8R8A8_UNORM,
  B8G8R8A8Snorm = VK_FORMAT_B8G8R8A8_SNORM,
  B8G8R8A8Uscaled = VK_FORMAT_B8G8R8A8_USCALED,
  B8G8R8A8Sscaled = VK_FORMAT_B8G8R8A8_SSCALED,
  B8G8R8A8Uint = VK_FORMAT_B8G8R8A8_UINT,
  B8G8R8A8Sint = VK_FORMAT_B8G8R8A8_SINT,
  B8G8R8A8Srgb = VK_FORMAT_B8G8R8A8_SRGB,
  A8B8G8R8UnormPack32 = VK_FORMAT_A8B8G8R8_UNORM_PACK32,
  A8B8G8R8SnormPack32 = VK_FORMAT_A8B8G8R8_SNORM_PACK32,
  A8B8G8R8UscaledPack32 = VK_FORMAT_A8B8G8R8_USCALED_PACK32,
  A8B8G8R8SscaledPack32 = VK_FORMAT_A8B8G8R8_SSCALED_PACK32,
  A8B8G8R8UintPack32 = VK_FORMAT_A8B8G8R8_UINT_PACK32,
  A8B8G8R8SintPack32 = VK_FORMAT_A8B8G8R8_SINT_PACK32,
  A8B8G8R8SrgbPack32 = VK_FORMAT_A8B8G8R8_SRGB_PACK32,
  A2R10G10B10UnormPack32 = VK_FORMAT_A2R10G10B10_UNORM_PACK32,
  A2R10G10B10SnormPack32 = VK_FORMAT_A2R10G10B10_SNORM_PACK32,
  A2R10G10B10UscaledPack32 = VK_FORMAT_A2R10G10B10_USCALED_PACK32,
  A2R10G10B10SscaledPack32 = VK_FORMAT_A2R10G10B10_SSCALED_PACK32,
  A2R10G10B10UintPack32 = VK_FORMAT_A2R10G10B10_UINT_PACK32,
  A2R10G10B10SintPack32 = VK_FORMAT_A2R10G10B10_SINT_PACK32,
  A2B10G10R10UnormPack32 = VK_FORMAT_A2B10G10R10_UNORM_PACK32,
  A2B10G10R10SnormPack32 = VK_FORMAT_A2B10G10R10_SNORM_PACK32,
  A2B10G10R10UscaledPack32 = VK_FORMAT_A2B10G10R10_USCALED_PACK32,
  A2B10G10R10SscaledPack32 = VK_FORMAT_A2B10G10R10_SSCALED_PACK32,
  A2B10G10R10UintPack32 = VK_FORMAT_A2B10G10R10_UINT_PACK32,
  A2B10G10R10SintPack32 = VK_FORMAT_A2B10G10R10_SINT_PACK32,
  R16Unorm = VK_FORMAT_R16_UNORM,
  R16Snorm = VK_FORMAT_R16_SNORM,
  R16Uscaled = VK_FORMAT_R16_USCALED,
  R16Sscaled = VK_FORMAT_R16_SSCALED,
  R16Uint = VK_FORMAT_R16_UINT,
  R16Sint = VK_FORMAT_R16_SINT,
  R16Sfloat = VK_FORMAT_R16_SFLOAT,
  R16G16Unorm = VK_FORMAT_R16G16_UNORM,
  R16G16Snorm = VK_FORMAT_R16G16_SNORM,
  R16G16Uscaled = VK_FORMAT_R16G16_USCALED,
  R16G16Sscaled = VK_FORMAT_R16G16_SSCALED,
  R16G16Uint = VK_FORMAT_R16G16_UINT,
  R16G16Sint = VK_FORMAT_R16G16_SINT,
  R16G16Sfloat = VK_FORMAT_R16G16_SFLOAT,
  R16G16B16Unorm = VK_FORMAT_R16G16B16_UNORM,
  R16G16B16Snorm = VK_FORMAT_R16G16B16_SNORM,
  R16G16B16Uscaled = VK_FORMAT_R16G16B16_USCALED,
  R16G16B16Sscaled = VK_FORMAT_R16G16B16_SSCALED,
  R16G16B16Uint = VK_FORMAT_R16G16B16_UINT,
  R16G16B16Sint = VK_FORMAT_R16G16B16_SINT,
  R16G16B16Sfloat = VK_FORMAT_R16G16B16_SFLOAT,
  R16G16B16A16Unorm = VK_FORMAT_R16G16B16A16_UNORM,
  R16G16B16A16Snorm = VK_FORMAT_R16G16B16A16_SNORM,
  R16G16B16A16Uscaled = VK_FORMAT_R16G16B16A16_USCALED,
  R16G16B16A16Sscaled = VK_FORMAT_R16G16B16A16_SSCALED,
  R16G16B16A16Uint = VK_FORMAT_R16G16B16A16_UINT,
  R16G16B16A16Sint = VK_FORMAT_R16G16B16A16_SINT,
  R16G16B16A16Sfloat = VK_FORMAT_R16G16B16A16_SFLOAT,
  R32Uint = VK_FORMAT_R32_UINT,
  R32Sint = VK_FORMAT_R32_SINT,
  R32Sfloat = VK_FORMAT_R32_SFLOAT,
  R32G32Uint = VK_FORMAT_R32G32_UINT,
  R32G32Sint = VK_FORMAT_R32G32_SINT,
  R32G32Sfloat = VK_FORMAT_R32G32_SFLOAT,
  R32G32B32Uint = VK_FORMAT_R32G32B32_UINT,
  R32G32B32Sint = VK_FORMAT_R32G32B32_SINT,
  R32G32B32Sfloat = VK_FORMAT_R32G32B32_SFLOAT,
  R32G32B32A32Uint = VK_FORMAT_R32G32B32A32_UINT,
  R32G32B32A32Sint = VK_FORMAT_R32G32B32A32_SINT,
  R32G32B32A32Sfloat = VK_FORMAT_R32G32B32A32_SFLOAT,
  R64Uint = VK_FORMAT_R64_UINT,
  R64Sint = VK_FORMAT_R64_SINT,
  R64Sfloat = VK_FORMAT_R64_SFLOAT,
  R64G64Uint = VK_FORMAT_R64G64_UINT,
  R64G64Sint = VK_FORMAT_R64G64_SINT,
  R64G64Sfloat = VK_FORMAT_R64G64_SFLOAT,
  R64G64B64Uint = VK_FORMAT_R64G64B64_UINT,
  R64G64B64Sint = VK_FORMAT_R64G64B64_SINT,
  R64G64B64Sfloat = VK_FORMAT_R64G64B64_SFLOAT,
  R64G64B64A64Uint = VK_FORMAT_R64G64B64A64_UINT,
  R64G64B64A64Sint = VK_FORMAT_R64G64B64A64_SINT,
  R64G64B64A64Sfloat = VK_FORMAT_R64G64B64A64_SFLOAT,
  B10G11R11UfloatPack32 = VK_FORMAT_B10G11R11_UFLOAT_PACK32,
  E5B9G9R9UfloatPack32 = VK_FORMAT_E5B9G9R9_UFLOAT_PACK32,
  D16Unorm = VK_FORMAT_D16_UNORM,
  X8D24UnormPack32 = VK_FORMAT_X8_D24_UNORM_PACK32,
  D32Sfloat = VK_FORMAT_D32_SFLOAT,
  S8Uint = VK_FORMAT_S8_UINT,
  D16UnormS8Uint = VK_FORMAT_D16_UNORM_S8_UINT,
  D24UnormS8Uint = VK_FORMAT_D24_UNORM_S8_UINT,
  D32SfloatS8Uint = VK_FORMAT_D32_SFLOAT_S8_UINT,
  Bc1RgbUnormBlock = VK_FORMAT_BC1_RGB_UNORM_BLOCK,
  Bc1RgbSrgbBlock = VK_FORMAT_BC1_RGB_SRGB_BLOCK,
  Bc1RgbaUnormBlock = VK_FORMAT_BC1_RGBA_UNORM_BLOCK,
  Bc1RgbaSrgbBlock = VK_FORMAT_BC1_RGBA_SRGB_BLOCK,
  Bc2UnormBlock = VK_FORMAT_BC2_UNORM_BLOCK,
  Bc2SrgbBlock = VK_FORMAT_BC2_SRGB_BLOCK,
  Bc3UnormBlock = VK_FORMAT_BC3_UNORM_BLOCK,
  Bc3SrgbBlock = VK_FORMAT_BC3_SRGB_BLOCK,
  Bc4UnormBlock = VK_FORMAT_BC4_UNORM_BLOCK,
  Bc4SnormBlock = VK_FORMAT_BC4_SNORM_BLOCK,
  Bc5UnormBlock = VK_FORMAT_BC5_UNORM_BLOCK,
  Bc5SnormBlock = VK_FORMAT_BC5_SNORM_BLOCK,
  Bc6HUfloatBlock = VK_FORMAT_BC6H_UFLOAT_BLOCK,
  Bc6HSfloatBlock = VK_FORMAT_BC6H_SFLOAT_BLOCK,
  Bc7UnormBlock = VK_FORMAT_BC7_UNORM_BLOCK,
  Bc7SrgbBlock = VK_FORMAT_BC7_SRGB_BLOCK,
  Etc2R8G8B8UnormBlock = VK_FORMAT_ETC2_R8G8B8_UNORM_BLOCK,
  Etc2R8G8B8SrgbBlock = VK_FORMAT_ETC2_R8G8B8_SRGB_BLOCK,
  Etc2R8G8B8A1UnormBlock = VK_FORMAT_ETC2_R8G8B8A1_UNORM_BLOCK,
  Etc2R8G8B8A1SrgbBlock = VK_FORMAT_ETC2_R8G8B8A1_SRGB_BLOCK,
  Etc2R8G8B8A8UnormBlock = VK_FORMAT_ETC2_R8G8B8A8_UNORM_BLOCK,
  Etc2R8G8B8A8SrgbBlock = VK_FORMAT_ETC2_R8G8B8A8_SRGB_BLOCK,
  EacR11UnormBlock = VK_FORMAT_EAC_R11_UNORM_BLOCK,
  EacR11SnormBlock = VK_FORMAT_EAC_R11_SNORM_BLOCK,
  EacR11G11UnormBlock = VK_FORMAT_EAC_R11G11_UNORM_BLOCK,
  EacR11G11SnormBlock = VK_FORMAT_EAC_R11G11_SNORM_BLOCK,
  Astc4x4UnormBlock = VK_FORMAT_ASTC_4x4_UNORM_BLOCK,
  Astc4x4SrgbBlock = VK_FORMAT_ASTC_4x4_SRGB_BLOCK,
  Astc5x4UnormBlock = VK_FORMAT_ASTC_5x4_UNORM_BLOCK,
  Astc5x4SrgbBlock = VK_FORMAT_ASTC_5x4_SRGB_BLOCK,
  Astc5x5UnormBlock = VK_FORMAT_ASTC_5x5_UNORM_BLOCK,
  Astc5x5SrgbBlock = VK_FORMAT_ASTC_5x5_SRGB_BLOCK,
  Astc6x5UnormBlock = VK_FORMAT_ASTC_6x5_UNORM_BLOCK,
  Astc6x5SrgbBlock = VK_FORMAT_ASTC_6x5_SRGB_BLOCK,
  Astc6x6UnormBlock = VK_FORMAT_ASTC_6x6_UNORM_BLOCK,
  Astc6x6SrgbBlock = VK_FORMAT_ASTC_6x6_SRGB_BLOCK,
  Astc8x5UnormBlock = VK_FORMAT_ASTC_8x5_UNORM_BLOCK,
  Astc8x5SrgbBlock = VK_FORMAT_ASTC_8x5_SRGB_BLOCK,
  Astc8x6UnormBlock = VK_FORMAT_ASTC_8x6_UNORM_BLOCK,
  Astc8x6SrgbBlock = VK_FORMAT_ASTC_8x6_SRGB_BLOCK,
  Astc8x8UnormBlock = VK_FORMAT_ASTC_8x8_UNORM_BLOCK,
  Astc8x8SrgbBlock = VK_FORMAT_ASTC_8x8_SRGB_BLOCK,
  Astc10x5UnormBlock = VK_FORMAT_ASTC_10x5_UNORM_BLOCK,
  Astc10x5SrgbBlock = VK_FORMAT_ASTC_10x5_SRGB_BLOCK,
  Astc10x6UnormBlock = VK_FORMAT_ASTC_10x6_UNORM_BLOCK,
  Astc10x6SrgbBlock = VK_FORMAT_ASTC_10x6_SRGB_BLOCK,
  Astc10x8UnormBlock = VK_FORMAT_ASTC_10x8_UNORM_BLOCK,
  Astc10x8SrgbBlock = VK_FORMAT_ASTC_10x8_SRGB_BLOCK,
  Astc10x10UnormBlock = VK_FORMAT_ASTC_10x10_UNORM_BLOCK,
  Astc10x10SrgbBlock = VK_FORMAT_ASTC_10x10_SRGB_BLOCK,
  Astc12x10UnormBlock = VK_FORMAT_ASTC_12x10_UNORM_BLOCK,
  Astc12x10SrgbBlock = VK_FORMAT_ASTC_12x10_SRGB_BLOCK,
  Astc12x12UnormBlock = VK_FORMAT_ASTC_12x12_UNORM_BLOCK,
  Astc12x12SrgbBlock = VK_FORMAT_ASTC_12x12_SRGB_BLOCK,
  G8B8G8R8422Unorm = VK_FORMAT_G8B8G8R8_422_UNORM,
  B8G8R8G8422Unorm = VK_FORMAT_B8G8R8G8_422_UNORM,
  G8B8R83Plane420Unorm = VK_FORMAT_G8_B8_R8_3PLANE_420_UNORM,
  G8B8R82Plane420Unorm = VK_FORMAT_G8_B8R8_2PLANE_420_UNORM,
  G8B8R83Plane422Unorm = VK_FORMAT_G8_B8_R8_3PLANE_422_UNORM,
  G8B8R82Plane422Unorm = VK_FORMAT_G8_B8R8_2PLANE_422_UNORM,
  G8B8R83Plane444Unorm = VK_FORMAT_G8_B8_R8_3PLANE_444_UNORM,
  R10X6UnormPack16 = VK_FORMAT_R10X6_UNORM_PACK16,
  R10X6G10X6Unorm2Pack16 = VK_FORMAT_R10X6G10X6_UNORM_2PACK16,
  R10X6G10X6B10X6A10X6Unorm4Pack16 = VK_FORMAT_R10X6G10X6B10X6A10X6_UNORM_4PACK16,
  G10X6B10X6G10X6R10X6422Unorm4Pack16 = VK_FORMAT_G10X6B10X6G10X6R10X6_422_UNORM_4PACK16,
  B10X6G10X6R10X6G10X6422Unorm4Pack16 = VK_FORMAT_B10X6G10X6R10X6G10X6_422_UNORM_4PACK16,
  G10X6B10X6R10X63Plane420Unorm3Pack16 = VK_FORMAT_G10X6_B10X6_R10X6_3PLANE_420_UNORM_3PACK16,
  G10X6B10X6R10X62Plane420Unorm3Pack16 = VK_FORMAT_G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16,
  G10X6B10X6R10X63Plane422Unorm3Pack16 = VK_FORMAT_G10X6_B10X6_R10X6_3PLANE_422_UNORM_3PACK16,
  G10X6B10X6R10X62Plane422Unorm3Pack16 = VK_FORMAT_G10X6_B10X6R10X6_2PLANE_422_UNORM_3PACK16,
  G10X6B10X6R10X63Plane444Unorm3Pack16 = VK_FORMAT_G10X6_B10X6_R10X6_3PLANE_444_UNORM_3PACK16,
  R12X4UnormPack16 = VK_FORMAT_R12X4_UNORM_PACK16,
  R12X4G12X4Unorm2Pack16 = VK_FORMAT_R12X4G12X4_UNORM_2PACK16,
  R12X4G12X4B12X4A12X4Unorm4Pack16 = VK_FORMAT_R12X4G12X4B12X4A12X4_UNORM_4PACK16,
  G12X4B12X4G12X4R12X4422Unorm4Pack16 = VK_FORMAT_G12X4B12X4G12X4R12X4_422_UNORM_4PACK16,
  B12X4G12X4R12X4G12X4422Unorm4Pack16 = VK_FORMAT_B12X4G12X4R12X4G12X4_422_UNORM_4PACK16,
  G12X4B12X4R12X43Plane420Unorm3Pack16 = VK_FORMAT_G12X4_B12X4_R12X4_3PLANE_420_UNORM_3PACK16,
  G12X4B12X4R12X42Plane420Unorm3Pack16 = VK_FORMAT_G12X4_B12X4R12X4_2PLANE_420_UNORM_3PACK16,
  G12X4B12X4R12X43Plane422Unorm3Pack16 = VK_FORMAT_G12X4_B12X4_R12X4_3PLANE_422_UNORM_3PACK16,
  G12X4B12X4R12X42Plane422Unorm3Pack16 = VK_FORMAT_G12X4_B12X4R12X4_2PLANE_422_UNORM_3PACK16,
  G12X4B12X4R12X43Plane444Unorm3Pack16 = VK_FORMAT_G12X4_B12X4_R12X4_3PLANE_444_UNORM_3PACK16,
  G16B16G16R16422Unorm = VK_FORMAT_G16B16G16R16_422_UNORM,
  B16G16R16G16422Unorm = VK_FORMAT_B16G16R16G16_422_UNORM,
  G16B16R163Plane420Unorm = VK_FORMAT_G16_B16_R16_3PLANE_420_UNORM,
  G16B16R162Plane420Unorm = VK_FORMAT_G16_B16R16_2PLANE_420_UNORM,
  G16B16R163Plane422Unorm = VK_FORMAT_G16_B16_R16_3PLANE_422_UNORM,
  G16B16R162Plane422Unorm = VK_FORMAT_G16_B16R16_2PLANE_422_UNORM,
  G16B16R163Plane444Unorm = VK_FORMAT_G16_B16_R16_3PLANE_444_UNORM,
  Pvrtc12BppUnormBlockIMG = VK_FORMAT_PVRTC1_2BPP_UNORM_BLOCK_IMG,
  Pvrtc14BppUnormBlockIMG = VK_FORMAT_PVRTC1_4BPP_UNORM_BLOCK_IMG,
  Pvrtc22BppUnormBlockIMG = VK_FORMAT_PVRTC2_2BPP_UNORM_BLOCK_IMG,
  Pvrtc24BppUnormBlockIMG = VK_FORMAT_PVRTC2_4BPP_UNORM_BLOCK_IMG,
  Pvrtc12BppSrgbBlockIMG = VK_FORMAT_PVRTC1_2BPP_SRGB_BLOCK_IMG,
  Pvrtc14BppSrgbBlockIMG = VK_FORMAT_PVRTC1_4BPP_SRGB_BLOCK_IMG,
  Pvrtc22BppSrgbBlockIMG = VK_FORMAT_PVRTC2_2BPP_SRGB_BLOCK_IMG,
  Pvrtc24BppSrgbBlockIMG = VK_FORMAT_PVRTC2_4BPP_SRGB_BLOCK_IMG,
  Astc4x4SfloatBlockEXT = VK_FORMAT_ASTC_4x4_SFLOAT_BLOCK_EXT,
  Astc5x4SfloatBlockEXT = VK_FORMAT_ASTC_5x4_SFLOAT_BLOCK_EXT,
  Astc5x5SfloatBlockEXT = VK_FORMAT_ASTC_5x5_SFLOAT_BLOCK_EXT,
  Astc6x5SfloatBlockEXT = VK_FORMAT_ASTC_6x5_SFLOAT_BLOCK_EXT,
  Astc6x6SfloatBlockEXT = VK_FORMAT_ASTC_6x6_SFLOAT_BLOCK_EXT,
  Astc8x5SfloatBlockEXT = VK_FORMAT_ASTC_8x5_SFLOAT_BLOCK_EXT,
  Astc8x6SfloatBlockEXT = VK_FORMAT_ASTC_8x6_SFLOAT_BLOCK_EXT,
  Astc8x8SfloatBlockEXT = VK_FORMAT_ASTC_8x8_SFLOAT_BLOCK_EXT,
  Astc10x5SfloatBlockEXT = VK_FORMAT_ASTC_10x5_SFLOAT_BLOCK_EXT,
  Astc10x6SfloatBlockEXT = VK_FORMAT_ASTC_10x6_SFLOAT_BLOCK_EXT,
  Astc10x8SfloatBlockEXT = VK_FORMAT_ASTC_10x8_SFLOAT_BLOCK_EXT,
  Astc10x10SfloatBlockEXT = VK_FORMAT_ASTC_10x10_SFLOAT_BLOCK_EXT,
  Astc12x10SfloatBlockEXT = VK_FORMAT_ASTC_12x10_SFLOAT_BLOCK_EXT,
  Astc12x12SfloatBlockEXT = VK_FORMAT_ASTC_12x12_SFLOAT_BLOCK_EXT,
  B10X6G10X6R10X6G10X6422Unorm4Pack16KHR = VK_FORMAT_B10X6G10X6R10X6G10X6_422_UNORM_4PACK16_KHR,
  B12X4G12X4R12X4G12X4422Unorm4Pack16KHR = VK_FORMAT_B12X4G12X4R12X4G12X4_422_UNORM_4PACK16_KHR,
  B16G16R16G16422UnormKHR = VK_FORMAT_B16G16R16G16_422_UNORM_KHR,
  B8G8R8G8422UnormKHR = VK_FORMAT_B8G8R8G8_422_UNORM_KHR,
  G10X6B10X6G10X6R10X6422Unorm4Pack16KHR = VK_FORMAT_G10X6B10X6G10X6R10X6_422_UNORM_4PACK16_KHR,
  G10X6B10X6R10X62Plane420Unorm3Pack16KHR = VK_FORMAT_G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16_KHR,
  G10X6B10X6R10X62Plane422Unorm3Pack16KHR = VK_FORMAT_G10X6_B10X6R10X6_2PLANE_422_UNORM_3PACK16_KHR,
  G10X6B10X6R10X63Plane420Unorm3Pack16KHR =
      VK_FORMAT_G10X6_B10X6_R10X6_3PLANE_420_UNORM_3PACK16_KHR,
  G10X6B10X6R10X63Plane422Unorm3Pack16KHR =
      VK_FORMAT_G10X6_B10X6_R10X6_3PLANE_422_UNORM_3PACK16_KHR,
  G10X6B10X6R10X63Plane444Unorm3Pack16KHR =
      VK_FORMAT_G10X6_B10X6_R10X6_3PLANE_444_UNORM_3PACK16_KHR,
  G12X4B12X4G12X4R12X4422Unorm4Pack16KHR = VK_FORMAT_G12X4B12X4G12X4R12X4_422_UNORM_4PACK16_KHR,
  G12X4B12X4R12X42Plane420Unorm3Pack16KHR = VK_FORMAT_G12X4_B12X4R12X4_2PLANE_420_UNORM_3PACK16_KHR,
  G12X4B12X4R12X42Plane422Unorm3Pack16KHR = VK_FORMAT_G12X4_B12X4R12X4_2PLANE_422_UNORM_3PACK16_KHR,
  G12X4B12X4R12X43Plane420Unorm3Pack16KHR =
      VK_FORMAT_G12X4_B12X4_R12X4_3PLANE_420_UNORM_3PACK16_KHR,
  G12X4B12X4R12X43Plane422Unorm3Pack16KHR =
      VK_FORMAT_G12X4_B12X4_R12X4_3PLANE_422_UNORM_3PACK16_KHR,
  G12X4B12X4R12X43Plane444Unorm3Pack16KHR =
      VK_FORMAT_G12X4_B12X4_R12X4_3PLANE_444_UNORM_3PACK16_KHR,
  G16B16G16R16422UnormKHR = VK_FORMAT_G16B16G16R16_422_UNORM_KHR,
  G16B16R162Plane420UnormKHR = VK_FORMAT_G16_B16R16_2PLANE_420_UNORM_KHR,
  G16B16R162Plane422UnormKHR = VK_FORMAT_G16_B16R16_2PLANE_422_UNORM_KHR,
  G16B16R163Plane420UnormKHR = VK_FORMAT_G16_B16_R16_3PLANE_420_UNORM_KHR,
  G16B16R163Plane422UnormKHR = VK_FORMAT_G16_B16_R16_3PLANE_422_UNORM_KHR,
  G16B16R163Plane444UnormKHR = VK_FORMAT_G16_B16_R16_3PLANE_444_UNORM_KHR,
  G8B8G8R8422UnormKHR = VK_FORMAT_G8B8G8R8_422_UNORM_KHR,
  G8B8R82Plane420UnormKHR = VK_FORMAT_G8_B8R8_2PLANE_420_UNORM_KHR,
  G8B8R82Plane422UnormKHR = VK_FORMAT_G8_B8R8_2PLANE_422_UNORM_KHR,
  G8B8R83Plane420UnormKHR = VK_FORMAT_G8_B8_R8_3PLANE_420_UNORM_KHR,
  G8B8R83Plane422UnormKHR = VK_FORMAT_G8_B8_R8_3PLANE_422_UNORM_KHR,
  G8B8R83Plane444UnormKHR = VK_FORMAT_G8_B8_R8_3PLANE_444_UNORM_KHR,
  R10X6G10X6B10X6A10X6Unorm4Pack16KHR = VK_FORMAT_R10X6G10X6B10X6A10X6_UNORM_4PACK16_KHR,
  R10X6G10X6Unorm2Pack16KHR = VK_FORMAT_R10X6G10X6_UNORM_2PACK16_KHR,
  R10X6UnormPack16KHR = VK_FORMAT_R10X6_UNORM_PACK16_KHR,
  R12X4G12X4B12X4A12X4Unorm4Pack16KHR = VK_FORMAT_R12X4G12X4B12X4A12X4_UNORM_4PACK16_KHR,
  R12X4G12X4Unorm2Pack16KHR = VK_FORMAT_R12X4G12X4_UNORM_2PACK16_KHR,
  R12X4UnormPack16KHR = VK_FORMAT_R12X4_UNORM_PACK16_KHR
};

enum class ResourceMiscFlag : u8 {
  None = 0,
  ImageCube = 1 << 0,
  ImageSwapchain = 1 << 1,
};

enum class Usage : u8 {
  Default,   // GPU only
  Readback,  // GPU to CPU
  Upload     // CPU to GPU
};

enum class BindFlag : u8 {
  None = 0,
  VertexBuffer = 1 << 0,
  IndexBuffer = 1 << 1,
  UniformBuffer = 1 << 2,
  ShaderResource = 1 << 3,  // sampled images
  ColorAttachment = 1 << 4,
  DepthStencilAttachment = 1 << 5,
  Storage = 1 << 6,  // storage images/buffers
};

// using ImageUsageFlags = uint8_t;
// enum ImageUsageFlagBits : ImageUsageFlags {
//   ImageUsageTransferSrcBit = 0x1,
//   ImageUsageTransferDstBit = 0x2,
//   ImageUsageSampledBit = 0x4,
//   ImageUsageStorageBit = 0x8,
//   ImageUsageColorAttachmentBit = 0x00000010,
//   ImageUsageDepthStencilAttachmentBit = 0x00000020,
// };

enum Access : uint16_t {
  None = 1ULL << 0,
  ColorWrite = 1ULL << 1,
  ColorRead = 1ULL << 2,
  ColorRW = ColorRead | ColorWrite,
  DepthStencilRead = 1ULL << 3,
  DepthStencilWrite = 1ULL << 4,
  DepthStencilRW = DepthStencilRead | DepthStencilWrite,
  VertexRead = 1ULL << 5,
  IndexRead = 1ULL << 6,
  IndirectRead = 1ULL << 7,
  ComputeRead = 1ULL << 8,
  ComputeWrite = 1ULL << 9,
  ComputeRW = ComputeRead | ComputeWrite,
  TransferRead = 1ULL << 10,
  TransferWrite = 1ULL << 11,
  FragmentRead = 1ULL << 12,
  ComputeSample = 1ULL << 13,
};

using AccessFlags = uint32_t;

enum class SizeClass : uint8_t { Absolute, SwapchainRelative, InputRelative };

struct AttachmentInfo {
  SizeClass size_class{SizeClass::SwapchainRelative};
  uvec3 dims{1};
  Format format{};
  u32 layers{1};
  u32 levels{1};
};
enum class PipelineBindPoint : u8 { Graphics, Compute };

enum class FilterMode : u8 { Nearest, Linear };

enum class BorderColor : u8 {
  FloatTransparentBlack,
  IntTransparentBlack,
  FloatOpaqueBlack,
  IntOpaqueBlack,
  FLoatOpaqueWhite,
  IntOpaqueWhite
};
enum class AddressMode : u8 {
  Repeat,
  MirroredRepeat,
  ClampToEdge,
  ClampToBorder,
  MirrorClampToEdge
};

struct SamplerCreateInfo {
  FilterMode min_filter{FilterMode::Nearest};
  FilterMode mag_filter{FilterMode::Nearest};
  FilterMode mipmap_mode{FilterMode::Nearest};
  float min_lod{0.f};
  float max_lod{1000.f};
  AddressMode address_mode{AddressMode::Repeat};
  BorderColor border_color{BorderColor::FloatTransparentBlack};
  bool anisotropy_enable{};
  float max_anisotropy{};
  bool compare_enable{};
  CompareOp compare_op{};
};
class ImageView;
class Image;
class Buffer;
class Sampler;

using ImageHandle = GenerationalHandle<class ::gfx::Image>;
using BufferHandle = GenerationalHandle<class ::gfx::Buffer>;
using SamplerHandle = GenerationalHandle<class ::gfx::Sampler>;
// TODO: move
VK2_DEFINE_HANDLE_WITH_NAME(Pipeline, PipelineAndMetadata);

enum class SubresourceType : u8 { Storage, Shader, Attachment };

union ClearValue {
  vec4 color;
  struct {
    float depth;
    u32 stencil;
  } depth_stencil;
};
enum class IndexType : u8 { uint8, uint16, uint32 };

enum class LoadOp : u8 { Load, Clear, DontCare };
enum class StoreOp : u8 { Store, DontCare };

struct RenderingAttachmentInfo {
  enum class Type : u8 { Color, DepthStencil };

  ImageHandle image;
  i32 subresource{-1};
  Type type{Type::Color};
  LoadOp load_op{LoadOp::Load};
  StoreOp store_op{StoreOp::Store};
  ClearValue clear_value{};

  static RenderingAttachmentInfo color_att(ImageHandle image, LoadOp load_op = LoadOp::Load,
                                           ClearValue clear_value = {},
                                           StoreOp store_op = StoreOp::Store,
                                           int subresource = -1) {
    return {.image = image,
            .subresource = subresource,
            .type = Type::Color,
            .load_op = load_op,
            .store_op = store_op,
            .clear_value = clear_value};
  }

  static RenderingAttachmentInfo depth_stencil_att(ImageHandle image, LoadOp load_op = LoadOp::Load,
                                                   ClearValue clear_value = {},
                                                   StoreOp store_op = StoreOp::Store,
                                                   int subresource = -1) {
    return {.image = image,
            .subresource = subresource,
            .type = Type::DepthStencil,
            .load_op = load_op,
            .store_op = store_op,
            .clear_value = clear_value};
  }
};

}  // namespace gfx

template <>
struct EnableBitmaskOperators<gfx::ResourceMiscFlag> {
  static constexpr bool enable = true;
};
template <>
struct EnableBitmaskOperators<gfx::BindFlag> {
  static constexpr bool enable = true;
};
