// Copyright 2020 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <array>
#include <memory>
#include <utility>
#include <vector>
#include <google/protobuf/util/message_differencer.h>

#include "absl/log/absl_log.h"
#include "mediapipe/calculators/tensor/image_to_tensor_calculator.pb.h"
#include "mediapipe/calculators/tensor/image_to_tensor_calculator_core.h"
#include "mediapipe/calculators/tensor/image_to_tensor_converter.h"
#include "mediapipe/calculators/tensor/image_to_tensor_utils.h"
#include "mediapipe/framework/api2/node.h"
#include "mediapipe/framework/api2/packet.h"
#include "mediapipe/framework/api2/port.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/memory_manager.h"
#include "mediapipe/framework/memory_manager_service.h"
#include "mediapipe/framework/port.h"
#include "mediapipe/framework/port/canonical_errors.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/port/statusor.h"
#include "mediapipe/gpu/gpu_origin.pb.h"
#include "mediapipe/gpu/gpu_origin_utils.h"
#include "mediapipe/gpu/webgpu/webgpu_check.h"

#if !MEDIAPIPE_DISABLE_OPENCV
#include "mediapipe/calculators/tensor/image_to_tensor_converter_opencv.h"
#elif MEDIAPIPE_ENABLE_HALIDE
#include "mediapipe/calculators/tensor/image_to_tensor_converter_frame_buffer.h"
#endif

#if !MEDIAPIPE_DISABLE_GPU
#include "mediapipe/gpu/gpu_buffer.h"

#if MEDIAPIPE_METAL_ENABLED
#include "mediapipe/calculators/tensor/image_to_tensor_converter_metal.h"
#include "mediapipe/gpu/MPPMetalHelper.h"
#elif MEDIAPIPE_OPENGL_ES_VERSION >= MEDIAPIPE_OPENGL_ES_31
#include "mediapipe/calculators/tensor/image_to_tensor_converter_gl_buffer.h"
#include "mediapipe/gpu/gl_calculator_helper.h"
#include "mediapipe/gpu/gpu_service.h"
#else
#include "mediapipe/calculators/tensor/image_to_tensor_converter_gl_texture.h"
#include "mediapipe/gpu/gl_calculator_helper.h"
#include "mediapipe/gpu/gpu_service.h"
#if MEDIAPIPE_USE_WEBGPU
#include "mediapipe/gpu/webgpu/image_to_tensor_converter_webgpu_texture.h"
#include "mediapipe/gpu/webgpu/webgpu_service.h"
#include "mediapipe/gpu/webgpu/webgpu_texture_buffer.h"
#endif  // MEDIAPIPE_USE_WEBGPU
#endif  // MEDIAPIPE_METAL_ENABLED
#endif  // !MEDIAPIPE_DISABLE_GPU

namespace mediapipe {
namespace api2 {

// Converts image into Tensor, possibly with cropping, resizing and
// normalization, according to specified inputs and options.
//
// Inputs:
//   IMAGE - Image[ImageFormat::SRGB / SRGBA, GpuBufferFormat::kBGRA32] or
//           ImageFrame [ImageFormat::SRGB/SRGBA] (for backward compatibility
//           with existing graphs that use IMAGE for ImageFrame input)
//   IMAGE_GPU - GpuBuffer [GpuBufferFormat::kBGRA32]
//     Image to extract from.
//
//   Note:
//   - One and only one of IMAGE and IMAGE_GPU should be specified.
//   - IMAGE input of type Image is processed on GPU if the data is already on
//     GPU (i.e., Image::UsesGpu() returns true), or otherwise processed on CPU.
//   - IMAGE input of type ImageFrame is always processed on CPU.
//   - IMAGE_GPU input (of type GpuBuffer) is always processed on GPU.
//
//   NORM_RECT - NormalizedRect @Optional
//     Describes region of image to extract.
//     @Optional: rect covering the whole image is used if not specified.
//
// Outputs:
//   TENSORS - std::vector<Tensor>
//     Vector containing a single Tensor populated with an extracted RGB image.
//   MATRIX - std::array<float, 16> @Optional
//     An std::array<float, 16> representing a 4x4 row-major-order matrix that
//     maps a point on the input image to a point on the output tensor, and
//     can be used to reverse the mapping by inverting the matrix.
//   LETTERBOX_PADDING - std::array<float, 4> @Optional
//     An std::array<float, 4> representing the letterbox padding from the 4
//     sides ([left, top, right, bottom]) of the output image, normalized to
//     [0.f, 1.f] by the output dimensions. The padding values are non-zero only
//     when the "keep_aspect_ratio" is true.
//
//     For instance, when the input image is 10x10 (width x height) and the
//     output dimensions specified in the calculator option are 20x40 and
//     "keep_aspect_ratio" is true, the calculator scales the input image to
//     20x20 and places it in the middle of the output image with an equal
//     padding of 10 pixels at the top and the bottom. The resulting array is
//     therefore [0.f, 0.25f, 0.f, 0.25f] (10/40 = 0.25f).
//
class ImageToTensorCalculator : public Node {
 public:
  static constexpr Input<
      OneOf<mediapipe::Image, mediapipe::ImageFrame>>::Optional kIn{"IMAGE"};
  static constexpr Input<GpuBuffer>::Optional kInGpu{"IMAGE_GPU"};
  static constexpr Input<mediapipe::NormalizedRect>::Optional kInNormRect{
      "NORM_RECT"};
  static constexpr Output<std::vector<Tensor>>::Optional kOutTensors{"TENSORS"};
  static constexpr Output<Tensor>::Optional kOutTensor{"TENSOR"};
  static constexpr Output<std::array<float, 4>>::Optional kOutLetterboxPadding{
      "LETTERBOX_PADDING"};
  static constexpr Output<std::array<float, 16>>::Optional kOutMatrix{"MATRIX"};

  MEDIAPIPE_NODE_CONTRACT(kIn, kInGpu, kInNormRect, kOutTensors, kOutTensor,
                          kOutLetterboxPadding, kOutMatrix);

  static absl::Status UpdateContract(CalculatorContract* cc) {
    const auto& options =
        cc->Options<mediapipe::ImageToTensorCalculatorOptions>();
    cc->UseService(kMemoryManagerService).Optional();
    return absl::OkStatus();
  }

  absl::Status Open(CalculatorContext* cc) {
    if (cc->Service(kMemoryManagerService).IsAvailable()) {
      memory_manager_ = &cc->Service(kMemoryManagerService).GetObject();
    }
    options_ = cc->Options<mediapipe::ImageToTensorCalculatorOptions>();
    params_ = GetOutputTensorParams(options_);
    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) {
    if ((kIn(cc).IsConnected() && kIn(cc).IsEmpty()) || (kInGpu(cc).IsConnected() && kInGpu(cc).IsEmpty())) { return absl::OkStatus(); }

    absl::optional<mediapipe::NormalizedRect> norm_rect;
    if (kInNormRect(cc).IsConnected()) {
      if (kInNormRect(cc).IsEmpty()) {
        // Timestamp bound update happens automatically. (See Open().)
        return absl::OkStatus();
      }
      norm_rect = *kInNormRect(cc);
    }

    MP_ASSIGN_OR_RETURN(auto image, GetInputImage(kIn(cc)));

    options_ = ImageToTensorCalculatorOptions();
    options_.set_output_tensor_width(192);
    options_.set_output_tensor_height(192);
    options_.set_keep_aspect_ratio(true);
    options_.mutable_output_tensor_float_range()->set_min(0.0f);
    options_.mutable_output_tensor_float_range()->set_max(1.0f);
    options_.set_border_mode(mediapipe::ImageToTensorCalculatorOptions::BORDER_ZERO);
    params_ = GetOutputTensorParams(options_);
    auto tensor_width = params_.output_width.value_or(image->width());
    auto tensor_height = params_.output_height.value_or(image->height());

    ImageToTensorCoreResult core_result;
    MP_RETURN_IF_ERROR(ImageToTensorCalculatorCore(
      *image,
      options_,
      tensor_width,
      tensor_height,
      params_,
      gpu_converter_,
      cpu_converter_,
      memory_manager_,
      norm_rect,
      &core_result));

    if (kOutMatrix(cc).IsConnected()) { kOutMatrix(cc).Send(core_result.matrix); }
    if (kOutLetterboxPadding(cc).IsConnected()) { kOutLetterboxPadding(cc).Send(core_result.padding); }
    if (kOutTensors(cc).IsConnected()) {
      auto result = std::make_unique<std::vector<Tensor>>();
      *result = std::move(core_result.tensors);
      kOutTensors(cc).Send(std::move(result));
    } else if (core_result.tensor) {
      kOutTensor(cc).Send(std::move(*core_result.tensor));
    }
    return absl::OkStatus();
  }

 private:

  std::unique_ptr<ImageToTensorConverter> gpu_converter_;
  std::unique_ptr<ImageToTensorConverter> cpu_converter_;
  mediapipe::ImageToTensorCalculatorOptions options_;
  OutputTensorParams params_;
  MemoryManager* memory_manager_ = nullptr;
};

MEDIAPIPE_REGISTER_NODE(ImageToTensorCalculator);

}  // namespace api2
}  // namespace mediapipe
