#include "mediapipe/calculators/tensor/image_to_tensor_calculator_core.h"
#include "mediapipe/calculators/tensor/image_to_tensor_utils.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/framework/memory_manager.h"
#include "mediapipe/calculators/tensor/image_to_tensor_converter.h"
#include "mediapipe/calculators/tensor/image_to_tensor_calculator.pb.h"
#include "absl/status/status.h"
#include "absl/types/optional.h"
#include <array>
#include <memory>
#include <vector>

#if !MEDIAPIPE_DISABLE_OPENCV
#include "mediapipe/calculators/tensor/image_to_tensor_converter_opencv.h"
#elif MEDIAPIPE_ENABLE_HALIDE
#include "mediapipe/calculators/tensor/image_to_tensor_converter_frame_buffer.h"
#endif

namespace mediapipe {

namespace {
absl::Status InitConverterIfNecessary(
    const mediapipe::Image& image,
    const mediapipe::ImageToTensorCalculatorOptions& options,
    const mediapipe::OutputTensorParams& params,
    std::unique_ptr<mediapipe::ImageToTensorConverter>& gpu_converter,
    std::unique_ptr<mediapipe::ImageToTensorConverter>& cpu_converter) {
  if (image.UsesGpu()) {
    if (!params.is_float_output) {
      return absl::UnimplementedError(
          "ImageToTensorConverter for the input GPU image currently doesn't support quantization.");
    }
    if (!gpu_converter) {
      // TODO: Fill in GPU converter initialization as per your platform.
      return absl::UnimplementedError("GPU converter initialization not implemented in core.");
    }
  } else {
    if (!cpu_converter) {
#if !MEDIAPIPE_DISABLE_OPENCV
      MP_ASSIGN_OR_RETURN(
          cpu_converter,
          CreateOpenCvConverter(
              /*context*/ nullptr, GetBorderMode(options.border_mode()),
              GetOutputTensorType(/*uses_gpu=*/false, params)));
#elif MEDIAPIPE_ENABLE_HALIDE
      MP_ASSIGN_OR_RETURN(
          cpu_converter,
          CreateFrameBufferConverter(
              /*context*/ nullptr, GetBorderMode(options.border_mode()),
              GetOutputTensorType(/*uses_gpu=*/false, params)));
#else
      return absl::UnimplementedError("No CPU converter available for this build.");
#endif
    }
  }
  return absl::OkStatus();
}
} // namespace

namespace api2 {
absl::Status ImageToTensorCalculatorCore(
    const mediapipe::Image& image,
    const mediapipe::ImageToTensorCalculatorOptions& options,
    int tensor_width,
    int tensor_height,
    const mediapipe::OutputTensorParams& params,
    std::unique_ptr<mediapipe::ImageToTensorConverter>& gpu_converter,
    std::unique_ptr<mediapipe::ImageToTensorConverter>& cpu_converter,
    mediapipe::MemoryManager* memory_manager,
    absl::optional<mediapipe::NormalizedRect> norm_rect,
    ImageToTensorCoreResult* result) {
  mediapipe::RotatedRect roi = mediapipe::GetRoi(image.width(), image.height(), norm_rect);
  MP_ASSIGN_OR_RETURN(auto padding, mediapipe::PadRoi(tensor_width, tensor_height, options.keep_aspect_ratio(), &roi));
  result->padding = padding;
  mediapipe::GetRotatedSubRectToRectTransformMatrix(
      roi, image.width(), image.height(), /*flip_horizontally=*/false, &result->matrix);
  MP_RETURN_IF_ERROR(InitConverterIfNecessary(image, options, params, gpu_converter, cpu_converter));
  mediapipe::Tensor::ElementType output_tensor_type = mediapipe::GetOutputTensorType(image.UsesGpu(), params);
  mediapipe::Tensor tensor(output_tensor_type, {1, tensor_height, tensor_width, mediapipe::GetNumOutputChannels(image)}, memory_manager);
  MP_RETURN_IF_ERROR((image.UsesGpu() ? gpu_converter.get() : cpu_converter.get())->Convert(
      image, roi, params.range_min, params.range_max, 0, tensor));
  result->tensor = std::move(tensor);
  result->tensors.clear();
  if (result->tensor) {
    result->tensors.emplace_back(std::move(*result->tensor));
  }
  return absl::OkStatus();
}

} // namespace api2
} // namespace mediapipe
