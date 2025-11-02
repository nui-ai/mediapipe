#pragma once
#include <array>
#include <memory>
#include <optional>
#include <vector>
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/calculators/tensor/image_to_tensor_converter.h"
#include "mediapipe/calculators/tensor/image_to_tensor_calculator.pb.h"
#include "mediapipe/calculators/tensor/image_to_tensor_utils.h"
#include "absl/status/status.h"
#include "absl/types/optional.h"

namespace mediapipe_v01013_based {
namespace api2 {

struct ImageToTensorCoreResult {
  std::array<float, 4> padding;
  std::array<float, 16> matrix;
  std::vector<Tensor> tensors;
  std::optional<Tensor> tensor;
};

// Refactored into a class: constructor captures configuration and resources;
// Process executes on a given image and optional normalized rect.
class ImageToTensorCalculatorCore {
 public:
  ImageToTensorCalculatorCore(
      const mediapipe_v01013_based::ImageToTensorCalculatorOptions& options,
      int tensor_width,
      int tensor_height,
      const OutputTensorParams& params,
      std::unique_ptr<ImageToTensorConverter>& gpu_converter,
      std::unique_ptr<ImageToTensorConverter>& cpu_converter,
      mediapipe_v01013_based::MemoryManager* memory_manager);

  absl::Status Process(const mediapipe_v01013_based::Image& image,
                       absl::optional<mediapipe_v01013_based::NormalizedRect> norm_rect,
                       ImageToTensorCoreResult* result);

 private:
  const mediapipe_v01013_based::ImageToTensorCalculatorOptions options_;
  const int tensor_width_;
  const int tensor_height_;
  const OutputTensorParams params_;
  // References to converters so the core can lazily initialize them.
  std::unique_ptr<ImageToTensorConverter>& gpu_converter_;
  std::unique_ptr<ImageToTensorConverter>& cpu_converter_;
  mediapipe_v01013_based::MemoryManager* memory_manager_ = nullptr;
};

}  // namespace api2
}  // namespace mediapipe_v01013_based