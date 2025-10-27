//
// Created by matan on 10/27/25.
//

#include "mediapipe/liberated/liberated.h"

namespace mediapipe {

Liberated::Liberated(MemoryManager* memory_manager) {

  // Configure fixed 192x192 core options (mirrors HeadCalculator::Open).
  auto options = ImageToTensorCalculatorOptions();
  options.set_output_tensor_width(192);
  options.set_output_tensor_height(192);
  options.set_keep_aspect_ratio(true);
  options.mutable_output_tensor_float_range()->set_min(0.0f);
  options.mutable_output_tensor_float_range()->set_max(1.0f);
  options.set_border_mode(mediapipe::ImageToTensorCalculatorOptions::BORDER_ZERO);

  auto params = GetOutputTensorParams(options);
  int tensor_width = params.output_width.value_or(0);
  int tensor_height = params.output_height.value_or(0);

  image_to_tensor_core_ = std::make_unique<api2::ImageToTensorCalculatorCore>(
      options, tensor_width, tensor_height, params,
      gpu_converter_, cpu_converter_, memory_manager);
}

}  // namespace mediapipe