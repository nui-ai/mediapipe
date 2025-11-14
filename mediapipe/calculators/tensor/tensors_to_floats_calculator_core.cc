// Copyright 2019 The MediaPipe Authors.
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

#include "mediapipe/calculators/tensor/tensors_to_floats_calculator_core.h"

namespace hand_tracking_mp_lean {
namespace tensors_to_floats_calculator_core {

namespace {

inline float Sigmoid(float value) { return 1.0f / (1.0f + std::exp(-value)); }

}  // namespace

absl::Status Open(const ::hand_tracking_mp_lean::TensorsToFloatsCalculatorOptions& options) {
  // Nothing to do here, just return OK status
  return absl::OkStatus();
}

ProcessingResult HandPresenceExtract(
    const std::vector<Tensor>& input_tensors,
    const ::hand_tracking_mp_lean::TensorsToFloatsCalculatorOptions& options) {  // the options arg is no longer necessary actually

  ProcessingResult result;

  if (input_tensors.empty()) {
    result.status = absl::Status(absl::StatusCode::kInvalidArgument,
                                "Input tensor vector is empty");
    return result;
  }

  if (input_tensors[0].element_type() != Tensor::ElementType::kFloat32) {
    result.status = absl::Status(absl::StatusCode::kInvalidArgument,
                               "Input tensor element type is not float32");
    return result;
  }

  // Extract data from tensor
  auto view = input_tensors[0].GetCpuReadView();
  auto raw_floats = view.buffer<float>();
  int num_values = input_tensors[0].shape().num_elements();
  result.output_floats = std::make_unique<std::vector<float>>(
      raw_floats, raw_floats + num_values);
  result.num_values = num_values;

  // Apply activation if specified
  switch (options.activation()) {
    case TensorsToFloatsCalculatorOptions::SIGMOID:
      std::transform(result.output_floats->begin(), result.output_floats->end(),
                   result.output_floats->begin(), Sigmoid);
      break;
    case TensorsToFloatsCalculatorOptions::NONE:
      break;
  }

  result.status = absl::OkStatus();
  return result;
}

}  // namespace tensors_to_floats_calculator_core
}  // namespace hand_tracking_mp_lean
