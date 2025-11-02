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

#ifndef MEDIAPIPE_CALCULATORS_TENSOR_TENSORS_TO_FLOATS_CALCULATOR_CORE_H_
#define MEDIAPIPE_CALCULATORS_TENSOR_TENSORS_TO_FLOATS_CALCULATOR_CORE_H_

#include <memory>
#include <vector>

#include "mediapipe/calculators/tensor/tensors_to_floats_calculator.pb.h"
#include "mediapipe/framework/formats/tensor.h"
#include "absl/status/status.h"

namespace mediapipe_v01013_based {
namespace tensors_to_floats_calculator_core {

// Struct to hold the result of processing
struct ProcessingResult {
  // Status of the processing operation
  absl::Status status;
  // The extracted float values
  std::unique_ptr<std::vector<float>> output_floats;
  // Number of values in the tensor
  int num_values;
};

// Open function that processes the calculator options
absl::Status Open(const ::mediapipe_v01013_based::TensorsToFloatsCalculatorOptions& options);

// Process function that handles the tensor processing
ProcessingResult Process(
    const std::vector<Tensor>& input_tensors,
    const ::mediapipe_v01013_based::TensorsToFloatsCalculatorOptions& options);

}  // namespace tensors_to_floats_calculator_core
}  // namespace mediapipe_v01013_based

#endif  // MEDIAPIPE_CALCULATORS_TENSOR_TENSORS_TO_FLOATS_CALCULATOR_CORE_H_
