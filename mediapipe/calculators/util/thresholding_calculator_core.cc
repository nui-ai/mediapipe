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

#include "mediapipe/calculators/util/thresholding_calculator_core.h"

#include "mediapipe/calculators/util/thresholding_calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"

namespace mediapipe_v01013_based {
namespace thresholding_calculator {

absl::Status InitializeCalculator(const ThresholdingCalculatorOptions& options,
                                 bool has_threshold_tag,
                                 bool has_threshold_side_packet,
                                 double& threshold) {
  if (options.has_threshold()) {
    if (has_threshold_tag) {
      return absl::InvalidArgumentError(
          "Using both the threshold option and input stream is not supported.");
    }
    if (has_threshold_side_packet) {
      return absl::InvalidArgumentError(
          "Using both the threshold option and input side packet is not supported.");
    }
    threshold = options.threshold();
  }
  return absl::OkStatus();
}

bool ProcessCalculator(double& threshold,
                      bool has_threshold_tag,
                      bool threshold_input_empty,
                      double threshold_input_value,
                      float input_float_value) {
  if (has_threshold_tag && !threshold_input_empty) {
    threshold = threshold_input_value;
  }

  // Compare the input float value to the threshold
  return static_cast<double>(input_float_value) > threshold;
}

}  // namespace thresholding_calculator
}  // namespace mediapipe_v01013_based
