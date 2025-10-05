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

#ifndef MEDIAPIPE_CALCULATORS_UTIL_THRESHOLDING_CALCULATOR_CORE_H_
#define MEDIAPIPE_CALCULATORS_UTIL_THRESHOLDING_CALCULATOR_CORE_H_

#include "absl/status/status.h"

namespace mediapipe {

class ThresholdingCalculatorOptions;

namespace thresholding_calculator {

// Initializes the calculator by setting up the threshold value from various possible sources.
absl::Status InitializeCalculator(const ThresholdingCalculatorOptions& options,
                                 bool has_threshold_tag,
                                 bool has_threshold_side_packet,
                                 double& threshold);

// Processes the input float value, updating the threshold if needed, and returns whether the
// value exceeds the threshold.
bool ProcessCalculator(double& threshold,
                      bool has_threshold_tag,
                      bool threshold_input_empty,
                      double threshold_input_value,
                      float input_float_value);

}  // namespace thresholding_calculator
}  // namespace mediapipe

#endif  // MEDIAPIPE_CALCULATORS_UTIL_THRESHOLDING_CALCULATOR_CORE_H_
