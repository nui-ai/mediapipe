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

#ifndef MEDIAPIPE_CALCULATORS_TENSOR_TENSORS_TO_LANDMARKS_CALCULATOR_CORE_H_
#define MEDIAPIPE_CALCULATORS_TENSOR_TENSORS_TO_LANDMARKS_CALCULATOR_CORE_H_

#include "mediapipe/calculators/tensor/tensors_to_landmarks_calculator.pb.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/tensor.h"
#include "absl/status/status.h"

namespace mediapipe {
namespace api2 {

// Core processing function extracted from Process method
absl::Status TensorsToLandmarks(
    const std::vector<Tensor>& input_tensors,
    const ::mediapipe::TensorsToLandmarksCalculatorOptions& options,
    int num_landmarks,
    bool flip_horizontally,
    bool flip_vertically,
    LandmarkList* output_landmarks,
    NormalizedLandmarkList* output_norm_landmarks = nullptr);

}  // namespace api2
}  // namespace mediapipe

#endif  // MEDIAPIPE_CALCULATORS_TENSOR_TENSORS_TO_LANDMARKS_CALCULATOR_CORE_H_
