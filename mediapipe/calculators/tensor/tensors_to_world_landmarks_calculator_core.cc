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

#include "mediapipe/calculators/tensor/tensors_to_world_landmarks_calculator_core.h"
#include "mediapipe/framework/port/ret_check.h"
#include <cmath>

namespace hand_tracking_mp_lean {
namespace api2 {

absl::Status OutputTensorsToWorldLandmarks(
  const std::vector<Tensor>& input_tensors,
  LandmarkList* output_landmarks) {

  const int num_landmarks = 21;

  RET_CHECK(input_tensors[0].element_type() == Tensor::ElementType::kFloat32);
  int num_values = input_tensors[0].shape().num_elements();
  const int num_dimensions = num_values / num_landmarks;
  RET_CHECK_GT(num_dimensions, 0);

  auto view = input_tensors[0].GetCpuReadView();
  auto raw_landmarks = view.buffer<float>();

  output_landmarks->clear_landmark();

  for (int ld = 0; ld < num_landmarks; ++ld) {
    const int offset = ld * num_dimensions;
    Landmark* landmark = output_landmarks->add_landmark();

    landmark->set_x(raw_landmarks[offset]);
    if (num_dimensions > 1) {
      landmark->set_y(raw_landmarks[offset + 1]);
    }
    if (num_dimensions > 2) {
      landmark->set_z(raw_landmarks[offset + 2]);
    }
  }

  return absl::OkStatus();
}

TensorsToWorldLandmarksCore::TensorsToWorldLandmarksCore(
    ::hand_tracking_mp_lean::TensorsToLandmarksCalculatorOptions::Activation visibility_activation,
    ::hand_tracking_mp_lean::TensorsToLandmarksCalculatorOptions::Activation presence_activation,
    int num_landmarks)
    : num_landmarks_(num_landmarks),
      visibility_activation_(visibility_activation),
      presence_activation_(presence_activation) {}

/// merely puts the tensors into the mediapipe output object, no more.
absl::Status TensorsToWorldLandmarksCore::Process(
    const std::vector<Tensor>& input_tensors,
    LandmarkList* output_landmarks) {
   return OutputTensorsToWorldLandmarks(input_tensors, output_landmarks);
}


}  // namespace api2
}  // namespace hand_tracking_mp_lean
