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

namespace mediapipe_v01013_based {
namespace api2 {

namespace {

inline float Sigmoid(float value) { return 1.0f / (1.0f + std::exp(-value)); }

/// optionally applies sigmoid to the activation value of hand visibility/presence. our code paths never reach it.
float ProcessExtraActivation(::mediapipe_v01013_based::TensorsToLandmarksCalculatorOptions::Activation activation, float value) {
  switch (activation) {
    case TensorsToLandmarksCalculatorOptions::SIGMOID:
      return Sigmoid(value);
      break;
    default:
      return value;
  }
}

}  // namespace

TensorsToWorldLandmarksCore::TensorsToWorldLandmarksCore(
    ::mediapipe_v01013_based::TensorsToLandmarksCalculatorOptions::Activation visibility_activation,
    ::mediapipe_v01013_based::TensorsToLandmarksCalculatorOptions::Activation presence_activation,
    int num_landmarks)
    : num_landmarks_(num_landmarks),
      visibility_activation_(visibility_activation),
      presence_activation_(presence_activation) {}

/// merely puts the tensors into the mediapipe output object, no more.
absl::Status TensorsToWorldLandmarksCore::TensorsToWorldLandmarks(
    const std::vector<Tensor>& input_tensors,
    LandmarkList* output_landmarks) {

  RET_CHECK(input_tensors[0].element_type() == Tensor::ElementType::kFloat32);
  int num_values = input_tensors[0].shape().num_elements();
  const int num_dimensions = num_values / num_landmarks_;
  RET_CHECK_GT(num_dimensions, 0);

  auto view = input_tensors[0].GetCpuReadView();
  auto raw_landmarks = view.buffer<float>();

  output_landmarks->clear_landmark();

  for (int ld = 0; ld < num_landmarks_; ++ld) {
    const int offset = ld * num_dimensions;
    Landmark* landmark = output_landmarks->add_landmark();

    landmark->set_x(raw_landmarks[offset]);
    if (num_dimensions > 1) {
      landmark->set_y(raw_landmarks[offset + 1]);
    }
    if (num_dimensions > 2) {
      landmark->set_z(raw_landmarks[offset + 2]);
    }
    if (num_dimensions > 3) {  // we never get here, this extra signal is not directly intelligible and almost surely an a abandoned training objective of the network as trained. https://chatgpt.com/s/t_68fb6338573c81919bef075a6bce50a8.
      landmark->set_visibility(ProcessExtraActivation(visibility_activation_, raw_landmarks[offset + 3]));
    }
    if (num_dimensions > 4) {  // we never get here, this extra signal is not directly intelligible and almost surely an a abandoned training objective of the network as trained. https://chatgpt.com/s/t_68fb6338573c81919bef075a6bce50a8.
      landmark->set_presence(ProcessExtraActivation(presence_activation_, raw_landmarks[offset + 4]));
    }
  }

  return absl::OkStatus();
}

}  // namespace api2
}  // namespace mediapipe_v01013_based
