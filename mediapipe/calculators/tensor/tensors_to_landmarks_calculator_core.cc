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

#include "mediapipe/calculators/tensor/tensors_to_landmarks_calculator_core.h"
#include "mediapipe/framework/port/ret_check.h"
#include <cmath>

namespace mediapipe_v01013_based {
namespace api2 {

namespace {

inline float Sigmoid(float value) { return 1.0f / (1.0f + std::exp(-value)); }

/// optionally applies sigmoid to the input activation value. our code paths never reach it.
float ProcessExtraActivation(
    ::mediapipe_v01013_based::TensorsToLandmarksCalculatorOptions::Activation activation, float value) {
  switch (activation) {
    case ::mediapipe_v01013_based::TensorsToLandmarksCalculatorOptions::SIGMOID:
      // ABSL_LOG(INFO) << "activation is sigmoid";
      return Sigmoid(value);
      break;
    default:
      return Sigmoid(value);
      // return value;
  }
}

}  // namespace

TensorsToLandmarksCore::TensorsToLandmarksCore(
    int input_image_width, int input_image_height,
    ::mediapipe_v01013_based::TensorsToLandmarksCalculatorOptions::Activation visibility_activation,
    ::mediapipe_v01013_based::TensorsToLandmarksCalculatorOptions::Activation presence_activation,
    float normalize_z,
    int num_landmarks)
    : input_image_width_(input_image_width),
      input_image_height_(input_image_height),
      num_landmarks_(num_landmarks),
      visibility_activation_(visibility_activation),
      presence_activation_(presence_activation),
      normalize_z_(normalize_z) {}

absl::Status TensorsToLandmarksCore::TensorsToLandmarks(
    const std::vector<Tensor>& input_tensors,
    NormalizedLandmarkList* output_norm_landmarks) {

  RET_CHECK(input_tensors[0].element_type() == Tensor::ElementType::kFloat32);
  int num_values = input_tensors[0].shape().num_elements();
  const int num_dimensions = num_values / num_landmarks_;
  RET_CHECK_GT(num_dimensions, 0);

  auto view = input_tensors[0].GetCpuReadView();
  auto raw_landmarks = view.buffer<float>();

  LandmarkList output_landmarks;
  LandmarkList* output_landmarks_ptr = &output_landmarks;
  output_landmarks_ptr->clear_landmark();

  // fill the landmarks output collection
  for (int ld = 0; ld < num_landmarks_; ++ld) {
    const int offset = ld * num_dimensions;
    Landmark* landmark = output_landmarks_ptr->add_landmark();

    landmark->set_x(raw_landmarks[offset]);
    if (num_dimensions > 1) {
      landmark->set_y(raw_landmarks[offset + 1]);
    }
    if (num_dimensions > 2) {
      landmark->set_z(raw_landmarks[offset + 2]);
    }
    if (num_dimensions > 3) {  // we never get here, this extra signal is not directly intelligible and almost surely an a abandoned training objective of the network as trained. https://chatgpt.com/s/t_68fb6338573c81919bef075a6bce50a8.
      auto visibility = ProcessExtraActivation(visibility_activation_, raw_landmarks[offset + 3]);
      landmark->set_visibility(visibility);
    }
    if (num_dimensions > 4) {  // we never get here, this extra signal is not directly intelligible and almost surely an a abandoned training objective of the network as trained. https://chatgpt.com/s/t_68fb6338573c81919bef075a6bce50a8.
      auto presence = ProcessExtraActivation(presence_activation_, raw_landmarks[offset + 4]);
      landmark->set_presence(presence);
    }
  }

  // fill the normalized output collection.
  // they are just normalized to the image dimensions to range between 0 and 1, and by a (rather arbitrary) constant for Z.
  // the constant for Z doesn't matter to us since we consider these Z values of no relevant semantics.
  output_norm_landmarks->clear_landmark();
  for (int i = 0; i < output_landmarks_ptr->landmark_size(); ++i) {
    const Landmark& landmark = output_landmarks_ptr->landmark(i);
    NormalizedLandmark* norm_landmark = output_norm_landmarks->add_landmark();
    norm_landmark->set_x(landmark.x() / input_image_width_);
    norm_landmark->set_y(landmark.y() / input_image_height_);

    // Scale Z coordinate as X + allow additional uniform normalization.
    norm_landmark->set_z(landmark.z() / input_image_width_ / normalize_z_);

    if (landmark.has_visibility()) {  norm_landmark->set_visibility(landmark.visibility()); }
    if (landmark.has_presence()) {  norm_landmark->set_presence(landmark.presence()); }
  }

  return absl::OkStatus();
}

}  // namespace api2
}  // namespace mediapipe_v01013_based
