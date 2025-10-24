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

namespace mediapipe {
namespace api2 {

namespace {

inline float Sigmoid(float value) { return 1.0f / (1.0f + std::exp(-value)); }

float ApplyActivation(
    TensorsToLandmarksCalculatorOptions::Activation activation, float value) {
  switch (activation) {
    case TensorsToLandmarksCalculatorOptions::SIGMOID:
      return Sigmoid(value);
      break;
    default:
      return value;
  }
}

}  // namespace

absl::Status ValidateCalculatorOptions(
    const ::mediapipe::TensorsToLandmarksCalculatorOptions& options,
    bool output_normalized_landmarks,
    bool output_landmarks,
    bool has_flip_inputs) {
  if (output_normalized_landmarks) {
    RET_CHECK(options.has_input_image_height() &&
              options.has_input_image_width())
        << "Must provide input width/height for getting normalized landmarks.";
  }
  if (output_landmarks &&
      (options.flip_horizontally() || options.flip_vertically() ||
       has_flip_inputs)) {
    RET_CHECK(options.has_input_image_height() &&
              options.has_input_image_width())
        << "Must provide input width/height for using flipping when outputting "
           "landmarks in absolute coordinates.";
  }
  return absl::OkStatus();
}

absl::Status ProcessTensorsToLandmarks(
    const std::vector<Tensor>& input_tensors,
    const ::mediapipe::TensorsToLandmarksCalculatorOptions& options,
    int num_landmarks,
    bool flip_horizontally,
    bool flip_vertically,
    LandmarkList* output_landmarks,
    NormalizedLandmarkList* output_norm_landmarks) {
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

    if (flip_horizontally) {
      landmark->set_x(options.input_image_width() - raw_landmarks[offset]);
    } else {
      landmark->set_x(raw_landmarks[offset]);
    }
    if (num_dimensions > 1) {
      if (flip_vertically) {
        landmark->set_y(options.input_image_height() -
                        raw_landmarks[offset + 1]);
      } else {
        landmark->set_y(raw_landmarks[offset + 1]);
      }
    }
    if (num_dimensions > 2) {
      landmark->set_z(raw_landmarks[offset + 2]);
    }
    if (num_dimensions > 3) {
      landmark->set_visibility(ApplyActivation(options.visibility_activation(),
                                               raw_landmarks[offset + 3]));
    }
    if (num_dimensions > 4) {
      landmark->set_presence(ApplyActivation(options.presence_activation(),
                                             raw_landmarks[offset + 4]));
    }
  }

  // Generate normalized landmarks if requested.
  if (output_norm_landmarks != nullptr) {
    output_norm_landmarks->clear_landmark();
    for (int i = 0; i < output_landmarks->landmark_size(); ++i) {
      const Landmark& landmark = output_landmarks->landmark(i);
      NormalizedLandmark* norm_landmark = output_norm_landmarks->add_landmark();
      norm_landmark->set_x(landmark.x() / options.input_image_width());
      norm_landmark->set_y(landmark.y() / options.input_image_height());
      // Scale Z coordinate as X + allow additional uniform normalization.
      norm_landmark->set_z(landmark.z() / options.input_image_width() /
                           options.normalize_z());
      if (landmark.has_visibility()) {  // Set only if supported in the model.
        norm_landmark->set_visibility(landmark.visibility());
      }
      if (landmark.has_presence()) {  // Set only if supported in the model.
        norm_landmark->set_presence(landmark.presence());
      }
    }
  }

  return absl::OkStatus();
}

}  // namespace api2
}  // namespace mediapipe
