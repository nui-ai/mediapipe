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

#include "mediapipe/calculators/util/landmark_letterbox_removal_calculator_core.h"

#include <array>

namespace mediapipe {

NormalizedLandmark AdjustLandmarkForLetterboxRemoval(
    const NormalizedLandmark& landmark,
    float left, float top, float left_and_right, float top_and_bottom) {
  NormalizedLandmark new_landmark = landmark;
  const float new_x = (landmark.x() - left) / (1.0f - left_and_right);
  const float new_y = (landmark.y() - top) / (1.0f - top_and_bottom);
  const float new_z = landmark.z() / (1.0f - left_and_right);  // Scale Z coordinate as X.

  new_landmark.set_x(new_x);
  new_landmark.set_y(new_y);
  new_landmark.set_z(new_z);

  return new_landmark;
}

NormalizedLandmarkList AdjustLandmarkListForLetterboxRemoval(
    const NormalizedLandmarkList& input_landmarks,
    const std::array<float, 4>& letterbox_padding) {
  const float left = letterbox_padding[0];
  const float top = letterbox_padding[1];
  const float left_and_right = letterbox_padding[0] + letterbox_padding[2];
  const float top_and_bottom = letterbox_padding[1] + letterbox_padding[3];

  NormalizedLandmarkList output_landmarks;
  for (int i = 0; i < input_landmarks.landmark_size(); ++i) {
    const NormalizedLandmark& landmark = input_landmarks.landmark(i);
    NormalizedLandmark* new_landmark = output_landmarks.add_landmark();
    *new_landmark = AdjustLandmarkForLetterboxRemoval(
        landmark, left, top, left_and_right, top_and_bottom);
  }

  return output_landmarks;
}

}  // namespace mediapipe
