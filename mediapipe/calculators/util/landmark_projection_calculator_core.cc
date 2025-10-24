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

#include "mediapipe/calculators/util/landmark_projection_calculator_core.h"

#include <array>
#include <cmath>
#include <functional>
#include <utility>

#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "mediapipe/calculators/tensor/image_to_tensor_utils.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/rect.pb.h"

namespace mediapipe {

/// adapts the landmark coordinates originating in the inference model's output, to the image coordinates ―
/// this completes the work of TensorsToLandmarksCalculator which performs initial steps for this adaptation
/// (under normal cohesion it would leave that to the current calculator).
///
/// • applies the rotation of the rectangle
/// • scales the rotated coordinates by the width and height of the rectangle
/// • shifts the landmark coordinates so that (0.5, 0.5) is the origin, thus centering.
/// • shifts the scaled coordinates to the rectangle’s center
/// • scales the z-coordinate by the rectangle’s width.
///
/// the above work is a bit more than the world landmarks set's parallel handling,
/// where there is no real shifting and scaling work to be done, as the latter
/// has its inference of the object's shape fully-baked in the neural model.
///
/// the current function, handling the viewport landmarks, just needs this
/// little extra effort to stretch and shift the network's viewport landmakrs
/// outputs back into the only natural viewport interpretation of them that's all!
///
/// assumes a square rectangle as input! our lanamarks inference model only takes a square one anyway so that's always the case.
std::function<void(const NormalizedLandmark&, NormalizedLandmark*)> CreateProjectionFunction(const NormalizedRect* input_rect) {

   return [input_rect](const NormalizedLandmark& landmark, NormalizedLandmark* new_landmark) {
      const float x = landmark.x() - 0.5f;
      const float y = landmark.y() - 0.5f;
      const float angle = input_rect->rotation();
      float new_x = std::cos(angle) * x - std::sin(angle) * y;
      float new_y = std::sin(angle) * x + std::cos(angle) * y;

      new_x = new_x * input_rect->width() + input_rect->x_center();
      new_y = new_y * input_rect->height() + input_rect->y_center();
      const float new_z = landmark.z() * input_rect->width();  // Scale Z coordinate as X.

      *new_landmark = landmark;
      new_landmark->set_x(new_x);
      new_landmark->set_y(new_y);
      new_landmark->set_z(new_z);
    };
}

void ProcessLandmarkList(
    const NormalizedLandmarkList& input_landmarks,
    const NormalizedRect* input_rect,
    NormalizedLandmarkList* output_landmarks) {

  auto project_fn = CreateProjectionFunction(input_rect);

  for (int j = 0; j < input_landmarks.landmark_size(); ++j) {
    const NormalizedLandmark& landmark = input_landmarks.landmark(j);
    NormalizedLandmark* new_landmark = output_landmarks->add_landmark();
    project_fn(landmark, new_landmark);
  }
}

}  // namespace mediapipe
