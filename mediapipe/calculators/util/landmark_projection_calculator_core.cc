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

void ProjectXY(const NormalizedLandmark& lm,
               const std::array<float, 16>& matrix,
               NormalizedLandmark* out) {
  out->set_x(lm.x() * matrix[0] + lm.y() * matrix[1] + lm.z() * matrix[2] +
             matrix[3]);
  out->set_y(lm.x() * matrix[4] + lm.y() * matrix[5] + lm.z() * matrix[6] +
             matrix[7]);
}

float CalculateZScale(const std::array<float, 16>& matrix) {
  NormalizedLandmark a;
  a.set_x(0.0f);
  a.set_y(0.0f);
  NormalizedLandmark b;
  b.set_x(1.0f);
  b.set_y(0.0f);
  NormalizedLandmark a_projected;
  ProjectXY(a, matrix, &a_projected);
  NormalizedLandmark b_projected;
  ProjectXY(b, matrix, &b_projected);
  return std::sqrt(std::pow(b_projected.x() - a_projected.x(), 2) +
                   std::pow(b_projected.y() - a_projected.y(), 2));
}

std::function<void(const NormalizedLandmark&, NormalizedLandmark*)>
CreateProjectionFunction(
    const NormalizedRect* input_rect,
    bool ignore_rotation,
    const std::pair<int, int>* image_dimensions,
    const std::array<float, 16>* projection_matrix) {

  if (input_rect && !image_dimensions) {
    // Square ROI case
    ABSL_LOG_FIRST_N(WARNING, 1)
        << "Using NORM_RECT without IMAGE_DIMENSIONS is only "
           "supported for the square ROI. Provide "
           "IMAGE_DIMENSIONS or use PROJECTION_MATRIX.";

    return [input_rect, ignore_rotation](const NormalizedLandmark& landmark,
                                         NormalizedLandmark* new_landmark) {
      const float x = landmark.x() - 0.5f;
      const float y = landmark.y() - 0.5f;
      const float angle = ignore_rotation ? 0 : input_rect->rotation();
      float new_x = std::cos(angle) * x - std::sin(angle) * y;
      float new_y = std::sin(angle) * x + std::cos(angle) * y;

      new_x = new_x * input_rect->width() + input_rect->x_center();
      new_y = new_y * input_rect->height() + input_rect->y_center();
      const float new_z =
          landmark.z() * input_rect->width();  // Scale Z coordinate as X.

      *new_landmark = landmark;
      new_landmark->set_x(new_x);
      new_landmark->set_y(new_y);
      new_landmark->set_z(new_z);
    };
  } else if (input_rect && image_dimensions) {
    // Rect with image dimensions case
    RotatedRect rotated_rect = {
        /*center_x=*/input_rect->x_center() * image_dimensions->first,
        /*center_y=*/input_rect->y_center() * image_dimensions->second,
        /*width=*/input_rect->width() * image_dimensions->first,
        /*height=*/input_rect->height() * image_dimensions->second,
        /*rotation=*/ignore_rotation ? 0.0f : input_rect->rotation()};

    std::array<float, 16> project_mat;
    GetRotatedSubRectToRectTransformMatrix(
        rotated_rect, image_dimensions->first, image_dimensions->second,
        /*flip_horizontaly=*/false, &project_mat);
    const float z_scale = CalculateZScale(project_mat);

    return [project_mat, z_scale](const NormalizedLandmark& lm,
                                 NormalizedLandmark* new_landmark) {
      *new_landmark = lm;
      ProjectXY(lm, project_mat, new_landmark);
      new_landmark->set_z(z_scale * lm.z());
    };
  } else if (projection_matrix) {
    // Projection matrix case
    const float z_scale = CalculateZScale(*projection_matrix);
    return [projection_matrix, z_scale](const NormalizedLandmark& lm,
                                       NormalizedLandmark* new_landmark) {
      *new_landmark = lm;
      ProjectXY(lm, *projection_matrix, new_landmark);
      new_landmark->set_z(z_scale * lm.z());
    };
  }

  // Should not reach here as validation is done at caller level
  return nullptr;
}

void ProcessLandmarkList(
    const NormalizedLandmarkList& input_landmarks,
    const NormalizedRect* input_rect,
    bool ignore_rotation,
    const std::pair<int, int>* image_dimensions,
    const std::array<float, 16>* projection_matrix,
    NormalizedLandmarkList* output_landmarks) {

  auto project_fn = CreateProjectionFunction(input_rect, ignore_rotation,
                                             image_dimensions, projection_matrix);

  for (int j = 0; j < input_landmarks.landmark_size(); ++j) {
    const NormalizedLandmark& landmark = input_landmarks.landmark(j);
    NormalizedLandmark* new_landmark = output_landmarks->add_landmark();
    project_fn(landmark, new_landmark);
  }
}

}  // namespace mediapipe
