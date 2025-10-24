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

#ifndef MEDIAPIPE_CALCULATORS_UTIL_LANDMARK_PROJECTION_CALCULATOR_CORE_H_
#define MEDIAPIPE_CALCULATORS_UTIL_LANDMARK_PROJECTION_CALCULATOR_CORE_H_

#include <array>
#include <functional>
#include <utility>

#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/rect.pb.h"

namespace mediapipe {

// Projects XY coordinates of a landmark using a transformation matrix.
void ProjectXY(const NormalizedLandmark& lm,
               const std::array<float, 16>& matrix,
               NormalizedLandmark* out);

// Calculates Z scale factor based on the transformation matrix.
// Landmark's Z scale is equal to a relative (to image) width of region of
// interest used during detection. To calculate based on matrix:
// 1. Project (0,0) --- (1,0) segment using matrix.
// 2. Calculate length of the projected segment.
float CalculateZScale(const std::array<float, 16>& matrix);

// Creates a projection function based on the provided inputs.
// Only one of the input parameter groups should be non-null:
// - input_rect without image_dimensions (square ROI case)
// - input_rect with image_dimensions (general rect case)
// - projection_matrix (matrix-based projection)
std::function<void(const NormalizedLandmark&, NormalizedLandmark*)>
CreateProjectionFunction(
    const NormalizedRect* input_rect,
    bool ignore_rotation,
    const std::pair<int, int>* image_dimensions,
    const std::array<float, 16>* projection_matrix);

// Processes a landmark list using the inputs to build a projection function.
void ProcessLandmarkList(
    const NormalizedLandmarkList& input_landmarks,
    const NormalizedRect* input_rect,
    bool ignore_rotation,
    const std::pair<int, int>* image_dimensions,
    const std::array<float, 16>* projection_matrix,
    NormalizedLandmarkList* output_landmarks);

}  // namespace mediapipe

#endif  // MEDIAPIPE_CALCULATORS_UTIL_LANDMARK_PROJECTION_CALCULATOR_CORE_H_
