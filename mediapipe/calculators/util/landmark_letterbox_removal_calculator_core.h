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

#ifndef MEDIAPIPE_CALCULATORS_UTIL_LANDMARK_LETTERBOX_REMOVAL_CALCULATOR_CORE_H_
#define MEDIAPIPE_CALCULATORS_UTIL_LANDMARK_LETTERBOX_REMOVAL_CALCULATOR_CORE_H_

#include <array>

#include "mediapipe/framework/formats/landmark.pb.h"

namespace mediapipe {

// Adjusts a single landmark's coordinates based on letterbox padding.
// Returns a new NormalizedLandmark with adjusted coordinates.
NormalizedLandmark AdjustLandmarkForLetterboxRemoval(
    const NormalizedLandmark& landmark,
    float left, float top, float left_and_right, float top_and_bottom);

// Adjusts all landmarks in a NormalizedLandmarkList based on letterbox padding.
// Returns a new NormalizedLandmarkList with adjusted coordinates.
NormalizedLandmarkList AdjustLandmarkListForLetterboxRemoval(
    const NormalizedLandmarkList& input_landmarks,
    const std::array<float, 4>& letterbox_padding);

}  // namespace mediapipe

#endif  // MEDIAPIPE_CALCULATORS_UTIL_LANDMARK_LETTERBOX_REMOVAL_CALCULATOR_CORE_H_
