// Copyright 2025 The MediaPipe Authors.
// Licensed under the Apache License, Version 2.0.

#ifndef MEDIAPIPE_CALCULATORS_UTIL_WORLD_LANDMARK_PROJECTION_CALCULATOR_CORE_H_
#define MEDIAPIPE_CALCULATORS_UTIL_WORLD_LANDMARK_PROJECTION_CALCULATOR_CORE_H_

#include <functional>
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/rect.pb.h"

namespace mediapipe_v01013_based::api3 {

std::function<void(const mediapipe_v01013_based::Landmark&, mediapipe_v01013_based::Landmark*)>
CreateRotationFunction(const mediapipe_v01013_based::NormalizedRect* rect);

LandmarkList Process(
    const mediapipe_v01013_based::LandmarkList& in_landmarks,
    const NormalizedRect *hand_rect);

} // namespace mediapipe_v01013_based::api3

#endif // MEDIAPIPE_CALCULATORS_UTIL_WORLD_LANDMARK_PROJECTION_CALCULATOR_CORE_H_
