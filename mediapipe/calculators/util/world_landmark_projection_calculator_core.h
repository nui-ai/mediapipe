// Copyright 2025 The MediaPipe Authors.
// Licensed under the Apache License, Version 2.0.

#ifndef MEDIAPIPE_CALCULATORS_UTIL_WORLD_LANDMARK_PROJECTION_CALCULATOR_CORE_H_
#define MEDIAPIPE_CALCULATORS_UTIL_WORLD_LANDMARK_PROJECTION_CALCULATOR_CORE_H_

#include <functional>
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/rect.pb.h"

namespace mediapipe::api3 {

std::function<void(const mediapipe::Landmark&, mediapipe::Landmark*)>
CreateRotationFunction(const mediapipe::NormalizedRect* rect);

mediapipe::LandmarkList ProcessLandmarks(
    const mediapipe::LandmarkList& in_landmarks,
    const std::function<void(const mediapipe::Landmark&, mediapipe::Landmark*)>& rotate_fn);

} // namespace mediapipe::api3

#endif // MEDIAPIPE_CALCULATORS_UTIL_WORLD_LANDMARK_PROJECTION_CALCULATOR_CORE_H_
