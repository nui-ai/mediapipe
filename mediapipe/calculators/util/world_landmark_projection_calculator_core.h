// Copyright 2025 The MediaPipe Authors.
// Licensed under the Apache License, Version 2.0.

#ifndef MEDIAPIPE_CALCULATORS_UTIL_WORLD_LANDMARK_PROJECTION_CALCULATOR_CORE_H_
#define MEDIAPIPE_CALCULATORS_UTIL_WORLD_LANDMARK_PROJECTION_CALCULATOR_CORE_H_

#include <functional>
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/rect.pb.h"

namespace hand_tracking_mp_lean::api3 {

std::function<void(const hand_tracking_mp_lean::Landmark&, hand_tracking_mp_lean::Landmark*)>
CreateRotationFunction(const hand_tracking_mp_lean::NormalizedRect* rect);

LandmarkList RotateWorldLandmarks(
    const hand_tracking_mp_lean::LandmarkList& in_landmarks,
    const NormalizedRect *hand_rect);

} // namespace hand_tracking_mp_lean::api3

#endif // MEDIAPIPE_CALCULATORS_UTIL_WORLD_LANDMARK_PROJECTION_CALCULATOR_CORE_H_
