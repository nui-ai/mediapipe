// Copyright 2020 The MediaPipe Authors.
// Licensed under the Apache License, Version 2.0.
#ifndef MEDIAPIPE_MODULES_HAND_LANDMARK_CALCULATORS_HAND_LANDMARKS_TO_RECT_CALCULATOR_CORE_H_
#define MEDIAPIPE_MODULES_HAND_LANDMARK_CALCULATORS_HAND_LANDMARKS_TO_RECT_CALCULATOR_CORE_H_

#include <utility>
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/framework/port/status.h"

namespace mediapipe {

NormalizedLandmarkList GetPartialLandmarks(const NormalizedLandmarkList& landmarks);
absl::Status ComputeHandRect(const NormalizedLandmarkList& landmarks,
                             const std::pair<int, int>& image_size,
                             NormalizedRect* rect);
absl::Status NormalizedLandmarkListToRect(
    const NormalizedLandmarkList& landmarks,
    const std::pair<int, int>& image_size, NormalizedRect* rect);

} // namespace mediapipe

#endif  // MEDIAPIPE_MODULES_HAND_LANDMARK_CALCULATORS_HAND_LANDMARKS_TO_RECT_CALCULATOR_CORE_H_
