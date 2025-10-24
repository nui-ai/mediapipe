// Copyright 2025 The MediaPipe Authors.
// Licensed under the Apache License, Version 2.0.

#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "absl/log/absl_log.h"
#include <cmath>
#include <functional>
#include <glog/log_severity.h>

namespace mediapipe::api3 {

/// helper function returning a rotation function which takes a landmark and rotates it the same
/// as the rotation which the rectangle object containing it has (if rect is provided, otherwise returns nullptr).
/// this function is then applied to each landmark contained in the rectangle.
/// it should always have rect provided and yield a function, in our pipeline.
std::function<void(const Landmark&, Landmark*)> CreateRotationFunction(const mediapipe::NormalizedRect* rect) {
  if (!rect) return nullptr;
  const float cosa = std::cos(rect->rotation());
  const float sina = std::sin(rect->rotation());
  return [cosa, sina](const Landmark& in_landmark, Landmark* out_landmark) {
    out_landmark->set_x(cosa * in_landmark.x() - sina * in_landmark.y());
    out_landmark->set_y(sina * in_landmark.x() + cosa * in_landmark.y());
  };
}

/// applies the given rectangle's rotation, to each landmark, that's all
LandmarkList Process(const LandmarkList& in_landmarks, const NormalizedRect *hand_rect) {

  auto landmark_rotate = CreateRotationFunction(hand_rect);

  mediapipe::LandmarkList out_landmarks;
  for (int i = 0; i < in_landmarks.landmark_size(); ++i) {
    const auto& in_landmark = in_landmarks.landmark(i);
    Landmark* out_landmark = out_landmarks.add_landmark();
    *out_landmark = in_landmark;
    if (landmark_rotate) {
      landmark_rotate(in_landmark, out_landmark);
    }
  }
  return out_landmarks;
}

} // namespace mediapipe::api3
