// Copyright 2025 The MediaPipe Authors.
// Licensed under the Apache License, Version 2.0.

#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include <cmath>
#include <functional>

namespace mediapipe::api3 {

// Returns a rotation function if rect is provided, otherwise nullptr.
std::function<void(const mediapipe::Landmark&, mediapipe::Landmark*)> CreateRotationFunction(const mediapipe::NormalizedRect* rect) {
  if (!rect) return nullptr;
  const float cosa = std::cos(rect->rotation());
  const float sina = std::sin(rect->rotation());
  return [cosa, sina](const mediapipe::Landmark& in_landmark, mediapipe::Landmark* out_landmark) {
    out_landmark->set_x(cosa * in_landmark.x() - sina * in_landmark.y());
    out_landmark->set_y(sina * in_landmark.x() + cosa * in_landmark.y());
  };
}

// Processes landmarks, applies rotation if rotate_fn is provided.
mediapipe::LandmarkList ProcessLandmarks(const mediapipe::LandmarkList& in_landmarks,
                              const std::function<void(const mediapipe::Landmark&, mediapipe::Landmark*)>& rotate_fn) {
  mediapipe::LandmarkList out_landmarks;
  for (int i = 0; i < in_landmarks.landmark_size(); ++i) {
    const auto& in_landmark = in_landmarks.landmark(i);
    mediapipe::Landmark* out_landmark = out_landmarks.add_landmark();
    *out_landmark = in_landmark;
    if (rotate_fn) {
      rotate_fn(in_landmark, out_landmark);
    }
  }
  return out_landmarks;
}

} // namespace mediapipe::api3
