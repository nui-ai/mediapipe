// Copyright 2020 The MediaPipe Authors.
// Licensed under the Apache License, Version 2.0.
#include "mediapipe/modules/hand_landmark/calculators/hand_landmarks_to_rect_calculator_core.h"
#include <vector>

namespace mediapipe_v01013_based {
constexpr int kNumLandmarks = 21;

// Partial landmark indices for legacy behavior.
static constexpr int kPartialLandmarkIndices[]{0, 1, 2, 3, 5, 6, 9, 10, 13, 14, 17, 18};

// Extracts partial landmarks if input is full set, else returns input.
NormalizedLandmarkList GetPartialLandmarks(const NormalizedLandmarkList& landmarks) {
  if (landmarks.landmark_size() == kNumLandmarks) {
    NormalizedLandmarkList partial_landmarks;
    for (int i : kPartialLandmarkIndices) {
      *partial_landmarks.add_landmark() = landmarks.landmark(i);
    }
    return partial_landmarks;
  } else {
    return landmarks;
  }
}

// Computes the hand rect from landmarks and image size.
absl::Status ComputeHandRect(const NormalizedLandmarkList& landmarks,
                             const std::pair<int, int>& image_size,
                             NormalizedRect* rect) {
  extern absl::Status NormalizedLandmarkListToRect(
      const NormalizedLandmarkList&, const std::pair<int, int>&, NormalizedRect*);
  return NormalizedLandmarkListToRect(landmarks, image_size, rect);
}

// Unified API: derives the necessary landmarks and computes the hand rect.
absl::Status AdjustHandRectByInferredLanmdarks(const NormalizedLandmarkList& input_landmarks,
                          const std::pair<int, int>& image_size,
                          NormalizedRect* rect) {
  const auto partial = GetPartialLandmarks(input_landmarks);
  return ComputeHandRect(partial, image_size, rect);
}

float NormalizeRadians(float angle) {
  return angle - 2 * M_PI * std::floor((angle - (-M_PI)) / (2 * M_PI));
}

float ComputeRotation(const NormalizedLandmarkList& landmarks,
                      const std::pair<int, int>& image_size) {
  constexpr int kWristJoint = 0;
  constexpr int kMiddleFingerPIPJoint = 6;
  constexpr int kIndexFingerPIPJoint = 4;
  constexpr int kRingFingerPIPJoint = 8;
  constexpr float kTargetAngle = M_PI * 0.5f;

  const float x0 = landmarks.landmark(kWristJoint).x() * image_size.first;
  const float y0 = landmarks.landmark(kWristJoint).y() * image_size.second;

  float x1 = (landmarks.landmark(kIndexFingerPIPJoint).x() +
              landmarks.landmark(kRingFingerPIPJoint).x()) / 2.f;
  float y1 = (landmarks.landmark(kIndexFingerPIPJoint).y() +
              landmarks.landmark(kRingFingerPIPJoint).y()) / 2.f;
  x1 = (x1 + landmarks.landmark(kMiddleFingerPIPJoint).x()) / 2.f * image_size.first;
  y1 = (y1 + landmarks.landmark(kMiddleFingerPIPJoint).y()) / 2.f * image_size.second;

  const float rotation = NormalizeRadians(kTargetAngle - std::atan2(-(y1 - y0), x1 - x0));
  return rotation;
}

absl::Status NormalizedLandmarkListToRect(
    const NormalizedLandmarkList& landmarks,
    const std::pair<int, int>& image_size, NormalizedRect* rect) {
  const float rotation = ComputeRotation(landmarks, image_size);
  const float reverse_angle = NormalizeRadians(-rotation);

  float max_x = std::numeric_limits<float>::min();
  float max_y = std::numeric_limits<float>::min();
  float min_x = std::numeric_limits<float>::max();
  float min_y = std::numeric_limits<float>::max();
  for (int i = 0; i < landmarks.landmark_size(); ++i) {
    max_x = std::max(max_x, landmarks.landmark(i).x());
    max_y = std::max(max_y, landmarks.landmark(i).y());
    min_x = std::min(min_x, landmarks.landmark(i).x());
    min_y = std::min(min_y, landmarks.landmark(i).y());
  }
  const float axis_aligned_center_x = (max_x + min_x) / 2.f;
  const float axis_aligned_center_y = (max_y + min_y) / 2.f;

  max_x = std::numeric_limits<float>::min();
  max_y = std::numeric_limits<float>::min();
  min_x = std::numeric_limits<float>::max();
  min_y = std::numeric_limits<float>::max();
  for (int i = 0; i < landmarks.landmark_size(); ++i) {
    const float original_x = (landmarks.landmark(i).x() - axis_aligned_center_x) * image_size.first;
    const float original_y = (landmarks.landmark(i).y() - axis_aligned_center_y) * image_size.second;

    const float projected_x = original_x * std::cos(reverse_angle) - original_y * std::sin(reverse_angle);
    const float projected_y = original_x * std::sin(reverse_angle) + original_y * std::cos(reverse_angle);

    max_x = std::max(max_x, projected_x);
    max_y = std::max(max_y, projected_y);
    min_x = std::min(min_x, projected_x);
    min_y = std::min(min_y, projected_y);
  }
  const float projected_center_x = (max_x + min_x) / 2.f;
  const float projected_center_y = (max_y + min_y) / 2.f;

  const float center_x = projected_center_x * std::cos(rotation) - projected_center_y * std::sin(rotation) + image_size.first * axis_aligned_center_x;
  const float center_y = projected_center_x * std::sin(rotation) + projected_center_y * std::cos(rotation) + image_size.second * axis_aligned_center_y;
  const float width = (max_x - min_x) / image_size.first;
  const float height = (max_y - min_y) / image_size.second;

  rect->set_x_center(center_x / image_size.first);
  rect->set_y_center(center_y / image_size.second);
  rect->set_width(width);
  rect->set_height(height);
  rect->set_rotation(rotation);

  return absl::OkStatus();
}

} // namespace mediapipe_v01013_based
