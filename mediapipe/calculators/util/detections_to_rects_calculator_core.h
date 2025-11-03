// Copyright 2025 The MediaPipe Authors.
//
// Core logic extracted from DetectionsToRectsCalculator for modularity.
#ifndef MEDIAPIPE_CALCULATORS_UTIL_DETECTIONS_TO_RECTS_CALCULATOR_CORE_H_
#define MEDIAPIPE_CALCULATORS_UTIL_DETECTIONS_TO_RECTS_CALCULATOR_CORE_H_

#include "mediapipe/calculators/util/detections_to_rects_calculator.pb.h"
#include "mediapipe/framework/formats/detection.pb.h"
#include "absl/status/status.h"
#include <vector>
#include <utility>

#include "absl/types/optional.h"

namespace mediapipe_v01013_based {

class Rect;
class NormalizedRect;

// Internal config used by the core class. Not exposed to callers.
struct DetectionsToRectsCoreConfig {
  int start_keypoint_index;
  int end_keypoint_index;
  float target_angle;  // Radians.
  bool rotate;
  bool output_zero_rect_for_empty_detections;
};

class DetectionsToRectsCore {
 public:

  explicit DetectionsToRectsCore(
    float target_angle_radians,
    bool output_zero_rect_for_empty_detections=false);

  bool NeedsImageSize() const;

  bool OutputZeroForEmptyDetections() const;

  absl::Status ComputeRectsFromDetections(
      const std::vector<Detection>& detections,
      const absl::optional<std::pair<int, int>>& image_size,
      std::vector<NormalizedRect>* norm_rects,
      std::vector<Rect>* rects) const;

 private:
  DetectionsToRectsCalculatorOptions options_;
  DetectionsToRectsCoreConfig config_{};
};

}

#endif  // MEDIAPIPE_CALCULATORS_UTIL_DETECTIONS_TO_RECTS_CALCULATOR_CORE_H_
