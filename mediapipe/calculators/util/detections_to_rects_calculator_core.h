// Copyright 2025 The MediaPipe Authors.
//
// Core logic extracted from DetectionsToRectsCalculator for modularity.
#ifndef MEDIAPIPE_CALCULATORS_UTIL_DETECTIONS_TO_RECTS_CALCULATOR_CORE_H_
#define MEDIAPIPE_CALCULATORS_UTIL_DETECTIONS_TO_RECTS_CALCULATOR_CORE_H_

#include "mediapipe/calculators/util/detections_to_rects_calculator.pb.h"
#include "mediapipe/framework/formats/detection.pb.h"
#include <vector>
#include <utility>
#include "absl/types/optional.h"

namespace mediapipe_v01013_based {

class Rect;
class NormalizedRect;

struct DetectionsToRectsCoreConfig {
  int start_keypoint_index;
  int end_keypoint_index;
  float target_angle;
  bool rotate;
  bool output_zero_rect_for_empty_detections;
};

DetectionsToRectsCoreConfig SetDetectionsToRectsConfig(const DetectionsToRectsCalculatorOptions& options);

void ComputeRectsFromDetections(
    const std::vector<Detection>& detections,
    const DetectionsToRectsCoreConfig& config,
    const absl::optional<std::pair<int, int>>& image_size,
    std::vector<NormalizedRect>* norm_rects,
    std::vector<Rect>* rects);

}  // namespace mediapipe_v01013_based

#endif  // MEDIAPIPE_CALCULATORS_UTIL_DETECTIONS_TO_RECTS_CALCULATOR_CORE_H_
