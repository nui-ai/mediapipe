// Copyright 2025 The MediaPipe Authors.
// Core logic extracted from DetectionsToRectsCalculator for modularity.

#include "mediapipe/calculators/util/detections_to_rects_calculator_core.h"
#include <cmath>
#include <limits>
#include "mediapipe/framework/port/ret_check.h"
#include "absl/types/optional.h"
#include <cassert>
#include "mediapipe/framework/formats/rect.pb.h"

namespace mediapipe_v01013_based {

static float NormalizeRadians(float angle) {
  return angle - 2 * M_PI * std::floor((angle - (-M_PI)) / (2 * M_PI));
}

/// helper function
static absl::Status ComputeRotation(const Detection& detection, const DetectionsToRectsCoreConfig& config, const absl::optional<std::pair<int, int>>& image_size, float* rotation) {
  const auto& location_data = detection.location_data();
  RET_CHECK(image_size) << "Image size is required to calculate rotation";
  const float x0 = location_data.relative_keypoints(config.start_keypoint_index).x() * image_size->first;
  const float y0 = location_data.relative_keypoints(config.start_keypoint_index).y() * image_size->second;
  const float x1 = location_data.relative_keypoints(config.end_keypoint_index).x() * image_size->first;
  const float y1 = location_data.relative_keypoints(config.end_keypoint_index).y() * image_size->second;
  *rotation = NormalizeRadians(config.target_angle - std::atan2(-(y1 - y0), x1 - x0));
  return absl::OkStatus();
}

/// helper function
static absl::Status DetectionToRect(const Detection& detection, Rect* rect) {
  const auto& location_data = detection.location_data();
  RET_CHECK(location_data.format() == LocationData::RELATIVE_BOUNDING_BOX);
  rect->set_x_center(location_data.relative_bounding_box().xmin() + location_data.relative_bounding_box().width() / 2.0f);
  rect->set_y_center(location_data.relative_bounding_box().ymin() + location_data.relative_bounding_box().height() / 2.0f);
  rect->set_width(location_data.relative_bounding_box().width());
  rect->set_height(location_data.relative_bounding_box().height());
  return absl::OkStatus();
}

/// helper function
static absl::Status DetectionToNormalizedRect(const Detection& detection, NormalizedRect* rect) {
  const auto& location_data = detection.location_data();
  RET_CHECK(location_data.format() == LocationData::RELATIVE_BOUNDING_BOX);
  rect->set_x_center(location_data.relative_bounding_box().xmin() + location_data.relative_bounding_box().width() / 2.0f);
  rect->set_y_center(location_data.relative_bounding_box().ymin() + location_data.relative_bounding_box().height() / 2.0f);
  rect->set_width(location_data.relative_bounding_box().width());
  rect->set_height(location_data.relative_bounding_box().height());
  return absl::OkStatus();
}


/// from a raw axes parallel detection rect of the SSD model, orients a rect based on keypoints of the palm detection!
DetectionsToOrientedRects::DetectionsToOrientedRects(float target_angle_radians, bool output_zero_rect_for_empty_detections) {

  const int start_keypoint_index = 0;  // Center of wrist.
  const int end_keypoint_index = 2;    // MCP of middle finger.

  // Initialize options_ similarly to the previous code path in Open().
  options_ = DetectionsToRectsCalculatorOptions();
  options_.set_rotation_vector_start_keypoint_index(start_keypoint_index);
  options_.set_rotation_vector_end_keypoint_index(end_keypoint_index);
  options_.set_rotation_vector_target_angle(target_angle_radians);
  options_.set_output_zero_rect_for_empty_detections(output_zero_rect_for_empty_detections);

  // Build internal config from options_.
  config_ = DetectionsToRectsCoreConfig();
  config_.rotate = false;
  config_.target_angle = 0.0f;
  config_.start_keypoint_index = 0;
  config_.end_keypoint_index = 0;
  if (options_.has_rotation_vector_start_keypoint_index()) {
    assert(options_.has_rotation_vector_end_keypoint_index() && "End keypoint index required if start is set.");
    assert((options_.has_rotation_vector_target_angle() ^ options_.has_rotation_vector_target_angle_degrees()) && "Exactly one target angle option must be set.");
    if (options_.has_rotation_vector_target_angle()) {
      config_.target_angle = options_.rotation_vector_target_angle();
    } else {
      config_.target_angle = M_PI * options_.rotation_vector_target_angle_degrees() / 180.f;
    }
    config_.start_keypoint_index = options_.rotation_vector_start_keypoint_index();
    config_.end_keypoint_index = options_.rotation_vector_end_keypoint_index();
    config_.rotate = true;
  }
  config_.output_zero_rect_for_empty_detections = options_.output_zero_rect_for_empty_detections();
}

bool DetectionsToOrientedRects::NeedsImageSize() const {
  return config_.rotate;
}

bool DetectionsToOrientedRects::OutputZeroForEmptyDetections() const {
  return config_.output_zero_rect_for_empty_detections;
}

absl::Status DetectionsToOrientedRects::OrientedRectsFromDetections(
    const std::vector<Detection>& detections,
    const absl::optional<std::pair<int, int>>& image_size,
    std::vector<NormalizedRect>* norm_rects,
    std::vector<Rect>* rects) const {
  rects->clear();
  norm_rects->clear();
  for (const auto& detection : detections) {
    Rect rect;
    MP_RETURN_IF_ERROR(DetectionToRect(detection, &rect));
    if (config_.rotate) {
      float rotation;
      MP_RETURN_IF_ERROR(ComputeRotation(detection, config_, image_size, &rotation));
      rect.set_rotation(rotation);
    }
    rects->push_back(rect);
    NormalizedRect norm_rect;
    MP_RETURN_IF_ERROR(DetectionToNormalizedRect(detection, &norm_rect));
    if (config_.rotate) {
      float rotation;
      MP_RETURN_IF_ERROR(ComputeRotation(detection, config_, image_size, &rotation));
      norm_rect.set_rotation(rotation);
    }
    norm_rects->push_back(norm_rect);
  }
  return absl::OkStatus();
}

}  // namespace mediapipe_v01013_based
