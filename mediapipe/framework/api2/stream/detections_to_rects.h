#ifndef MEDIAPIPE_FRAMEWORK_API2_STREAM_DETECTIONS_TO_RECTS_H_
#define MEDIAPIPE_FRAMEWORK_API2_STREAM_DETECTIONS_TO_RECTS_H_

#include <utility>
#include <vector>

#include "mediapipe/framework/api2/builder.h"
#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/formats/rect.pb.h"

namespace hand_tracking_mp_lean::api2::builder {

// Updates @graph to convert @detection into a `NormalizedRect` according to
// passed parameters.
Stream<hand_tracking_mp_lean::NormalizedRect> ConvertAlignmentPointsDetectionToRect(
    Stream<hand_tracking_mp_lean::Detection> detection,
    Stream<std::pair<int, int>> image_size, int start_keypoint_index,
    int end_keypoint_index, float target_angle,
    hand_tracking_mp_lean::api2::builder::Graph& graph);

// Updates @graph to convert first detection from @detections into a
// `NormalizedRect` according to passed parameters.
Stream<hand_tracking_mp_lean::NormalizedRect> ConvertAlignmentPointsDetectionsToRect(
    Stream<std::vector<hand_tracking_mp_lean::Detection>> detections,
    Stream<std::pair<int, int>> image_size, int start_keypoint_index,
    int end_keypoint_index, float target_angle,
    hand_tracking_mp_lean::api2::builder::Graph& graph);

// Updates @graph to convert @detection into a `NormalizedRect` according to
// passed parameters.
Stream<hand_tracking_mp_lean::NormalizedRect> ConvertDetectionToRect(
    Stream<hand_tracking_mp_lean::Detection> detections,
    Stream<std::pair<int, int>> image_size, int start_keypoint_index,
    int end_keypoint_index, float target_angle,
    hand_tracking_mp_lean::api2::builder::Graph& graph);

// Updates @graph to convert @detections into a stream holding vector of
// `NormalizedRect` according to passed parameters.
Stream<std::vector<hand_tracking_mp_lean::NormalizedRect>> ConvertDetectionsToRects(
    Stream<std::vector<hand_tracking_mp_lean::Detection>> detections,
    Stream<std::pair<int, int>> image_size, int start_keypoint_index,
    int end_keypoint_index, float target_angle,
    hand_tracking_mp_lean::api2::builder::Graph& graph);

// Updates @graph to convert @detections into a stream holding vector of
// `NormalizedRect` according to passed parameters and using keypoints.
Stream<hand_tracking_mp_lean::NormalizedRect> ConvertDetectionsToRectUsingKeypoints(
    Stream<std::vector<hand_tracking_mp_lean::Detection>> detections,
    Stream<std::pair<int, int>> image_size, int start_keypoint_index,
    int end_keypoint_index, float target_angle,
    hand_tracking_mp_lean::api2::builder::Graph& graph);

}  // namespace hand_tracking_mp_lean::api2::builder

#endif  // MEDIAPIPE_FRAMEWORK_API2_STREAM_DETECTIONS_TO_RECTS_H_
