#ifndef MEDIAPIPE_FRAMEWORK_API2_STREAM_SMOOTHING_H_
#define MEDIAPIPE_FRAMEWORK_API2_STREAM_SMOOTHING_H_

#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "absl/types/optional.h"
#include "mediapipe/calculators/util/landmarks_smoothing_calculator.pb.h"
#include "mediapipe/framework/api2/builder.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/rect.pb.h"

namespace hand_tracking_mp_lean::api2::builder {

struct OneEuroFilterConfig {
  float min_cutoff;
  float beta;
  float derivate_cutoff;
};

// Updates graph to smooth normalized landmarks and returns resulting stream.
//
// @landmarks - normalized landmarks.
// @image_size - size of image where landmarks were detected.
// @scale_roi - can be used to specify object scale.
// @config - filter config.
// @graph - graph to update.
//
// Returns: smoothed/filtered normalized landmarks.
//
// NOTE: one-euro filter is exposed only. Other filter options can be exposed
//   on demand.
Stream<hand_tracking_mp_lean::NormalizedLandmarkList> SmoothLandmarks(
    Stream<hand_tracking_mp_lean::NormalizedLandmarkList> landmarks,
    Stream<std::pair<int, int>> image_size,
    std::optional<Stream<NormalizedRect>> scale_roi,
    const OneEuroFilterConfig& config, Graph& graph);

// Updates graph to smooth absolute landmarks and returns resulting stream.
//
// @landmarks - absolute landmarks.
// @scale_roi - can be used to specify object scale.
// @config - filter config.
// @graph - graph to update.
//
// Returns: smoothed/filtered absolute landmarks.
//
// NOTE: one-euro filter is exposed only. Other filter options can be exposed
//   on demand.
Stream<hand_tracking_mp_lean::LandmarkList> SmoothLandmarks(
    Stream<hand_tracking_mp_lean::LandmarkList> landmarks,
    std::optional<Stream<NormalizedRect>> scale_roi,
    const OneEuroFilterConfig& config, Graph& graph);

// Updates graph to smooth normalized landmarks and returns resulting stream.
//
// @landmarks - normalized landmarks vector.
// @tracking_ids - tracking IDs associated with landmarks
// @image_size - size of image where landmarks were detected.
// @scale_roi - can be used to specify object scales.
// @config - filter config.
// @graph - graph to update.
//
// Returns: smoothed/filtered normalized landmarks.
//
// NOTE: one-euro filter is exposed only. Other filter options can be exposed
//   on demand.
Stream<std::vector<hand_tracking_mp_lean::NormalizedLandmarkList>> SmoothMultiLandmarks(
    Stream<std::vector<hand_tracking_mp_lean::NormalizedLandmarkList>> landmarks,
    Stream<std::vector<int64_t>> tracking_ids,
    Stream<std::pair<int, int>> image_size,
    std::optional<Stream<std::vector<NormalizedRect>>> scale_roi,
    const hand_tracking_mp_lean::LandmarksSmoothingCalculatorOptions& config, Graph& graph);

// Updates graph to smooth absolute landmarks and returns resulting stream.
//
// @landmarks - absolute landmarks vector.
// @tracking_ids - tracking IDs associated with landmarks
// @scale_roi - can be used to specify object scales.
// @config - filter config.
// @graph - graph to update.
//
// Returns: smoothed/filtered absolute landmarks.
//
// NOTE: one-euro filter is exposed only. Other filter options can be exposed
//   on demand.
Stream<std::vector<hand_tracking_mp_lean::LandmarkList>> SmoothMultiWorldLandmarks(
    Stream<std::vector<hand_tracking_mp_lean::LandmarkList>> landmarks,
    Stream<std::vector<int64_t>> tracking_ids,
    std::optional<Stream<std::vector<hand_tracking_mp_lean::Rect>>> scale_roi,
    const hand_tracking_mp_lean::LandmarksSmoothingCalculatorOptions& config, Graph& graph);

// Updates graph to smooth visibility of landmarks.
//
// @landmarks - normalized landmarks.
// @low_pass_filter_alpha - low pass filter alpha to use for smoothing.
// @graph - graph to update.
//
// Returns: normalized landmarks containing smoothed visibility.
Stream<hand_tracking_mp_lean::NormalizedLandmarkList> SmoothLandmarksVisibility(
    Stream<hand_tracking_mp_lean::NormalizedLandmarkList> landmarks,
    float low_pass_filter_alpha, Graph& graph);

// Updates graph to smooth visibility of landmarks.
//
// @landmarks - absolute landmarks.
// @low_pass_filter_alpha - low pass filter alpha to use for smoothing.
// @graph - graph to update.
//
// Returns: absolute landmarks containing smoothed visibility.
Stream<hand_tracking_mp_lean::LandmarkList> SmoothLandmarksVisibility(
    Stream<hand_tracking_mp_lean::LandmarkList> landmarks, float low_pass_filter_alpha,
    hand_tracking_mp_lean::api2::builder::Graph& graph);

}  // namespace hand_tracking_mp_lean::api2::builder

#endif  // MEDIAPIPE_FRAMEWORK_API2_STREAM_SMOOTHING_H_
