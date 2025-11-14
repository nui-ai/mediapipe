#ifndef MEDIAPIPE_FRAMEWORK_API2_STREAM_LANDMARKS_TO_TENSOR_H_
#define MEDIAPIPE_FRAMEWORK_API2_STREAM_LANDMARKS_TO_TENSOR_H_

#include <utility>
#include <vector>

#include "absl/types/span.h"
#include "mediapipe/calculators/tensor/landmarks_to_tensor_calculator.pb.h"
#include "mediapipe/framework/api2/builder.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/tensor.h"

namespace hand_tracking_mp_lean::api2::builder {

// Updates @graph to convert @landmarks to a Tensor. Values and their order are
// defined by @attributes. If @flatten is true resulting tensor will be 1D,
// otherwise tensor will be 2D with (n_landmarks, n_attributes) shape.
Stream<std::vector<Tensor>> ConvertLandmarksToTensor(
    Stream<hand_tracking_mp_lean::LandmarkList> landmarks,
    absl::Span<const hand_tracking_mp_lean::LandmarksToTensorCalculatorOptions::Attribute>
        attributes,
    bool flatten, Graph& graph);

// Updates @graph to convert @normalized_landmarks to a Tensor. Values and their
// order are defined by @attributes. X, Y and Z values are scaled using
// @image_size. If @flatten is true resulting tensor will be 1D, otherwise
// tensor will be 2D with (n_landmarks, n_attributes) shape.
Stream<std::vector<Tensor>> ConvertNormalizedLandmarksToTensor(
    Stream<hand_tracking_mp_lean::NormalizedLandmarkList> normalized_landmarks,
    Stream<std::pair<int, int>> image_size,
    absl::Span<const hand_tracking_mp_lean::LandmarksToTensorCalculatorOptions::Attribute>
        attributes,
    bool flatten, Graph& graph);

}  // namespace hand_tracking_mp_lean::api2::builder

#endif  // MEDIAPIPE_FRAMEWORK_API2_STREAM_LANDMARKS_TO_TENSOR_H_
