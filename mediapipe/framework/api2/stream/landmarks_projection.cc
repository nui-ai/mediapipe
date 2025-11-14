#include "mediapipe/framework/api2/stream/landmarks_projection.h"

#include <array>

#include "mediapipe/framework/api2/builder.h"
#include "mediapipe/framework/formats/landmark.pb.h"

namespace hand_tracking_mp_lean::api2::builder {

Stream<hand_tracking_mp_lean::NormalizedLandmarkList> ProjectLandmarks(
    Stream<hand_tracking_mp_lean::NormalizedLandmarkList> landmarks,
    Stream<std::array<float, 16>> projection_matrix, Graph& graph) {
  auto& projector = graph.AddNode("LandmarkProjectionCalculator");
  landmarks.ConnectTo(projector.In("NORM_LANDMARKS"));
  projection_matrix.ConnectTo(projector.In("PROJECTION_MATRIX"));
  return projector.Out("NORM_LANDMARKS")
      .Cast<hand_tracking_mp_lean::NormalizedLandmarkList>();
}

}  // namespace hand_tracking_mp_lean::api2::builder
