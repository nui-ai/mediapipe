#include "mediapipe/framework/api2/stream/landmarks_projection.h"

#include <array>

#include "mediapipe/framework/api2/builder.h"
#include "mediapipe/framework/formats/landmark.pb.h"

namespace mediapipe_v01013_based::api2::builder {

Stream<mediapipe_v01013_based::NormalizedLandmarkList> ProjectLandmarks(
    Stream<mediapipe_v01013_based::NormalizedLandmarkList> landmarks,
    Stream<std::array<float, 16>> projection_matrix, Graph& graph) {
  auto& projector = graph.AddNode("LandmarkProjectionCalculator");
  landmarks.ConnectTo(projector.In("NORM_LANDMARKS"));
  projection_matrix.ConnectTo(projector.In("PROJECTION_MATRIX"));
  return projector.Out("NORM_LANDMARKS")
      .Cast<mediapipe_v01013_based::NormalizedLandmarkList>();
}

}  // namespace mediapipe_v01013_based::api2::builder
