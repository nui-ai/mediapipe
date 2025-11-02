#include "mediapipe/framework/api2/stream/landmarks_to_detection.h"

#include "mediapipe/framework/api2/builder.h"
#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/formats/landmark.pb.h"

namespace mediapipe_v01013_based::api2::builder {

Stream<mediapipe_v01013_based::Detection> ConvertLandmarksToDetection(
    Stream<mediapipe_v01013_based::NormalizedLandmarkList> landmarks, Graph& graph) {
  auto& landmarks_to_detection =
      graph.AddNode("LandmarksToDetectionCalculator");
  landmarks.ConnectTo(landmarks_to_detection.In("NORM_LANDMARKS"));
  return landmarks_to_detection.Out("DETECTION").Cast<mediapipe_v01013_based::Detection>();
}

}  // namespace mediapipe_v01013_based::api2::builder
