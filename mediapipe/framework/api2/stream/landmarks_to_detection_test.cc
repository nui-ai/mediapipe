#include "mediapipe/framework/api2/stream/landmarks_to_detection.h"

#include "mediapipe/framework/api2/builder.h"
#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"

namespace mediapipe_v01013_based::api2::builder {
namespace {

TEST(LandmarksToDetection, VerifyConfig) {
  mediapipe_v01013_based::api2::builder::Graph graph;

  Stream<NormalizedLandmarkList> landmarks =
      graph.In("LANDMARKS").Cast<NormalizedLandmarkList>();
  Stream<Detection> detection = ConvertLandmarksToDetection(landmarks, graph);
  detection.SetName("detection");

  EXPECT_THAT(
      graph.GetConfig(),
      EqualsProto(mediapipe_v01013_based::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        node {
          calculator: "LandmarksToDetectionCalculator"
          input_stream: "NORM_LANDMARKS:__stream_0"
          output_stream: "DETECTION:detection"
        }
        input_stream: "LANDMARKS:__stream_0"
      )pb")));
}

}  // namespace
}  // namespace mediapipe_v01013_based::api2::builder
