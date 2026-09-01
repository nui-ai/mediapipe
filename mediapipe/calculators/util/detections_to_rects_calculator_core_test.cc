#include "mediapipe/calculators/util/detections_to_rects_calculator_core.h"

#include <string>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/formats/location_data.pb.h"
#include "mediapipe/framework/formats/rect.pb.h"

namespace hand_tracking_mp_lean {
namespace {

TEST(DetectionsToOrientedRectsTest,
     ReportsRequiredAndAvailableRotationKeypoints) {
  DetectionsToOrientedRects converter(/*target_angle_radians=*/0.0f);
  Detection detection;
  LocationData* location_data = detection.mutable_location_data();
  location_data->set_format(LocationData::RELATIVE_BOUNDING_BOX);
  location_data->add_relative_keypoints();
  location_data->add_relative_keypoints();

  std::vector<NormalizedRect> normalized_rects;
  std::vector<Rect> rects;
  const absl::Status status = converter.OrientedRectsFromDetections(
      {detection}, std::make_pair(640, 480), &normalized_rects, &rects);

  ASSERT_FALSE(status.ok());
  EXPECT_NE(status.message().find(
                "Detection rotation requires end keypoint index 2, but the "
                "detection has 2 relative keypoints."),
            std::string::npos)
      << status;
}

}  // namespace
}  // namespace hand_tracking_mp_lean
