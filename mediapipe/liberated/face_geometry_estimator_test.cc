#include "mediapipe/liberated/face_geometry_estimator.h"

#include "gtest/gtest.h"

namespace hand_tracking_mp_lean {
namespace {

// Builds the degenerate landmark set whose documented result is an absent pose.
NormalizedLandmarkList CompactLandmarks() {
  NormalizedLandmarkList landmarks;
  for (int index = 0; index < 468; ++index) {
    auto* landmark = landmarks.add_landmark();
    landmark->set_x(0.5f);
    landmark->set_y(0.5f);
  }
  return landmarks;
}

TEST(FaceGeometryEstimatorTest, CompactFaceHasNoPose) {
  auto estimator = FaceGeometryEstimator::Create(63.0f, nullptr);
  ASSERT_TRUE(estimator.ok()) << estimator.status();

  auto pose = estimator.value()->Estimate(CompactLandmarks(), 1280, 720);
  ASSERT_TRUE(pose.ok()) << pose.status();
  EXPECT_FALSE(pose.value().has_value());
}

TEST(FaceGeometryEstimatorTest, PropagatesPipelineInputError) {
  auto estimator = FaceGeometryEstimator::Create(63.0f, nullptr);
  ASSERT_TRUE(estimator.ok()) << estimator.status();

  EXPECT_FALSE(estimator.value()
                   ->Estimate(CompactLandmarks(), 0, 720)
                   .ok());
}

}  // namespace
}  // namespace hand_tracking_mp_lean
