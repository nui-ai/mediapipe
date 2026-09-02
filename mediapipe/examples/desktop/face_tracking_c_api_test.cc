#include "mediapipe/examples/desktop/face_tracking_c_api.h"

#include <cstdlib>
#include <cstdint>
#include <utility>
#include <vector>

#include "mediapipe/examples/desktop/face_tracking_c_conversion.h"
#include "gtest/gtest.h"

namespace {

TEST(FaceTrackingCApiTest, ProcessesAnRgbImageAndOwnsItsResult) {
  FaceTrackingOptionsC options = {
      /*max_faces=*/1,
      /*use_previous_landmarks=*/1,
      /*with_attention=*/1,
      /*min_detection_confidence=*/0.5f,
      /*min_tracking_confidence=*/0.5f,
      /*xnnpack_num_threads=*/1,
      /*estimate_pose=*/1,
      /*vertical_fov_degrees=*/63.0f,
  };
  FaceTrackingCoreOpaqueHandle tracker =
      face_tracking_core_create(&options, nullptr);
  ASSERT_NE(tracker, nullptr) << face_tracking_get_last_error();

  constexpr size_t kWidth = 128;
  constexpr size_t kHeight = 128;
  std::vector<std::uint8_t> pixels(kWidth * kHeight * 3, 0);
  FaceTrackingResultC* result = nullptr;
  ASSERT_EQ(face_tracking_core_process(tracker, pixels.data(), kWidth,
                                       kHeight, kWidth * 3, &result),
            0)
      << face_tracking_get_last_error();
  ASSERT_NE(result, nullptr);
  EXPECT_EQ(result->face_detector_ran, 1);
  for (size_t index = 0; index < result->face_count; ++index) {
    EXPECT_EQ(result->faces[index].landmark_count, 478);
    EXPECT_NE(result->faces[index].landmarks, nullptr);
  }

  face_tracking_result_destroy(result);
  EXPECT_EQ(face_tracking_core_reset(tracker), 0)
      << face_tracking_get_last_error();
  EXPECT_EQ(face_tracking_core_finalize(tracker), 0)
      << face_tracking_get_last_error();
}

TEST(FaceTrackingCApiTest, RejectsInvalidHandlesAndImages) {
  FaceTrackingResultC* result = nullptr;
  std::uint8_t pixel[3] = {};
  EXPECT_NE(face_tracking_core_process(nullptr, pixel, 1, 1, 3, &result), 0);
  EXPECT_EQ(result, nullptr);

  FaceTrackingCoreOpaqueHandle tracker =
      face_tracking_core_create(nullptr, nullptr);
  ASSERT_NE(tracker, nullptr) << face_tracking_get_last_error();
  EXPECT_NE(face_tracking_core_process(tracker, nullptr, 1, 1, 3, &result),
            0);
  EXPECT_EQ(result, nullptr);
  EXPECT_EQ(face_tracking_core_finalize(tracker), 0);
}

TEST(FaceTrackingCApiTest, CopiesOptionalPoseTransformIntoCResult) {
  hand_tracking_mp_lean::ImageFaceTrackingResult cpp_result;
  hand_tracking_mp_lean::FaceInference face;
  for (int index = 0; index < 468; ++index) {
    auto* landmark = face.landmarks.add_landmark();
    landmark->set_x(static_cast<float>(index));
  }
  hand_tracking_mp_lean::FacePoseTransform pose_transform;
  for (size_t index = 0;
       index < pose_transform.canonical_to_runtime_column_major.size();
       ++index) {
    pose_transform.canonical_to_runtime_column_major[index] =
        static_cast<float>(index) + 0.25f;
  }
  face.pose_transform = pose_transform;
  cpp_result.faces.push_back(std::move(face));

  auto* c_result = static_cast<FaceTrackingResultC*>(
      std::calloc(1, sizeof(FaceTrackingResultC)));
  ASSERT_NE(c_result, nullptr);
  ASSERT_EQ(ConvertFaceTrackingResultToC(cpp_result, c_result, nullptr), 0);
  ASSERT_EQ(c_result->face_count, 1);
  ASSERT_EQ(c_result->faces[0].has_pose_transform, 1);
  for (size_t index = 0; index < 16; ++index) {
    EXPECT_EQ(c_result->faces[0]
                  .pose_transform.canonical_to_runtime_column_major[index],
              static_cast<float>(index) + 0.25f);
  }
  face_tracking_result_destroy(c_result);
}

TEST(FaceTrackingCApiTest, ReportsBreakingAbiVersion) {
  EXPECT_STREQ(face_tracking_core_version(), "2.0.0");
}

}  // namespace
