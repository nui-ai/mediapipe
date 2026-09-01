#include "mediapipe/examples/desktop/face_tracking_c_api.h"

#include <cstdint>
#include <vector>

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

}  // namespace
