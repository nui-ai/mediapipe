// Copyright 2026 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <cstdint>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <memory>
#include <string>
#include <utility>

#include <opencv2/videoio.hpp>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "mediapipe/examples/desktop/face_tracking_output.pb.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status_macros.h"
#include "mediapipe/liberated/face_tracking.h"

#include <google/protobuf/util/delimited_message_util.h>

ABSL_FLAG(std::string, input_video_path, "",
          "Input video. An empty value uses camera 0.");
ABSL_FLAG(std::string, output_data_path, "face_tracking_output.pb",
          "Delimited FaceTrackingOutputData output. An empty value disables "
          "output persistence.");
ABSL_FLAG(std::uint32_t, max_num_faces, 1,
          "Maximum number of faces to detect and track.");
ABSL_FLAG(bool, static_image_mode, false,
          "Run face detection on every frame instead of reusing landmark ROIs.");
ABSL_FLAG(bool, refine_landmarks, false,
          "Use the attention model and return 478 landmarks, including irises, "
          "instead of the base model's 468 landmarks.");
ABSL_FLAG(double, min_detection_confidence, 0.5,
          "Minimum face detector confidence in the open interval (0, 1).");
ABSL_FLAG(double, min_tracking_confidence, 0.5,
          "Minimum landmark model face-presence confidence in the open "
          "interval (0, 1).");
ABSL_FLAG(int, xnnpack_num_threads, 1,
          "Number of XNNPACK delegate threads per model interpreter.");
ABSL_FLAG(std::string, models_root, "",
          "Optional directory prepended to the repository-relative model "
          "paths. Leave empty when running through Bazel.");

namespace hand_tracking_mp_lean {
namespace {

std::shared_ptr<const Image> MakeImage(cv::Mat rgb_image) {
  auto mat = std::make_shared<cv::Mat>(std::move(rgb_image));
  auto frame = std::make_shared<ImageFrame>(
      ImageFormat::SRGB, mat->cols, mat->rows, mat->step[0], mat->data,
      [mat](std::uint8_t*) {
        // Holding mat in this deleter keeps its external pixels alive for the
        // lifetime of the zero-copy ImageFrame.
      });
  return std::make_shared<const Image>(std::move(frame));
}

absl::Status Run(FaceTrackingCore* tracker) {
  cv::VideoCapture capture;
  const std::string input_path = absl::GetFlag(FLAGS_input_video_path);
  const bool reading_video = !input_path.empty();
  if (reading_video) {
    capture.open(input_path);
  } else {
    capture.open(0);
  }
  RET_CHECK(capture.isOpened()) << "Failed to open "
                                << (reading_video ? input_path : "camera 0");

  const std::string output_path = absl::GetFlag(FLAGS_output_data_path);
  std::ofstream output;
  if (!output_path.empty()) {
    output.open(output_path, std::ios::binary | std::ios::trunc);
    RET_CHECK(output.is_open()) << "Failed to open output file: "
                                << output_path;
  }

  int frame_number = 0;
  int frames_with_faces = 0;
  int detector_runs = 0;
  while (true) {
    cv::Mat bgr_image;
    capture >> bgr_image;
    if (bgr_image.empty()) {
      if (reading_video) {
        break;
      }
      ABSL_LOG(WARNING) << "Skipping empty camera frame " << frame_number;
      continue;
    }

    cv::Mat rgb_image;
    cv::cvtColor(bgr_image, rgb_image, cv::COLOR_BGR2RGB);
    if (!reading_video) {
      cv::flip(rgb_image, rgb_image, /*flipCode=*/1);
    }

    MP_ASSIGN_OR_RETURN(auto result, tracker->Process(MakeImage(rgb_image)));
    frames_with_faces += !result->face_landmarks.empty();
    detector_runs += result->face_detector_ran;
    if (frame_number == 0 || (frame_number + 1) % 30 == 0) {
      ABSL_LOG(INFO) << "Frame " << frame_number << ": "
                     << result->face_landmarks.size() << " face(s), "
                     << (result->face_landmarks.empty()
                             ? 0
                             : result->face_landmarks.front().landmark_size())
                     << " landmarks per first face";
    }

    if (output.is_open()) {
      FaceTrackingOutputData frame_output;
      frame_output.set_frame_number(frame_number);
      for (const auto& landmarks : result->face_landmarks) {
        *frame_output.add_multi_face_landmarks() = landmarks;
      }
      for (float score : result->face_presence_scores) {
        frame_output.add_face_presence_scores(score);
      }
      for (const auto& rect : result->face_rects_from_landmarks) {
        *frame_output.add_face_rects_from_landmarks() = rect;
      }
      RET_CHECK(google::protobuf::util::SerializeDelimitedToOstream(
          frame_output, &output))
          << "Failed to serialize frame " << frame_number;
    }
    ++frame_number;
  }

  ABSL_LOG(INFO) << "Processed " << frame_number << " frame(s); faces found "
                 << "in " << frames_with_faces << " frame(s); detector ran "
                 << detector_runs << " time(s)";
  if (output.is_open()) {
    output.close();
    RET_CHECK(output.good()) << "Failed while writing output file: "
                             << output_path;
    ABSL_LOG(INFO) << "Wrote " << output_path;
  }
  return absl::OkStatus();
}

}  // namespace
}  // namespace hand_tracking_mp_lean

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);

  hand_tracking_mp_lean::FaceTrackingOptions options;
  options.max_faces = absl::GetFlag(FLAGS_max_num_faces);
  options.use_previous_landmarks = !absl::GetFlag(FLAGS_static_image_mode);
  options.with_attention = absl::GetFlag(FLAGS_refine_landmarks);
  options.min_detection_confidence =
      static_cast<float>(absl::GetFlag(FLAGS_min_detection_confidence));
  options.min_tracking_confidence =
      static_cast<float>(absl::GetFlag(FLAGS_min_tracking_confidence));
  options.xnnpack_num_threads = absl::GetFlag(FLAGS_xnnpack_num_threads);

  try {
    const std::string models_root = absl::GetFlag(FLAGS_models_root);
    const std::string* models_root_ptr =
        models_root.empty() ? nullptr : &models_root;
    hand_tracking_mp_lean::FaceTrackingCore tracker(options, models_root_ptr);
    const absl::Status status = hand_tracking_mp_lean::Run(&tracker);
    if (!status.ok()) {
      ABSL_LOG(ERROR) << status;
      return EXIT_FAILURE;
    }
  } catch (const std::exception& error) {
    ABSL_LOG(ERROR) << error.what();
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
