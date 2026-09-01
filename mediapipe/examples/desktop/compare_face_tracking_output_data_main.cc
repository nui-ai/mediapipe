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

#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "mediapipe/examples/desktop/face_tracking_output.pb.h"

#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/util/delimited_message_util.h>

ABSL_FLAG(std::string, reference_data_path, "",
          "Delimited FaceTrackingOutputData produced by the legacy graph.");
ABSL_FLAG(std::string, candidate_data_path, "",
          "Delimited FaceTrackingOutputData produced by FaceTrackingCore.");
ABSL_FLAG(double, absolute_tolerance, 1e-5,
          "Maximum permitted absolute difference for any X/Y/Z coordinate.");

namespace hand_tracking_mp_lean {
namespace {

bool ReadFrames(const std::string& path,
                std::vector<FaceTrackingOutputData>* frames) {
  std::ifstream input(path, std::ios::binary);
  if (!input.is_open()) {
    std::cerr << "Failed to open " << path << "\n";
    return false;
  }
  google::protobuf::io::IstreamInputStream stream(&input);
  while (true) {
    FaceTrackingOutputData frame;
    bool clean_eof = false;
    if (!google::protobuf::util::ParseDelimitedFromZeroCopyStream(
            &frame, &stream, &clean_eof)) {
      if (!clean_eof) {
        std::cerr << "Malformed delimited protobuf in " << path << "\n";
        return false;
      }
      break;
    }
    frames->push_back(std::move(frame));
  }
  return true;
}

int Compare() {
  std::vector<FaceTrackingOutputData> reference;
  std::vector<FaceTrackingOutputData> candidate;
  if (!ReadFrames(absl::GetFlag(FLAGS_reference_data_path), &reference) ||
      !ReadFrames(absl::GetFlag(FLAGS_candidate_data_path), &candidate)) {
    return EXIT_FAILURE;
  }
  if (reference.size() != candidate.size()) {
    std::cerr << "Frame count differs: reference=" << reference.size()
              << ", candidate=" << candidate.size() << "\n";
    return EXIT_FAILURE;
  }

  double maximum_difference = 0.0;
  double difference_sum = 0.0;
  std::size_t coordinate_count = 0;
  int maximum_frame = -1;
  int maximum_face = -1;
  int maximum_landmark = -1;
  char maximum_axis = '?';

  for (std::size_t frame_index = 0; frame_index < reference.size();
       ++frame_index) {
    const auto& expected_frame = reference[frame_index];
    const auto& actual_frame = candidate[frame_index];
    if (expected_frame.frame_number() != actual_frame.frame_number()) {
      std::cerr << "Frame number differs at record " << frame_index << "\n";
      return EXIT_FAILURE;
    }
    if (expected_frame.multi_face_landmarks_size() !=
        actual_frame.multi_face_landmarks_size()) {
      std::cerr << "Face count differs at frame " << frame_index
                << ": reference="
                << expected_frame.multi_face_landmarks_size()
                << ", candidate="
                << actual_frame.multi_face_landmarks_size() << "\n";
      return EXIT_FAILURE;
    }

    for (int face_index = 0;
         face_index < expected_frame.multi_face_landmarks_size();
         ++face_index) {
      const auto& expected =
          expected_frame.multi_face_landmarks(face_index);
      const auto& actual = actual_frame.multi_face_landmarks(face_index);
      if (expected.landmark_size() != actual.landmark_size()) {
        std::cerr << "Landmark count differs at frame " << frame_index
                  << ", face " << face_index << ": reference="
                  << expected.landmark_size() << ", candidate="
                  << actual.landmark_size() << "\n";
        return EXIT_FAILURE;
      }

      for (int landmark_index = 0;
           landmark_index < expected.landmark_size(); ++landmark_index) {
        const auto& expected_landmark = expected.landmark(landmark_index);
        const auto& actual_landmark = actual.landmark(landmark_index);
        const double differences[3] = {
            std::abs(static_cast<double>(expected_landmark.x()) -
                     actual_landmark.x()),
            std::abs(static_cast<double>(expected_landmark.y()) -
                     actual_landmark.y()),
            std::abs(static_cast<double>(expected_landmark.z()) -
                     actual_landmark.z()),
        };
        for (int axis = 0; axis < 3; ++axis) {
          difference_sum += differences[axis];
          ++coordinate_count;
          if (differences[axis] > maximum_difference) {
            maximum_difference = differences[axis];
            maximum_frame = static_cast<int>(frame_index);
            maximum_face = face_index;
            maximum_landmark = landmark_index;
            maximum_axis = "xyz"[axis];
          }
        }
      }
    }
  }

  const double mean_difference =
      coordinate_count == 0 ? 0.0 : difference_sum / coordinate_count;
  std::cout << std::setprecision(10)
            << "Compared " << reference.size() << " frames and "
            << coordinate_count << " landmark coordinates\n"
            << "Maximum absolute difference: " << maximum_difference;
  if (maximum_frame >= 0) {
    std::cout << " at frame " << maximum_frame << ", face " << maximum_face
              << ", landmark " << maximum_landmark << ", axis "
              << maximum_axis;
  }
  std::cout << "\nMean absolute difference: " << mean_difference << "\n";

  if (maximum_difference > absl::GetFlag(FLAGS_absolute_tolerance)) {
    std::cerr << "Maximum difference exceeds --absolute_tolerance\n";
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}

}  // namespace
}  // namespace hand_tracking_mp_lean

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  return hand_tracking_mp_lean::Compare();
}
