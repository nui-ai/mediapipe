// Copyright 2019 The MediaPipe Authors.
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
//
// operates a MediaPipe pipeline for hand tracking over a given input video file,
// saving the per-frame output from the pipeline, and diffing the same outputs
// against a reference output data file in real-time as it goes frame by frame.

#include <cstdlib>
#include <fstream>
#include <filesystem>
#include <string>

#include "mediapipe/examples/desktop/pipeline_output.pb.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/memory_manager.h"
#include "mediapipe/liberated/liberated.h"
// #include "mediapipe/examples/desktop/hands_pipeline_operator.h"

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/opencv_video_inc.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/util/resource_util.h"
#include <google/protobuf/util/delimited_message_util.h>
#include <google/protobuf/util/message_differencer.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>

#include "mediapipe/framework/formats/image_opencv.h"

constexpr char kInputStream[] = "image";
constexpr char kOutputProtoFilename[] = "output_data_cpp.pb";

ABSL_FLAG(std::string, graph_file, "",
          "name of pipeline pbtxt file.");
ABSL_FLAG(std::string, input_video_path, "",
          "video to load. "
          "if not provided, attempts to use a webcam (not tested).");
ABSL_FLAG(std::string, reference_data_path, "",
          "reference data file");
ABSL_FLAG(std::string, output_video_path, "",
          "output video file path (.mp4 only). "
          "if not provided, shows the result images in a window without saving them to an output video file.");


bool get_project_root_dir(const std::string &filename, std::string &value1) {
  const char* root_dir = std::getenv("PROJECT_ROOT_DIR");
  if (root_dir && std::string(root_dir).length() > 0) {
    value1 = (std::filesystem::path(root_dir) / filename).string();
    return true;
  }
  return false;
}

// Returns the full path by prefixing with PROJECT_ROOT_DIR if set and path is not absolute.
std::string GetProjectRootedPath(const std::string& filename) {
  if (filename.empty()) {
    throw std::invalid_argument("no file name provided");
  }
  if (std::filesystem::path(filename).is_absolute()) {
    return filename;
  }
  std::string project_root_dir;
  if (get_project_root_dir(filename, project_root_dir)) {
    return project_root_dir;
  }

  throw std::runtime_error("a relative path cannot be resolved when the PROJECT_ROOT_DIR environment variable is not set");
}

// helper function to read the output reference data from file
bool ReadReferenceData(const std::string& filename, std::vector<mediapipe_v01013_based::PipelineOutputData>& out) {
    std::ifstream input(filename, std::ios::binary);
    if (!input.is_open()) {
        ABSL_LOG(ERROR) << "Failed to open reference file: " << filename;
        return false;
    }
    google::protobuf::io::IstreamInputStream zero_copy_input(&input);
    bool clean_eof = false;
    int msg_count = 0;
    while (true) {
        mediapipe_v01013_based::PipelineOutputData msg;
        std::streampos pos = input.tellg();
        if (!google::protobuf::util::ParseDelimitedFromZeroCopyStream(&msg, &zero_copy_input, &clean_eof)) {
            if (msg_count == 0) {
                ABSL_LOG(ERROR) << "Failed to parse any messages from " << filename
                               << " (parse error at file offset " << pos << ")";
                return false;
            } else {
                // EOF or trailing bytes after all messages parsed; treat as normal
                break;
            }
        }
        if (clean_eof) {
            // End of file reached after last message
            break;
        }
        out.push_back(msg);
        ++msg_count;
    }
    return true;
}

absl::Status RunPipelineWithDiffing() {
  // Load reference data if a path to a reference data file has been provided as a program argument
  bool diffing = false;
  std::vector<mediapipe_v01013_based::PipelineOutputData> reference_data;
  if (absl::GetFlag(FLAGS_reference_data_path).empty()) {
    ABSL_LOG(WARNING) << "no reference data file path provided";
  } else {
    auto reference_data_path = GetProjectRootedPath(absl::GetFlag(FLAGS_reference_data_path));
    if (!ReadReferenceData(reference_data_path, reference_data)) { return absl::AbortedError(std::string("failed to load reference data from the given path ") + GetProjectRootedPath(absl::GetFlag(FLAGS_reference_data_path))); }
    ABSL_LOG(INFO) << reference_data.size() << " records loaded from the reference data file " << reference_data_path;
    diffing = true;
  }

  // set of expected pipeline output streams
  const std::vector<std::string> graph_output_streams_names = {
    "multi_hand_landmarks",
    "multi_hand_world_landmarks",
    "multi_handedness",
    // "hand_rects_from_palm_detections"
  };

  auto memory_manager = mediapipe_v01013_based::MemoryManager();
  auto hand_tracking = mediapipe_v01013_based::HandTrackingCore(&memory_manager);

  // auto status_or_op = mediapipe_v01013_based::HandsPipelineOperator::Create(GetProjectRootedPath(absl::GetFlag(FLAGS_graph_file)), graph_output_streams_names);
  // if (!status_or_op.ok()) {
  //   std::cerr << "Failed to create HandsPipelineOperator: " << status_or_op.status().message() << std::endl;
  //   return status_or_op.status();
  // }
  // auto pipeline_operator = std::move(status_or_op.value());

  // initializing the camera or load the input video
  cv::VideoCapture capture;
  const bool video_file_input = !absl::GetFlag(FLAGS_input_video_path).empty();
  if (video_file_input) {
    capture.open(GetProjectRootedPath(absl::GetFlag(FLAGS_input_video_path)));
  } else {
    capture.open(0);
  }
  RET_CHECK(capture.isOpened());

  ABSL_LOG(INFO) << "starting a mediapipe graph";

  // Initialize output protobuf file (overwrite if exists)
  std::ofstream output_proto_file(GetProjectRootedPath(kOutputProtoFilename), std::ios::binary | std::ios::trunc);
  if (!output_proto_file.is_open()) {
    return absl::InternalError(std::string("failed to open the given output data path ") + GetProjectRootedPath(kOutputProtoFilename) + " for writing");
  }

  // process all input frames
  for (int i = 0; i < 999999; ++i) {

    // acquire an image
    cv::Mat input_frame_raw;
    capture >> input_frame_raw;
    if (input_frame_raw.empty()) {
      if (!video_file_input) {
        ABSL_LOG(WARNING) << "empty frame from camera for frame number " << i << " will not be processed";
        continue;
      }
      break; // end of video file (irrelevant when the input is from a camera)
    }
    ABSL_LOG(WARNING) << "processing frame number " << i;

    // convert to mediapipe image format
    cv::Mat input_frame;
    cv::cvtColor(input_frame_raw, input_frame, cv::COLOR_BGR2RGB);
    if (!video_file_input) { cv::flip(input_frame, input_frame, /*flipcode=HORIZONTAL*/ 1); }

    auto image_frame_ptr = std::make_shared<mediapipe_v01013_based::ImageFrame>(
        mediapipe_v01013_based::ImageFormat::SRGB,
        input_frame.cols, input_frame.rows,
        mediapipe_v01013_based::ImageFrame::kDefaultAlignmentBoundary);
    auto image_ptr = std::make_shared<const mediapipe_v01013_based::Image>(image_frame_ptr);

    // pass the image to the hand tracking workflow
    absl::StatusOr<std::unique_ptr<mediapipe_v01013_based::ImageHandTrackingAndInferenceResult>> inference;
    MP_ASSIGN_OR_RETURN(inference, hand_tracking.Process(image_ptr, 3));

    // put the result into protobuf format just in case we want to log it to file for offline juxtaposing to other runs' output
    // which is typically useful during development phases. we use protobuf format just because we already have code
    // for serializing/deserializing and comparing protobuf messages due to working within the mediapipe framework.
    // it can be any other format that's convenient to work with.
    mediapipe_v01013_based::PipelineOutputData inference_as_proto_msg;
    inference_as_proto_msg.set_frame_number(i);
    for (const auto& ol : *inference.value()->object_landmarkss) {
      *inference_as_proto_msg.add_multi_hand_world_landmarks() = ol;
    }
    for (const auto& vl : *inference.value()->viewport_landmarkss) {
      *inference_as_proto_msg.add_multi_hand_landmarks() = vl;
    }
    for (const auto& h : *inference.value()->handedness_classifications) {
      *inference_as_proto_msg.add_multi_handedness() = h;
    }
    // write the current frame's inference to file for optional offline analysis
    google::protobuf::util::SerializeDelimitedToOstream(inference_as_proto_msg, &output_proto_file);

    // Compare with reference data if provided
    if (diffing) {
      if (i < reference_data.size()) {
        google::protobuf::util::MessageDifferencer differ;
        std::string diff;
        differ.ReportDifferencesToString(&diff);

        if (!differ.Compare(inference_as_proto_msg, reference_data[i])) {
          ABSL_LOG(ERROR) << "Pipeline output at frame " << i << " is different than the reference output:\n" << diff;
          ABSL_LOG(ERROR) << "terminating early due to difference in output at frame " << i;
          break; // Early termination due to difference
        } else { ABSL_LOG(INFO) << "pipeline output for frame " << i << " is identical to its reference output read from the reference data file"; }
      } else { ABSL_LOG(WARNING) << "reference output file doesn't have data for frame " << i << " (it has only " << reference_data.size() << " records)"; }
    }
  }

  output_proto_file.close();
  ABSL_LOG(INFO) << kOutputProtoFilename << " was written";

  // absl::Status finalize_status = pipeline_operator->finalize();
  // if (!pipeline_operator->finalize().ok()) {
  //   ABSL_LOG(ERROR) << "Error during mediapipe graph finalization: " << finalize_status.message();
  //   return finalize_status;
  // }

  return absl::OkStatus();
}

// Validates and logs the PROJECT_ROOT_DIR env variable.
void ValidateProjectRootDirectoryOrExit() {
    const char* root_dir = std::getenv("PROJECT_ROOT_DIR");
    if (!root_dir || std::string(root_dir).empty()) {
      std::cerr << "ERROR: the environment variable PROJECT_ROOT_DIR must be set.";
      std::exit(EXIT_FAILURE);
    }
    std::string dir_str(root_dir);
    if (!std::filesystem::path(dir_str).is_absolute()) {
        std::cerr << "ERROR: the environment variable PROJECT_ROOT_DIR must be an absolute path, but got: '" << dir_str << "'\n";
        std::exit(EXIT_FAILURE);
    }
    if (!std::filesystem::exists(dir_str)) {
        std::cerr << "ERROR: the path provided by the environment variable PROJECT_ROOT_DIR does not exist: '" << dir_str << "'\n";
        std::exit(EXIT_FAILURE);
    }
    if (!std::filesystem::is_directory(dir_str)) {
        std::cerr << "ERROR: the value provided by the environment variable PROJECT_ROOT_DIR is not a directory: '" << dir_str << "'\n";
        std::exit(EXIT_FAILURE);
    }
    std::cerr << "PROJECT_ROOT_DIR is set to: '" << dir_str << "'\n";
}

// Checks that the given file exists. Exits with error if not.
void CheckFileExistsOrExit(const std::string& file_path, const char* flag_name) {
    if (!file_path.empty() && !std::filesystem::exists(file_path)) {
        std::cerr << "ERROR: File specified by --" << flag_name << " does not exist: '" << file_path << "'\n";
        std::exit(EXIT_FAILURE);
    }
}

int main(int argc, char** argv) {
  ValidateProjectRootDirectoryOrExit();

  google::InitGoogleLogging(argv[0]);
  ABSL_LOG(INFO) << "this is the pure c++ pipeline runner";
  ABSL_LOG(INFO) << "working directory: " << std::filesystem::current_path();

  absl::ParseCommandLine(argc, argv);

  // Check input files exist before use
  CheckFileExistsOrExit(GetProjectRootedPath(absl::GetFlag(FLAGS_graph_file)), "graph_file");
  if (!absl::GetFlag(FLAGS_input_video_path).empty()) {
    CheckFileExistsOrExit(GetProjectRootedPath(absl::GetFlag(FLAGS_input_video_path)), "input_video_path");
  }
  // Optionally check reference proto file if you want to require its existence
  // CheckFileExistsOrExit(GetProjectRootedPath(kReferenceProtoFilename), "reference_proto_filename");

  absl::Status run_status = RunPipelineWithDiffing();
  if (!run_status.ok()) {
    ABSL_LOG(INFO) << "aborted due to the following reason: " << run_status.message();
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
