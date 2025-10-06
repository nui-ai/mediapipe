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
// An example of sending OpenCV webcam frames into a MediaPipe graph.
#include <cstdlib>
#include <fstream>

#include "mediapipe/examples/desktop/pipeline_output.pb.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/examples/desktop/hands_pipeline_operator.h"

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

constexpr char kInputStream[] = "image";
constexpr char kOutputProtoFilename[] = "output_data_cpp.pb";
constexpr char kReferenceProtoFilename[] = "output_data_v0.10.13.pb";

ABSL_FLAG(std::string, graph_file, "",
          "Name of pipeline pbtxt file.");
ABSL_FLAG(std::string, input_video_path, "",
          "Full path of video to load. "
          "If not provided, attempt to use a webcam.");
ABSL_FLAG(std::string, output_video_path, "",
          "Full path of where to save result (.mp4 only). "
          "If not provided, show result in a window.");

// helper function to read the output reference data from file
bool ReadReferenceData(const std::string& filename, std::vector<mediapipe::PipelineOutputData>& out) {
    std::ifstream input(filename, std::ios::binary);
    if (!input.is_open()) {
        ABSL_LOG(ERROR) << "Failed to open reference file: " << filename;
        return false;
    }
    google::protobuf::io::IstreamInputStream zero_copy_input(&input);
    bool clean_eof = false;
    int msg_count = 0;
    while (true) {
        mediapipe::PipelineOutputData msg;
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
    ABSL_LOG(INFO) << "Successfully loaded " << msg_count << " reference records from " << filename;
    return true;
}

absl::Status RunPipelineWithDiffing() {

  // Load reference data from output_data_v0.10.13.pb
  std::vector<mediapipe::PipelineOutputData> reference_data;
  if (!ReadReferenceData(kReferenceProtoFilename, reference_data)) {
    ABSL_LOG(WARNING) << "failed to load reference data from " << kReferenceProtoFilename
                     << ". will proceed without real-time comparison.";
  } else {
    ABSL_LOG(INFO) << "loaded " << reference_data.size() << " records from the reference data file.";
  }

  std::string graph_protobuf_definition;  // the text protobuf definition of the graph, typically loaded from a pbtxt file
  MP_RETURN_IF_ERROR(mediapipe::file::GetContents(absl::GetFlag(FLAGS_graph_file), &graph_protobuf_definition));
  mediapipe::CalculatorGraphConfig pipeline_definition = mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(graph_protobuf_definition);

  // set of expected pipeline output streams
  const std::vector<std::string> graph_output_streams_names = {
    "multi_hand_landmarks",
    "multi_hand_world_landmarks",
    "multi_handedness",
    // "hand_rects_from_palm_detections"
  };

  mediapipe::HandsPipelineOperator pipeline_operator(pipeline_definition, graph_output_streams_names);

  // initializing the camera or load the input video
  cv::VideoCapture capture;
  const bool video_file_input = !absl::GetFlag(FLAGS_input_video_path).empty();
  if (video_file_input) {
    capture.open(absl::GetFlag(FLAGS_input_video_path));
  } else {
    capture.open(0);
  }
  RET_CHECK(capture.isOpened());

  ABSL_LOG(INFO) << "starting a mediapipe graph";

  // Initialize output protobuf file (overwrite if exists)
  std::ofstream output_proto_file(kOutputProtoFilename, std::ios::binary | std::ios::trunc);
  if (!output_proto_file.is_open()) {
    return absl::InternalError(std::string("failed to open ") + kOutputProtoFilename + " for writing");
  }

  // process all input frames
  for (int i = 0; i < 999999; ++i) {

    ABSL_LOG(WARNING) << "frame number: " << i;
    cv::Mat input_frame_raw;
    capture >> input_frame_raw;
    if (input_frame_raw.empty()) {
      if (!video_file_input) {
        ABSL_LOG(WARNING) << "empty frame from camera being ignored";
        continue;
      }
      break; // end of video file (irrelevant when the input is from a camera)
    }

    cv::Mat input_frame;
    cv::cvtColor(input_frame_raw, input_frame, cv::COLOR_BGR2RGB);
    if (!video_file_input) { cv::flip(input_frame, input_frame, /*flipcode=HORIZONTAL*/ 1); }

    size_t frame_timestamp_us = (double)cv::getTickCount() / (double)cv::getTickFrequency() * 1e6;
    MP_RETURN_IF_ERROR(pipeline_operator.push_image(input_frame, frame_timestamp_us));
    mediapipe::PipelineOutputData stream_data_msg;
    MP_RETURN_IF_ERROR(pipeline_operator.wait_for_output(&stream_data_msg, i));

    // write the current frame output to file
    google::protobuf::util::SerializeDelimitedToOstream(stream_data_msg, &output_proto_file);

    // Compare with reference data if available
    if (!reference_data.empty()) {
      if (i < reference_data.size()) {
        google::protobuf::util::MessageDifferencer differ;
        std::string diff;
        differ.ReportDifferencesToString(&diff);

        if (!differ.Compare(stream_data_msg, reference_data[i])) {
          ABSL_LOG(ERROR) << "Pipeline output at frame " << i << " is different than the reference output:\n" << diff;
          ABSL_LOG(ERROR) << "terminating early due to difference in output at frame " << i;
          break; // Early termination due to difference
        } else { ABSL_LOG(INFO) << "pipeline output for frame " << i << " is identical to its reference output read from " << kReferenceProtoFilename; }
      } else { ABSL_LOG(WARNING) << "reference output file doesn't have data for frame " << i << " (it has only " << reference_data.size() << " records)"; }
    }
  }

  output_proto_file.close();
  ABSL_LOG(INFO) << kOutputProtoFilename << " was written";

  absl::Status finalize_status = pipeline_operator.finalize();
  if (!pipeline_operator.finalize().ok()) {
    ABSL_LOG(ERROR) << "Error during mediapipe graph finalization: " << finalize_status.message();
    return finalize_status;
  }

  return absl::OkStatus();
}

int main(int argc, char** argv) {

  // the following logging intialization made our VLOG macro uses swallow, so don't use VLOG.
  // a subtle ABSL-GLOG interaction which other modules don't bump into but this one does, wasted 80 minutes in deep exploration of it and gave up.
  // motivation was that VLOG can log conditionally at the module level (e.g. only log for our module etc. a feature which ABSL_LOG does not seem to have)
  // something in the build dependencies order makes VLOG not log for the current module but only for original modules, which is too subtle to capture
  // at any budget of time that's proportional.
  google::InitGoogleLogging(argv[0]);

  ABSL_LOG(INFO) << "this is the pure c++ pipeline runner";

  absl::ParseCommandLine(argc, argv);

  absl::Status run_status = RunPipelineWithDiffing();
  if (!run_status.ok()) {
    ABSL_LOG(INFO) << "failed to run the mediapipe graph due to the following issue: " << run_status.message();
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
