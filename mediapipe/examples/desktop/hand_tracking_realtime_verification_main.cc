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
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/classification.pb.h"

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/opencv_highgui_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/opencv_video_inc.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/util/resource_util.h"

constexpr char kInputStream[] = "image";

ABSL_FLAG(std::string, calculator_graph_config_file, "",
          "Name of file containing text format CalculatorGraphConfig proto.");
ABSL_FLAG(std::string, input_video_path, "",
          "Full path of video to load. "
          "If not provided, attempt to use a webcam.");
ABSL_FLAG(std::string, output_video_path, "",
          "Full path of where to save result (.mp4 only). "
          "If not provided, show result in a window.");

absl::Status RunMPPGraph() {

  std::string calculator_graph_config_contents;
  MP_RETURN_IF_ERROR(mediapipe::file::GetContents(
      absl::GetFlag(FLAGS_calculator_graph_config_file),
      &calculator_graph_config_contents));
  mediapipe::CalculatorGraphConfig config =
      mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(
          calculator_graph_config_contents);

  // set of expected output streams
  const std::vector<std::string> graph_output_streams_names = {
    "multi_hand_landmarks",
    "multi_hand_world_landmarks",
    "multi_handedness",
    "hand_rects_from_palm_detections"
  };

  // Initializing the calculator graph
  mediapipe::CalculatorGraph graph;

  MP_RETURN_IF_ERROR(graph.Initialize(config));

  // poller per each expected output stream
  std::map<std::string, mediapipe::OutputStreamPoller> pollers;
  for (const auto& stream : graph_output_streams_names) {
    MP_ASSIGN_OR_RETURN(auto poller, graph.AddOutputStreamPoller(stream));
    pollers.emplace(stream, std::move(poller));
  }

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

  MP_RETURN_IF_ERROR(graph.StartRun({}));

  // Initialize output protobuf file (overwrite if exists)
  std::ofstream output_proto_file("output_data.pb", std::ios::binary | std::ios::trunc);
  if (!output_proto_file.is_open()) {
    return absl::InternalError("failed to open output_data.pb for writing");
  }

  // process all input frames
  for (int i = 0; i < 200; ++i) {

    ABSL_LOG(WARNING) << "frame number: " << i;

    std::map<std::string, bool> stream_outputs;
    for (const auto& stream_name : graph_output_streams_names) stream_outputs[stream_name] = false;

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

    // Wrap Mat into an ImageFrame.
    auto pipeline_ready_frame = absl::make_unique<mediapipe::ImageFrame>(
        mediapipe::ImageFormat::SRGB, input_frame.cols, input_frame.rows, mediapipe::ImageFrame::kDefaultAlignmentBoundary);
    cv::Mat input_frame_mat = mediapipe::formats::MatView(pipeline_ready_frame.get());
    input_frame.copyTo(input_frame_mat);

    // Send image packet into the graph.
    size_t frame_timestamp_us =
        (double)cv::getTickCount() / (double)cv::getTickFrequency() * 1e6;
    MP_RETURN_IF_ERROR(graph.AddPacketToInputStream(kInputStream, mediapipe::Adopt(pipeline_ready_frame.release()).At(mediapipe::Timestamp(frame_timestamp_us))));

    // Wait for graph to finish its processing of the current input fed to it
    MP_RETURN_IF_ERROR(graph.WaitUntilIdle());

    // then take out its (zero or) single expected packet per each of its output streams ―
    // currently when there's no hand detection, the graph will emit zero packets for
    // all of its output streams (this can change during the liberation development phase)
    // Declare output variables for all streams, accessible outside the if-else block
    std::vector<mediapipe::LandmarkList> hand_landmarks;
    std::vector<mediapipe::NormalizedLandmarkList> world_hand_landmarks;
    std::vector<mediapipe::ClassificationList> handedness;
    std::vector<mediapipe::NormalizedRect> hand_rects;

    for (auto& poller_pair : pollers) {
      auto& stream_name = poller_pair.first;
      auto& poller = poller_pair.second;

      if (poller.QueueSize() == 0) {
        ABSL_LOG(INFO) << "no packets in the " << stream_name << " poller queue, in this iteration";
      } else if (poller.QueueSize() > 1) {
        return absl::InternalError("More than one packet is available on the output stream " + stream_name + " (" + std::to_string(poller.QueueSize()) + ") poller queue, in this iteration");
      } else { // one packet is available

        mediapipe::Packet packet;
        MP_RETURN_IF_ERROR(poller.Next(&packet) ? absl::OkStatus() : absl::InternalError("poller queue size was one, but failed to retrieve its single packet, which is unexpected."));

        // extract each stream's typed packet, we can be more elegant by using a map from poller to variable pointer or something, but each name will still be explicitly used also in that case.
        // so avoiding over-complicated mapping from stream name to variable pointer is less code and simpler to read.
        if (stream_name == "multi_hand_world_landmarks") {
          hand_landmarks = packet.Get<std::vector<mediapipe::LandmarkList>>();
        } else if (stream_name == "multi_hand_landmarks") {
          world_hand_landmarks = packet.Get<std::vector<mediapipe::NormalizedLandmarkList>>();
        } else if (stream_name == "multi_handedness") {
          handedness = packet.Get<std::vector<mediapipe::ClassificationList>>();
        } else if (stream_name == "hand_rects_from_palm_detections") {
          hand_rects = packet.Get<std::vector<mediapipe::NormalizedRect>>();
        }

        stream_outputs[stream_name] = true;
        }
      }

    bool all_streams_have_output = std::all_of(stream_outputs.begin(), stream_outputs.end(), [](const auto& pair) { return pair.second; });
    if (!all_streams_have_output) { ABSL_LOG(INFO) << "at least one of the expected output streams has no packet in this iteration"; }

    // fill up a typed protobuf message containing the data from all output streams
    mediapipe::PipelineOutputData stream_data_msg;
    stream_data_msg.set_frame_number(i);
    for (const auto& l : hand_landmarks) {
      ABSL_LOG(INFO) << "number of lm collections: " << hand_landmarks.size();
      *stream_data_msg.add_multi_hand_world_landmarks() = l;
    }
    for (const auto& l : world_hand_landmarks) {
      *stream_data_msg.add_multi_hand_landmarks() = l;
    }
    for (const auto& c : handedness) {
      *stream_data_msg.add_multi_handedness() = c;
    }
    for (const auto& r : hand_rects) {
      *stream_data_msg.add_hand_rects_from_palm_detections() = r;
    }
    // log the size of the message:
    stream_data_msg.SerializeToOstream(&output_proto_file);
  }

  output_proto_file.close();
  ABSL_LOG(INFO) << "mediapipe graph shutting down";
  ABSL_LOG(INFO) << "output_data.pb was written";

  MP_RETURN_IF_ERROR(graph.CloseInputStream(kInputStream));
  return graph.WaitUntilDone();
}

int main(int argc, char** argv) {

  // the following logging intialization made our VLOG macro uses swallow, so don't use VLOG.
  // a subtle ABSL-GLOG interaction which other modules don't bump into but this one does, wasted 80 minutes in deep exploration of it and gave up.
  // motivation was that VLOG can log conditionally at the module level (e.g. only log for our module etc. a feature which ABSL_LOG does not seem to have)
  // something in the build dependencies order makes VLOG not log for the current module but only for original modules, which is too subtle to capture
  // at any budget of time that's proportional.
  google::InitGoogleLogging(argv[0]);

  absl::ParseCommandLine(argc, argv);

  absl::Status run_status = RunMPPGraph();
  if (!run_status.ok()) {
    ABSL_LOG(INFO) << "failed to run the mediapipe graph due to the following issue: " << run_status.message();
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
