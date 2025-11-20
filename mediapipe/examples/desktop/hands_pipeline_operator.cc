/// a class for succinctly operating our mediapipe pipeline of interest from cpp code.
/// this was not explicitly implemented in mediapipe v0.10.13 itself other than direct
/// use of the underlying graph object methods in its tests code, which is a little
/// more detailed than a bottom-line api for pipeline use should be ― the current
/// class provides that clean surface for pipeline use which direct use of
/// the graph object is not.
///
/// in addition to wrapping around the mediapipe graph object, this class is additionally
/// designed in some ways to be conducive to our C API being based on wrapping this class,
/// for its designation of exposing pipeline use to C and FFI code consumers safely.
///
/// this convergence of concerns can be opted out by a refactor making our C API drive
/// the pipeline on its own and this class be left as only catering to providing
/// a succinct safe facade over the hairy way of interacting with a pipeline only
/// for use from C++ code.
///
/// either way, c.f. the elaborated comment in the header file.

// Copyright 2025 The MediaPipe Authors.
// Licensed under the Apache License, Version 2.0.

#include "mediapipe/examples/desktop/hands_pipeline_operator.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/port/status_macros.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/parse_text_proto.h"

namespace hand_tracking_mp_lean {

// factory method instantiating an instance of this class, initializing it,
// providing ABSL upward error propagation, and returning the initialized instance
// when successful.
absl::StatusOr<std::unique_ptr<HandsPipelineOperator>> HandsPipelineOperator::Create(
    const uint32_t max_hands_to_track,
    const std::string& graph_file_path,
    const std::vector<std::string>& output_streams) {

  std::string graph_content;
  absl::Status status = hand_tracking_mp_lean::file::GetContents(graph_file_path, &graph_content);
  if (!status.ok()) {
    return absl::InvalidArgumentError(absl::StrCat("failed to read medaipipe graph file: ", graph_file_path, " - ", status.message()));
  }

  CalculatorGraphConfig config;
  if (!hand_tracking_mp_lean::ParseTextProto<CalculatorGraphConfig>(graph_content, &config)) {
    return absl::InvalidArgumentError(absl::StrCat("failed to parse the provided mediapipe graph file: ", graph_file_path));
  }
  ABSL_LOG(INFO) << "mediapipe graph " << graph_file_path << " successfully parsed";

  auto hands_pipeline_operator = std::make_unique<HandsPipelineOperator>(max_hands_to_track, output_streams);

  MP_RETURN_IF_ERROR(hands_pipeline_operator->graph_.Initialize(config, {{"num_hands", MakePacket<int>(static_cast<uint32_t>(max_hands_to_track))}}));

  for (const auto& stream : output_streams) {
    auto poller_status = hands_pipeline_operator->graph_.AddOutputStreamPoller(stream);
    if (poller_status.ok()) {
      hands_pipeline_operator->pollers_.emplace(stream, std::move(poller_status.value()));
    }
  }

  MP_RETURN_IF_ERROR(hands_pipeline_operator->graph_.StartRun({}));
  ABSL_LOG(INFO) << "mediapipe graph " << graph_file_path << " started";
  return hands_pipeline_operator;
}

HandsPipelineOperator::HandsPipelineOperator(const uint32_t max_hands_to_track, const std::vector<std::string>& output_streams)
    : max_hands_to_track_(max_hands_to_track), output_streams_names_(output_streams) {}

HandsPipelineOperator::~HandsPipelineOperator() = default;

absl::Status HandsPipelineOperator::PushImage(const cv::Mat& input_frame, int64_t timestamp_us) {
  auto frame = absl::make_unique<ImageFrame>(ImageFormat::SRGB, input_frame.cols, input_frame.rows, ImageFrame::kDefaultAlignmentBoundary);
  cv::Mat frame_mat = formats::MatView(frame.get());
  input_frame.copyTo(frame_mat);
  return graph_.AddPacketToInputStream("image", Adopt(frame.release()).At(Timestamp(timestamp_us)));
}

absl::Status HandsPipelineOperator::WaitForOutput(PipelineOutputData* output, int frame_number) {
  absl::Status status = graph_.WaitUntilIdle();
  if (!status.ok()) return status;
  std::vector<LandmarkList> hand_landmarks;
  std::vector<NormalizedLandmarkList> world_hand_landmarks;
  std::vector<ClassificationList> handedness;
  std::vector<NormalizedRect> hand_rects;

  for (auto& poller_pair : pollers_) {
    auto& stream_name = poller_pair.first;
    auto& poller = poller_pair.second;
    if (poller.QueueSize() == 1) {
      Packet packet;
      if (poller.Next(&packet)) {
        if (stream_name == "multi_hand_world_landmarks") {
          hand_landmarks = packet.Get<std::vector<LandmarkList>>();
        } else if (stream_name == "multi_hand_landmarks") {
          world_hand_landmarks = packet.Get<std::vector<NormalizedLandmarkList>>();
        } else if (stream_name == "multi_handedness") {
          handedness = packet.Get<std::vector<ClassificationList>>();
        } else if (stream_name == "hand_rects_from_palm_detections") {
          hand_rects = packet.Get<std::vector<NormalizedRect>>();
        }
      }
    }
  }

  output->set_frame_number(frame_number);
  for (const auto& l : hand_landmarks) {
    *output->add_multi_hand_world_landmarks() = l;
  }
  for (const auto& l : world_hand_landmarks) {
    *output->add_multi_hand_landmarks() = l;
  }
  for (const auto& c : handedness) {
    *output->add_multi_handedness() = c;
  }
  // for (const auto& r : hand_rects) {
  //   *output->add_hand_rects_from_palm_detections() = r;
  // }
  return absl::OkStatus();
}

absl::Status HandsPipelineOperator::Finalize() {
  absl::Status status = graph_.CloseInputStream("image");
  if (!status.ok()) return status;
  return graph_.WaitUntilDone();
}

}  // namespace hand_tracking_mp_lean
