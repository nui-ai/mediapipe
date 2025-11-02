// Copyright 2025 The MediaPipe Authors.
// Licensed under the Apache License, Version 2.0.
//
// HandsPipelineOperator: Encapsulates operating a MediaPipe pipeline

#include "mediapipe/examples/desktop/hands_pipeline_operator.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/port/status_macros.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/parse_text_proto.h"

namespace mediapipe_v01013_based {

absl::StatusOr<std::unique_ptr<HandsPipelineOperator>> HandsPipelineOperator::Create(
    const std::string& graph_file_path,
    const std::vector<std::string>& output_streams) {

  std::string graph_content;
  absl::Status status = mediapipe_v01013_based::file::GetContents(graph_file_path, &graph_content);
  if (!status.ok()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Failed to read graph file: ", graph_file_path, " - ", status.message()));
  } else ABSL_LOG(INFO) << "Read graph file: " << graph_file_path;

  CalculatorGraphConfig config;
  if (!mediapipe_v01013_based::ParseTextProto<CalculatorGraphConfig>(graph_content, &config)) {
    return absl::InvalidArgumentError(
        absl::StrCat("Failed to parse graph file: ", graph_file_path));
  } else ABSL_LOG(INFO) << "Successfully parsed graph file " << graph_file_path;

  auto op = std::unique_ptr<HandsPipelineOperator>(new HandsPipelineOperator(output_streams));
  status = op->graph_.Initialize(config);

  if (!status.ok()) {
    return absl::InternalError(
        absl::StrCat("Graph initialization failed: ", status.message()));
  } else ABSL_LOG(INFO) << "Successfully initialized the graph";

  for (const auto& stream : output_streams) {
    auto poller_status = op->graph_.AddOutputStreamPoller(stream);
    if (poller_status.ok()) {
      op->pollers_.emplace(stream, std::move(poller_status.value()));
    }
  }
  status = op->graph_.StartRun({});
  if (!status.ok()) {
    return absl::InternalError(
        absl::StrCat("Graph start failed: ", status.message()));
  }
  return op;
}

HandsPipelineOperator::HandsPipelineOperator(const std::vector<std::string>& output_streams)
    : output_streams_names_(output_streams) {}

HandsPipelineOperator::~HandsPipelineOperator() = default;

absl::Status HandsPipelineOperator::push_image(const cv::Mat& input_frame, int64_t timestamp_us) {
  auto frame = absl::make_unique<ImageFrame>(ImageFormat::SRGB, input_frame.cols, input_frame.rows, ImageFrame::kDefaultAlignmentBoundary);
  cv::Mat frame_mat = formats::MatView(frame.get());
  input_frame.copyTo(frame_mat);
  return graph_.AddPacketToInputStream("image", Adopt(frame.release()).At(Timestamp(timestamp_us)));
}

absl::Status HandsPipelineOperator::wait_for_output(PipelineOutputData* output, int frame_number) {
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

absl::Status HandsPipelineOperator::finalize() {
  absl::Status status = graph_.CloseInputStream("image");
  if (!status.ok()) return status;
  return graph_.WaitUntilDone();
}

}  // namespace mediapipe_v01013_based
