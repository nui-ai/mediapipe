// Copyright 2025 The MediaPipe Authors.
// Licensed under the Apache License, Version 2.0.
//
// Header for HandsPipelineOperator: Encapsulates MediaPipe graph operations for hand tracking.

#ifndef MEDIAPIPE_EXAMPLES_DESKTOP_HANDS_PIPELINE_OPERATOR_H_
#define MEDIAPIPE_EXAMPLES_DESKTOP_HANDS_PIPELINE_OPERATOR_H_

#include <map>
#include <string>
#include <vector>
#include <opencv2/core.hpp>
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/examples/desktop/pipeline_output.pb.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/framework/formats/classification.pb.h"
#include "mediapipe/framework/port/status.h"

namespace mediapipe {

class HandsPipelineOperator {
 public:
  HandsPipelineOperator(const CalculatorGraphConfig& config,
                       const std::vector<std::string>& output_streams);
  ~HandsPipelineOperator();

  absl::Status push_image(const cv::Mat& input_frame, int64_t timestamp_us);
  absl::Status wait_for_output(PipelineOutputData* output, int frame_number);
  absl::Status finalize();

 private:
  CalculatorGraph graph_;
  std::vector<std::string> output_streams_names_;
  std::map<std::string, OutputStreamPoller> pollers_;
};

}  // namespace mediapipe

#endif  // MEDIAPIPE_EXAMPLES_DESKTOP_HANDS_PIPELINE_OPERATOR_H_

