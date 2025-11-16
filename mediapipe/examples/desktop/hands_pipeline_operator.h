// Copyright 2025 The MediaPipe Authors.
// Licensed under the Apache License, Version 2.0.
//
// Header for HandsPipelineOperator: Encapsulates MediaPipe graph operations for hand tracking.

#ifndef MEDIAPIPE_EXAMPLES_DESKTOP_HANDS_PIPELINE_OPERATOR_H_
#define MEDIAPIPE_EXAMPLES_DESKTOP_HANDS_PIPELINE_OPERATOR_H_

#include <map>
#include <memory>
#include <string>
#include <vector>
#include <opencv2/core.hpp>
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/examples/desktop/pipeline_output.pb.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/framework/formats/classification.pb.h"

namespace hand_tracking_mp_lean {

/// a class for succinctly operating our mediapipe pipeline of interest from cpp code.
/// this was not explicitly implemented in mediapipe v0.10.13 itself other than use
/// of the underlying graph object methods in its tests code.
///
/// in addition to wrapping around the mediapipe graph object, this class is additionally
/// designed such that our C API can use it to expose the functionality to C and FFI code:
///
/// a C-compatible creation and a finalization methods are provided, as C cannot instantiate
/// a C++ instance directly and accordingly needs to explicitly handle destroying objects.
/// these C compatibility methods should have been part of the C API wrapper not this class,
/// in a simpler modularity implementation.
///
/// intended usage from C API:
///
///  - initialization
///  - pushing an image
///  - getting the pipeline output for the pushed image (by waiting for it)
///  - finalization
///
/// as mentioned, its more modular to push out the C API requirements to lie
/// within the C API leaving the current class to only provide the basic
/// convenience wrapping for operating a mediapipe pipeline.
class HandsPipelineOperator {
 public:

  /// a factory method designated for instantiating and initializing an instance of this class,
  /// to be used instead of direct use of the class's constructor.
  ///
  /// it performs various checks which, if performed inside a constructor, would not be able to nicely
  /// participate in the ABSL error propagation pattern being used across this codebase for upwards
  /// error propagation.
  ///
  /// alternatively we'd have to introduce a separate "Initialize" function that would also
  /// have to be called after instantiating. this is probably not the same approach as the
  /// general medipipe codebase approach where you'd instantiate and then call a separate
  /// function to initialize the instantiated object.
  static absl::StatusOr<std::unique_ptr<HandsPipelineOperator>> Create(
    uint32_t,
    const std::string& graph_file_path,
    const std::vector<std::string>& output_streams);

  // do not directly use the constructor, but instead use the below Create factory method.
  HandsPipelineOperator(uint32_t, const std::vector<std::string>& output_streams);

  absl::Status PushImage(const cv::Mat& input_frame, int64_t timestamp_us);

  absl::Status WaitForOutput(PipelineOutputData* output, int frame_number);

  absl::Status Finalize();

  ~HandsPipelineOperator();

private:

  CalculatorGraph graph_;
  const uint32_t max_hands_to_track_;
  std::vector<std::string> output_streams_names_;
  std::map<std::string, OutputStreamPoller> pollers_;
};

}
#endif
