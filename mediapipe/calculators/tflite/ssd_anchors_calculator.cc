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

#include <cmath>
#include <utility>
#include <vector>

#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "mediapipe/calculators/tflite/ssd_anchors_calculator.pb.h"
#include "mediapipe/calculators/tflite/ssd_anchors_calculator_utils.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/object_detection/anchor.pb.h"
#include "mediapipe/framework/port/ret_check.h"

namespace mediapipe {

// Generate anchors for SSD object detection model.
// Output:
//   ANCHORS: A list of anchors. Model generates predictions based on the
//   offsets of these anchors.
//
// Usage example:
// node {
//   calculator: "SsdAnchorsCalculator"
//   output_side_packet: "anchors"
//   options {
//     [mediapipe.SsdAnchorsCalculatorOptions.ext] {
//       num_layers: 6
//       min_scale: 0.2
//       max_scale: 0.95
//       input_size_height: 300
//       input_size_width: 300
//       anchor_offset_x: 0.5
//       anchor_offset_y: 0.5
//       strides: 16
//       strides: 32
//       strides: 64
//       strides: 128
//       strides: 256
//       strides: 512
//       aspect_ratios: 1.0
//       aspect_ratios: 2.0
//       aspect_ratios: 0.5
//       aspect_ratios: 3.0
//       aspect_ratios: 0.3333
//       reduce_boxes_in_lowest_layer: true
//     }
//   }
// }
class SsdAnchorsCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc) {
    cc->OutputSidePackets().Index(0).Set<std::vector<Anchor>>();
    return absl::OkStatus();
  }

  absl::Status Open(CalculatorContext* cc) override {
    cc->SetOffset(TimestampDiff(0));

    const SsdAnchorsCalculatorOptions& options =
        cc->Options<SsdAnchorsCalculatorOptions>();

    auto anchors = absl::make_unique<std::vector<Anchor>>();
    if (!options.fixed_anchors().empty()) {
      // Check fields for generating anchors are not set.
      if (options.has_input_size_height() || options.has_input_size_width() ||
          options.has_min_scale() || options.has_max_scale() ||
          options.has_num_layers() || options.multiscale_anchor_generation()) {
        return absl::InvalidArgumentError(
            "Fixed anchors are provided, but fields are set for generating "
            "anchors. When fixed anchors are set, fields for generating "
            "anchors must not be set.");
      }
      anchors->assign(options.fixed_anchors().begin(),
                      options.fixed_anchors().end());
      cc->OutputSidePackets().Index(0).Set(Adopt(anchors.release()));
      return absl::OkStatus();
    }
    MP_RETURN_IF_ERROR(
        SsdAnchorsCalculatorUtils::GenerateAnchors(anchors.get(), options));
    cc->OutputSidePackets().Index(0).Set(Adopt(anchors.release()));
    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) override {
    return absl::OkStatus();
  }
};

REGISTER_CALCULATOR(SsdAnchorsCalculator);

}  // namespace mediapipe
