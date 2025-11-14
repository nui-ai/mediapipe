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
#include <vector>

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/formats/location.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/calculators/util/detection_letterbox_removal.h"

namespace hand_tracking_mp_lean {

namespace {

constexpr char kDetectionsTag[] = "DETECTIONS";
constexpr char kLetterboxPaddingTag[] = "LETTERBOX_PADDING";

}  // namespace

// Adjusts detection locations on a letterboxed image to the corresponding
// locations on the same image with the letterbox removed. This is useful to map
// the detections inferred from a letterboxed image, for example, output of
// the ImageTransformationCalculator when the scale mode is FIT, back to the
// corresponding input image before letterboxing.
//
// Input:
//   DETECTIONS: An std::vector<Detection> representing detections on an
//   letterboxed image.
//
//   LETTERBOX_PADDING: An std::array<float, 4> representing the letterbox
//   padding from the 4 sides ([left, top, right, bottom]) of the letterboxed
//   image, normalized to [0.f, 1.f] by the letterboxed image dimensions.
//
// Output:
//   DETECTIONS: An std::vector<Detection> representing detections with their
//   locations adjusted to the letterbox-removed (non-padded) image.
//
// Usage example:
// node {
//   calculator: "DetectionLetterboxRemovalCalculator"
//   input_stream: "DETECTIONS:detections"
//   input_stream: "LETTERBOX_PADDING:letterbox_padding"
//   output_stream: "DETECTIONS:adjusted_detections"
// }
class UnletterboxCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc) {
    RET_CHECK(cc->Inputs().HasTag(kDetectionsTag) &&
              cc->Inputs().HasTag(kLetterboxPaddingTag))
        << "Missing one or more input streams.";

    cc->Inputs().Tag(kDetectionsTag).Set<std::vector<Detection>>();
    cc->Inputs().Tag(kLetterboxPaddingTag).Set<std::array<float, 4>>();

    cc->Outputs().Tag(kDetectionsTag).Set<std::vector<Detection>>();

    return absl::OkStatus();
  }

  absl::Status Open(CalculatorContext* cc) override {
    cc->SetOffset(TimestampDiff(0));

    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) override {
    // Only process if there's input detections.
    if (cc->Inputs().Tag(kDetectionsTag).IsEmpty()) {
      return absl::OkStatus();
    }

    const auto& input_detections =
        cc->Inputs().Tag(kDetectionsTag).Get<std::vector<Detection>>();
    const auto& letterbox_padding =
        cc->Inputs().Tag(kLetterboxPaddingTag).Get<std::array<float, 4>>();

    auto output_detections = UnLetterBox(
        input_detections, letterbox_padding);

    cc->Outputs()
        .Tag(kDetectionsTag)
        .Add(output_detections.release(), cc->InputTimestamp());
    return absl::OkStatus();
  }
};
REGISTER_CALCULATOR(UnletterboxCalculator);

}  // namespace hand_tracking_mp_lean
