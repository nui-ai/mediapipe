// Copyright 2020 The MediaPipe Authors.
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

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_options.pb.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/modules/hand_landmark/calculators/hand_landmarks_to_rect_calculator_core.h"

namespace mediapipe_v01013_based {

using ::mediapipe_v01013_based::NormalizedRect;

namespace {

// NORM_LANDMARKS is either the full set of landmarks for the hand, or
// a subset of the hand landmarks (indices 0, 1, 2, 3, 5, 6, 9, 10, 13, 14,
// 17 and 18). The latter is the legacy behavior, please just pass in
// the full set of hand landmarks.
//
// TODO: update clients to just pass all the landmarks in.
constexpr char kNormalizedLandmarksTag[] = "NORM_LANDMARKS";
constexpr char kNormRectTag[] = "NORM_RECT";
constexpr char kImageSizeTag[] = "IMAGE_SIZE";
// Indices within the partial landmarks.
constexpr int kWristJoint = 0;
constexpr int kMiddleFingerPIPJoint = 6;
constexpr int kIndexFingerPIPJoint = 4;
constexpr int kRingFingerPIPJoint = 8;
constexpr int kNumLandmarks = 21;
constexpr float kTargetAngle = M_PI * 0.5f;

}  // namespace

// A calculator that converts subset of hand landmarks to a bounding box
// NormalizedRect. The rotation angle of the bounding box is computed based on
// 1) the wrist joint and 2) the average of PIP joints of index finger, middle
// finger and ring finger. After rotation, the vector from the wrist to the mean
// of PIP joints is expected to be vertical with wrist at the bottom and the
// mean of PIP joints at the top.
class HandRectFromLandmarksStage : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc) {
    cc->Inputs().Tag(kNormalizedLandmarksTag).Set<NormalizedLandmarkList>();
    cc->Inputs().Tag(kImageSizeTag).Set<std::pair<int, int>>();
    cc->Outputs().Tag(kNormRectTag).Set<NormalizedRect>();
    return absl::OkStatus();
  }

  absl::Status Open(CalculatorContext* cc) override {
    cc->SetOffset(TimestampDiff(0));
    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) override {
    if (cc->Inputs().Tag(kNormalizedLandmarksTag).IsEmpty()) {
      return absl::OkStatus();
    }
    RET_CHECK(!cc->Inputs().Tag(kImageSizeTag).IsEmpty());

    std::pair<int, int> image_size =
        cc->Inputs().Tag(kImageSizeTag).Get<std::pair<int, int>>();
    const auto& input_landmarks =
        cc->Inputs().Tag(kNormalizedLandmarksTag).Get<NormalizedLandmarkList>();
    
    auto output_rect = absl::make_unique<NormalizedRect>();
    MP_RETURN_IF_ERROR(
        MakeHandRect(input_landmarks, image_size, output_rect.get()));

    cc->Outputs()
        .Tag(kNormRectTag)
        .Add(output_rect.release(), cc->InputTimestamp());

    return absl::OkStatus();
  }

 private:
};
REGISTER_CALCULATOR(HandRectFromLandmarksStage);

}  // namespace mediapipe_v01013_based
