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

#include "mediapipe/calculators/util/world_landmark_projection_calculator.h"
#include "mediapipe/calculators/util/world_landmark_projection_calculator_core.h"

#include <cmath>
#include <functional>
#include <utility>

#include "absl/status/status.h"
#include "mediapipe/framework/api3/calculator.h"
#include "mediapipe/framework/api3/calculator_context.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/rect.pb.h"

namespace mediapipe_v01013_based::api3 {

class WorldLandmarkProjectionNodeImpl
    : public Calculator<WorldLandmarkProjectionNode,
                        WorldLandmarkProjectionNodeImpl> {
 public:
  absl::Status Process(
      CalculatorContext<WorldLandmarkProjectionNode>& cc) final {
    // Check that landmarks and rect (if connected) are not empty.
    if (!cc.input_landmarks ||
        (cc.input_rect.IsConnected() && !cc.input_rect)) {
      return absl::OkStatus();
    }

    const auto& in_landmarks = cc.input_landmarks.GetOrDie();

    const NormalizedRect* in_rect = nullptr;
    in_rect = &cc.input_rect.GetOrDie();

    LandmarkList out_landmarks = mediapipe_v01013_based::api3::RotateWorldLandmarks(in_landmarks, in_rect);

    cc.output_landmarks.Send(std::move(out_landmarks));
    return absl::OkStatus();
  }
};

}  // namespace mediapipe_v01013_based::api3
