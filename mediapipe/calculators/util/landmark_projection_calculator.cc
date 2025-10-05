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

#include "mediapipe/calculators/util/landmark_projection_calculator.h"

#include <array>
#include <cmath>
#include <functional>
#include <utility>

#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "mediapipe/calculators/tensor/image_to_tensor_utils.h"
#include "mediapipe/calculators/util/landmark_projection_calculator.pb.h"
#include "mediapipe/calculators/util/landmark_projection_calculator_core.h"
#include "mediapipe/framework/api3/calculator.h"
#include "mediapipe/framework/api3/calculator_context.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/rect.pb.h"

namespace mediapipe::api3 {

using ::mediapipe::NormalizedRect;

class LandmarkProjectionNodeImpl
    : public Calculator<LandmarkProjectionNode, LandmarkProjectionNodeImpl> {
 public:
  absl::Status Process(CalculatorContext<LandmarkProjectionNode>& cc) override {
    const bool has_rect = cc.norm_rect.IsConnected();
    const bool has_image_dims = cc.image_dimensions.IsConnected();

    std::function<void(const NormalizedLandmark&, NormalizedLandmark*)>
        project_fn;
    if (has_rect && !has_image_dims) {
      if (!cc.norm_rect) {
        return absl::OkStatus();
      }
      const NormalizedRect& input_rect = cc.norm_rect.GetOrDie();
      const LandmarkProjectionCalculatorOptions& options = cc.options.Get();
      project_fn = CreateProjectionFunction(&input_rect, &options, nullptr,
                                           nullptr);
    } else if (has_rect && has_image_dims) {
      if (!cc.norm_rect || !cc.image_dimensions) {
        return absl::OkStatus();
      }
      const NormalizedRect& input_rect = cc.norm_rect.GetOrDie();
      const LandmarkProjectionCalculatorOptions& options = cc.options.Get();
      const std::pair<int, int>& image_dimensions =
          cc.image_dimensions.GetOrDie();
      project_fn = CreateProjectionFunction(&input_rect, &options,
                                           &image_dimensions, nullptr);
    } else if (cc.projection_matrix.IsConnected()) {
      if (!cc.projection_matrix) {
        return absl::OkStatus();
      }
      const std::array<float, 16>& matrix = cc.projection_matrix.GetOrDie();
      project_fn = CreateProjectionFunction(nullptr, nullptr, nullptr, &matrix);
    } else {
      return absl::InternalError("Either rect or matrix must be specified.");
    }

    const int count = cc.input_landmarks.Count();
    // Number of inputs and outputs is the same according to the contract.
    for (int i = 0; i < count; ++i) {
      const auto& input = cc.input_landmarks.At(i);
      if (!input) {
        continue;
      }

      const NormalizedLandmarkList& input_landmarks = input.GetOrDie();
      NormalizedLandmarkList output_landmarks;
      ProcessLandmarkList(input_landmarks, project_fn, &output_landmarks);
      cc.output_landmarks.At(i).Send(std::move(output_landmarks));
    }
    return absl::OkStatus();
  }
};

}  // namespace mediapipe::api3
