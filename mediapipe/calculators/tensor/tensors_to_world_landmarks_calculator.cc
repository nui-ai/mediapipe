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

#include "mediapipe/calculators/tensor/tensors_to_landmarks_calculator.pb.h"
#include "mediapipe/calculators/tensor/tensors_to_world_landmarks_calculator_core.h"
#include "mediapipe/framework/api2/node.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/port/ret_check.h"
#include <memory>

namespace mediapipe_v01013_based {
namespace api2 {

// A calculator for converting Tensors from regression models into landmarks.
// Note that if the landmarks in the tensor has more than 5 dimensions, only the
// first 5 dimensions will be converted to [x,y,z, visibility, presence]. The
// latter two fields may also stay unset if such attributes are not supported in
// the model.
//
// Input:
//  TENSORS - Vector of Tensors of type kFloat32. Only the first tensor will be
//  used. The size of the values must be (num_dimension x num_landmarks).
//
//  FLIP_HORIZONTALLY (optional): Whether to flip landmarks horizontally or
//  not. Overrides corresponding side packet and/or field in the calculator
//  options.
//
//  FLIP_VERTICALLY (optional): Whether to flip landmarks vertically or not.
//  Overrides corresponding side packet and/or field in the calculator options.
//
// Input side packet:
//   FLIP_HORIZONTALLY (optional): Whether to flip landmarks horizontally or
//   not. Overrides the corresponding field in the calculator options.
//
//   FLIP_VERTICALLY (optional): Whether to flip landmarks vertically or not.
//   Overrides the corresponding field in the calculator options.
//
// Output:
//  LANDMARKS(optional) - Result MediaPipe landmarks.
//  NORM_LANDMARKS(optional) - Result MediaPipe normalized landmarks.
//
// Notes:
//   To output normalized landmarks, user must provide the original input image
//   size to the model using calculator option input_image_width and
//   input_image_height.
// Usage example:
// node {
//   calculator: "TensorsToLandmarksCalculator"
//   input_stream: "TENSORS:landmark_tensors"
//   output_stream: "LANDMARKS:landmarks"
//   output_stream: "NORM_LANDMARKS:landmarks"
//   options: {
//     [mediapipe.TensorsToLandmarksCalculatorOptions.ext] {
//       num_landmarks: 21
//
//       input_image_width: 256
//       input_image_height: 256
//     }
//   }
// }
class ExtractWorldLandmarks : public Node {
 public:
  static constexpr Input<std::vector<Tensor>> kInTensors{"TENSORS"};
  static constexpr Output<LandmarkList>::Optional kOutLandmarkList{"LANDMARKS"};
  static constexpr Output<NormalizedLandmarkList>::Optional
      kOutNormalizedLandmarkList{"NORM_LANDMARKS"};
  MEDIAPIPE_NODE_CONTRACT(kInTensors, kOutLandmarkList, kOutNormalizedLandmarkList);

  absl::Status Open(CalculatorContext* cc) override;
  absl::Status Process(CalculatorContext* cc) override;

 private:
  absl::Status LoadOptions(CalculatorContext* cc);
  std::unique_ptr<TensorsToWorldLandmarksCore> core_;
  ::mediapipe_v01013_based::TensorsToLandmarksCalculatorOptions options_;
};
MEDIAPIPE_REGISTER_NODE(ExtractWorldLandmarks);

absl::Status ExtractWorldLandmarks::Open(CalculatorContext* cc) {
  MP_RETURN_IF_ERROR(LoadOptions(cc));
  // Instantiate the core with input image size and option-derived parameters.
  core_ = std::make_unique<TensorsToWorldLandmarksCore>(options_.visibility_activation(), options_.presence_activation());
  return absl::OkStatus();
}

absl::Status ExtractWorldLandmarks::Process(CalculatorContext* cc) {
  if (kInTensors(cc).IsEmpty()) {
    return absl::OkStatus();
  }

  const auto& input_tensors = *kInTensors(cc);

  LandmarkList output_landmarks;
  NormalizedLandmarkList output_norm_landmarks;

  NormalizedLandmarkList* norm_landmarks_ptr =
      kOutNormalizedLandmarkList(cc).IsConnected() ? &output_norm_landmarks : nullptr;

  MP_RETURN_IF_ERROR(core_->TensorsToWorldLandmarks(input_tensors, &output_landmarks));

  // Output normalized landmarks if required.
  if (kOutNormalizedLandmarkList(cc).IsConnected()) {
    kOutNormalizedLandmarkList(cc).Send(std::move(output_norm_landmarks));
  }

  // Output absolute landmarks.
  if (kOutLandmarkList(cc).IsConnected()) {
    kOutLandmarkList(cc).Send(std::move(output_landmarks));
  }

  return absl::OkStatus();
}

absl::Status ExtractWorldLandmarks::LoadOptions(CalculatorContext* cc) {
  // Get calculator options specified in the graph.
  options_ = cc->Options<::mediapipe_v01013_based::TensorsToLandmarksCalculatorOptions>();
  // num_landmarks is not required anymore; default of core is 21.
  return absl::OkStatus();
}
}  // namespace api2
}  // namespace mediapipe_v01013_based
