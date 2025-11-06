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

#ifndef MEDIAPIPE_CALCULATORS_TENSOR_TENSORS_TO_LANDMARKS_CALCULATOR_CORE_H_
#define MEDIAPIPE_CALCULATORS_TENSOR_TENSORS_TO_LANDMARKS_CALCULATOR_CORE_H_

#include "mediapipe/calculators/tensor/tensors_to_landmarks_calculator.pb.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/tensor.h"
#include "absl/status/status.h"

namespace mediapipe_v01013_based {
namespace api2 {

class TensorsToLandmarksCore {
 public:
  explicit TensorsToLandmarksCore(int input_image_width,
                                  int input_image_height,
                                  ::mediapipe_v01013_based::TensorsToLandmarksCalculatorOptions::Activation visibility_activation,
                                  ::mediapipe_v01013_based::TensorsToLandmarksCalculatorOptions::Activation presence_activation,
                                  float normalize_z,
                                  int num_landmarks = 21);

  absl::Status OutputTensorsToLandmarks(
      const std::vector<Tensor>& input_tensors,
      NormalizedLandmarkList* output_norm_landmarks = nullptr);

 private:
  int input_image_width_;
  int input_image_height_;
  int num_landmarks_;
  TensorsToLandmarksCalculatorOptions::Activation visibility_activation_; // not used
  TensorsToLandmarksCalculatorOptions::Activation presence_activation_; // not used
  float normalize_z_;
};

}  // namespace api2
}  // namespace mediapipe_v01013_based

#endif  // MEDIAPIPE_CALCULATORS_TENSOR_TENSORS_TO_LANDMARKS_CALCULATOR_CORE_H_
