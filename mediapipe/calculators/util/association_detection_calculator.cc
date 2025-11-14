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

#include "mediapipe/calculators/util/association_calculator.h"
#include "mediapipe/framework/calculator_context.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/formats/location.h"
#include "mediapipe/framework/port/rectangle.h"
#include "mediapipe/framework/port/status.h"

namespace hand_tracking_mp_lean {

// A subclass of AssociationCalculator<T> for Detection. Example:
// node {
//   calculator: "AssociationDetectionCalculator"
//   input_stream: "PREV:input_vec_0"
//   input_stream: "input_vec_1"
//   input_stream: "input_vec_2"
//   output_stream: "output_vec"
//   options {
//     [mediapipe.AssociationCalculatorOptions.ext] {
//       min_similarity_threshold: 0.1
//     }
// }
class AssociationDetectionCalculator
    : public AssociationCalculator<::hand_tracking_mp_lean::Detection> {
 public:
  static absl::Status GetContract(CalculatorContract* cc) {
    return AssociationCalculator<::hand_tracking_mp_lean::Detection>::GetContract(cc);
  }

  absl::Status Open(CalculatorContext* cc) override {
    return AssociationCalculator<::hand_tracking_mp_lean::Detection>::Open(cc);
  }

  absl::Status Process(CalculatorContext* cc) override {
    return AssociationCalculator<::hand_tracking_mp_lean::Detection>::Process(cc);
  }

  absl::Status Close(CalculatorContext* cc) override {
    return AssociationCalculator<::hand_tracking_mp_lean::Detection>::Close(cc);
  }

 protected:
  absl::StatusOr<Rectangle_f> GetRectangle(
      const ::hand_tracking_mp_lean::Detection& input) override {
    if (!input.has_location_data()) {
      return absl::InternalError("Missing location_data in Detection");
    }
    const Location location(input.location_data());
    return location.GetRelativeBBox();
  }

  std::pair<bool, int> GetId(const ::hand_tracking_mp_lean::Detection& input) override {
    return {input.has_detection_id(), input.detection_id()};
  }

  void SetId(::hand_tracking_mp_lean::Detection* input, int id) override {
    input->set_detection_id(id);
  }
};

REGISTER_CALCULATOR(AssociationDetectionCalculator);

}  // namespace hand_tracking_mp_lean
