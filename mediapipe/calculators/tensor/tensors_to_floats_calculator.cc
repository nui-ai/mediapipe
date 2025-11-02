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

#include "mediapipe/calculators/tensor/tensors_to_floats_calculator.pb.h"
#include "mediapipe/framework/api2/node.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/calculators/tensor/tensors_to_floats_calculator_core.h"

namespace mediapipe_v01013_based {

namespace {

inline float Sigmoid(float value) { return 1.0f / (1.0f + std::exp(-value)); }

}  // namespace

// A calculator for converting Tensors to to a float or a float vector.
//
// Input:
//  TENSORS - Vector of Tensors of type kFloat32. Only the first
//            tensor will be used.
// Output:
//  FLOAT(optional) - Converted single float number.
//  FLOATS(optional) - Converted float vector.
//
// Notes: To output FLOAT stream, the input tensor must have size 1, e.g.
//        only 1 float number in the tensor.
//
// Usage example:
// node {
//   calculator: "TensorsToFloatsCalculator"
//   input_stream: "TENSORS:tensors"
//   output_stream: "FLOATS:floats"
// }
namespace api2 {
class ExtractHandPresence : public Node {
 public:
  static constexpr Input<std::vector<Tensor>> kInTensors{"TENSORS"};
  static constexpr Output<float>::Optional kOutFloat{"FLOAT"};
  static constexpr Output<std::vector<float>>::Optional kOutFloats{"FLOATS"};
  MEDIAPIPE_NODE_INTERFACE(TensorsToFloatsCalculator, kInTensors, kOutFloat,
                           kOutFloats);

  static absl::Status UpdateContract(CalculatorContract* cc);
  absl::Status Open(CalculatorContext* cc) final;
  absl::Status Process(CalculatorContext* cc) final;

 private:
  ::mediapipe_v01013_based::TensorsToFloatsCalculatorOptions options_;
};
MEDIAPIPE_REGISTER_NODE(ExtractHandPresence);

absl::Status ExtractHandPresence::UpdateContract(CalculatorContract* cc) {
  // Only exactly a single output allowed.
  RET_CHECK(kOutFloat(cc).IsConnected() ^ kOutFloats(cc).IsConnected());
  return absl::OkStatus();
}

absl::Status ExtractHandPresence::Open(CalculatorContext* cc) {
  options_ = cc->Options<::mediapipe_v01013_based::TensorsToFloatsCalculatorOptions>();
  return tensors_to_floats_calculator_core::Open(options_);
}

absl::Status ExtractHandPresence::Process(CalculatorContext* cc) {
  const auto& input_tensors = *kInTensors(cc);

  auto result = tensors_to_floats_calculator_core::Process(input_tensors, options_);

  if (!result.status.ok()) {
    return result.status;
  }

  if (kOutFloat(cc).IsConnected()) {
    // TODO: Could add an index in the option to specifiy returning
    // one value of a float array.
    RET_CHECK_EQ(result.num_values, 1);
    kOutFloat(cc).Send(result.output_floats->at(0));
  } else {
    kOutFloats(cc).Send(std::move(result.output_floats));
  }
  return absl::OkStatus();
}
}  // namespace api2
}  // namespace mediapipe_v01013_based
