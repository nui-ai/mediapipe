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

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/time/time.h"
#include "mediapipe/calculators/tensor/inference_calculator.h"
#include "mediapipe/calculators/tensor/inference_calculator_utils.h"
#include "mediapipe/calculators/tensor/inference_calculator_core.h"
#include "mediapipe/calculators/tensor/inference_interpreter_delegate_runner_new.h"
#include "mediapipe/calculators/tensor/inference_interpreter_delegate_runner_new.h"
#include "mediapipe/calculators/tensor/inference_runner.h"
#include "mediapipe/calculators/tensor/tensor_span.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_context.h"
#include "mediapipe/framework/calculator_state.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status_macros.h"
#include "mediapipe/util/tflite/cpu_op_resolver.h"
#include "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h"

namespace mediapipe {
namespace api2 {

struct LandmarksInferenceCalculatorCpu : public InferenceCalculator {
  static constexpr char kCalculatorName[] = "LandmarksInferenceCalculator";
};

/// note that this class still uses a lot of generic inference code of the mediapipe framework
/// (who knows maybe some of it will turn out useful in later expansions).
class LandmarksInferenceCalculator
    : public InferenceCalculatorNodeImpl<LandmarksInferenceCalculatorCpu,
                                         LandmarksInferenceCalculator> {
 public:
  static absl::Status UpdateContract(CalculatorContract* cc);

  absl::Status Open(CalculatorContext* cc) override;
  absl::Status Close(CalculatorContext* cc) override;

 private:
  absl::StatusOr<std::vector<Tensor>> Process(
      CalculatorContext* cc, const TensorSpan& tensor_span) override;

  std::unique_ptr<InferenceRunner> inference_runner_;
  std::unique_ptr<ModelInference> core_;
  std::vector<int> input_tensor_indices_;
  std::vector<int> output_tensor_indices_;

};

absl::Status LandmarksInferenceCalculator::UpdateContract(CalculatorContract* cc) {
  MP_RETURN_IF_ERROR(TensorContractCheck(cc));
  return absl::OkStatus();
}

/// Open method no longer really uses its CalculatorContext argument.
absl::Status LandmarksInferenceCalculator::Open(CalculatorContext* cc) {

  const std::string& model_path = "mediapipe/modules/hand_landmark/hand_landmark_full.tflite";
  try {
    core_ = std::make_unique<ModelInference>(model_path);
  } catch (const std::exception& e) {
    return absl::InternalError(e.what());
  }
  return absl::OkStatus();
}

/// does not really use its CalculatorContext argument in the cascade of called functions
/// (only one of the chain of functions called from it merely used it only for performance tracing before)
absl::StatusOr<std::vector<Tensor>> LandmarksInferenceCalculator::Process(
    CalculatorContext* cc, const TensorSpan& tensor_span) {

  MP_ASSIGN_OR_RETURN(std::vector<Tensor> output_tensors, inference_runner_->Run(tensor_span));

  return output_tensors;
}

absl::Status LandmarksInferenceCalculator::Close(CalculatorContext* cc) {
  inference_runner_ = nullptr;
  return absl::OkStatus();
}

}  // namespace api2
}  // namespace mediapipe
