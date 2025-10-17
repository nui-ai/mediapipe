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
#include "mediapipe/calculators/tensor/detection_inference_calculator_core.h"
#include "mediapipe/calculators/tensor/inference_calculator.h"
#include "mediapipe/calculators/tensor/inference_calculator_utils.h"
#include "mediapipe/calculators/tensor/inference_interpreter_delegate_runner_new.h"
#include "mediapipe/calculators/tensor/inference_runner.h"
#include "mediapipe/calculators/tensor/tensor_span.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_context.h"
#include "mediapipe/framework/calculator_state.h"
#include "mediapipe/framework/api2/node.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status_macros.h"
#include "mediapipe/util/tflite/cpu_op_resolver.h"
#include "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h"

namespace mediapipe {
namespace api2 {

struct DetectionInferenceCalculatorCpu : public InferenceCalculator {
  static constexpr char kCalculatorName[] = "DetectionInferenceCalculator";
};

/// note that this class still uses a lot of generic inference code of the mediapipe framework
/// (who knows maybe some of it will turn out useful in later expansions).
class DetectionInferenceCalculator : public Node {
    // : public InferenceCalculatorNodeImpl<DetectionInferenceCalculatorCpu,
    //                                      DetectionInferenceCalculator> {
 public:

  static constexpr Input<std::vector<Tensor>>             kInTensors{"TENSORS"};
  static constexpr Output<std::vector<Tensor>>::Optional  kOutTensors{"TENSORS"};
  MEDIAPIPE_NODE_INTERFACE(DetectionInferenceCalculator, kInTensors, kOutTensors);

  absl::Status Open(CalculatorContext* cc) override;
  absl::Status Close(CalculatorContext* cc) override;

 private:
  absl::Status Process(CalculatorContext* cc) override;

  std::unique_ptr<DetectionInferenceCalculatorCore> core_;
  std::vector<int> input_tensor_indices_;
  std::vector<int> output_tensor_indices_;

};

/// Open method no longer really uses its CalculatorContext argument.
absl::Status DetectionInferenceCalculator::Open(CalculatorContext* cc) {
  ABSL_LOG(INFO) << "starting DetectionInferenceCalculator";
  const std::string& model_path = "mediapipe/modules/palm_detection/palm_detection_full.tflite";
  try {
    // io_mapper_ is a member of the InferenceCalculator class
    // io_mapper_ = std::make_unique<InferenceIoMapper>();
    // core_ = std::make_unique<DetectionInferenceCalculatorCore>(model_path, io_mapper_.get());
    core_ = std::make_unique<DetectionInferenceCalculatorCore>(model_path);
  } catch (const std::exception& e) {
    return absl::InternalError(e.what());
  }
  return absl::OkStatus();
}

/// does not really use its CalculatorContext argument in the cascade of called functions
/// (only one of the chain of functions called from it merely used it only for performance tracing before)
absl::Status DetectionInferenceCalculator::Process(CalculatorContext* cc) {

  if (kInTensors(cc).IsEmpty()) return absl::OkStatus();

  const auto& in_tensors = *kInTensors(cc);
  auto tensor_span = MakeTensorSpan(in_tensors);

  auto out_tensors = core_->Process(tensor_span);

  if (kOutTensors(cc).IsConnected()) {
    kOutTensors(cc).Send(std::move(out_tensors.value()));
  }
  return absl::OkStatus();

}

absl::Status DetectionInferenceCalculator::Close(CalculatorContext* cc) {
  core_.reset();
  return absl::OkStatus();
}

MEDIAPIPE_REGISTER_NODE(DetectionInferenceCalculator);
}  // namespace api2
}  // namespace mediapipe
