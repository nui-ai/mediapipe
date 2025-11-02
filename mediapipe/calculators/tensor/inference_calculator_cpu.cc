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

#include "inference_feedback_manager.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/time/time.h"
#include "mediapipe/calculators/tensor/inference_calculator.h"
#include "mediapipe/calculators/tensor/inference_calculator_utils.h"
#include "mediapipe/calculators/tensor/inference_interpreter_delegate_runner.h"
#include "mediapipe/calculators/tensor/inference_runner.h"
#include "mediapipe/calculators/tensor/tensor_span.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status_macros.h"
#include "mediapipe/util/tflite/cpu_op_resolver.h"
#if defined(MEDIAPIPE_ANDROID)
#include "tensorflow/lite/delegates/nnapi/nnapi_delegate.h"
#endif  // ANDROID
#include "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h"

namespace mediapipe_v01013_based {
namespace api2 {

class InferenceCalculatorCpuImpl
    : public InferenceCalculatorNodeImpl<InferenceCalculatorCpu,
                                         InferenceCalculatorCpuImpl> {
 public:
  static absl::Status UpdateContract(CalculatorContract* cc);

  absl::Status Open(CalculatorContext* cc) override;
  absl::Status Close(CalculatorContext* cc) override;

 private:
  absl::StatusOr<std::vector<Tensor>> Process(
      CalculatorContext* cc, const TensorSpan& tensor_span) override;

  std::unique_ptr<InferenceRunner> inference_runner_;
  // std::unique_ptr<InferenceIoMapper> io_mapper_;
  std::vector<int> input_tensor_indices_;
  std::vector<int> output_tensor_indices_;

};

absl::Status InferenceCalculatorCpuImpl::UpdateContract(
    CalculatorContract* cc) {
  const auto& options = cc->Options<mediapipe_v01013_based::InferenceCalculatorOptions>();
  RET_CHECK(!options.model_path().empty() ^ kSideInModel(cc).IsConnected() ^ kSideInModelPath(cc).IsConnected())
      << "One of: model path in options, MODEL side packet, or MODEL_PATH side packet is required.";

  MP_RETURN_IF_ERROR(TensorContractCheck(cc));

  return absl::OkStatus();
}

/// Open method no longer really uses its CalculatorContext argument.
absl::Status InferenceCalculatorCpuImpl::Open(CalculatorContext* cc) {
  ABSL_LOG(INFO) << "starting InferenceCalculatorCpuImpl";

  MP_ASSIGN_OR_RETURN(auto model_packet, GetModelAsPacket(cc));
  auto op_resolver = std::make_unique<mediapipe_v01013_based::CpuOpResolver>();

  auto xnnpack_opts = TfLiteXNNPackDelegateOptionsDefault();
  xnnpack_opts.num_threads = 1;
  auto delegate = TfLiteDelegatePtr(TfLiteXNNPackDelegateCreate(&xnnpack_opts),&TfLiteXNNPackDelegateDelete);

  tflite::InterpreterBuilder interpreter_builder(*model_packet.Get(), *op_resolver);
  interpreter_builder.AddDelegate(delegate.get());
  interpreter_builder.SetNumThreads(-1);

  auto options = InferenceCalculatorOptions();

  MP_ASSIGN_OR_RETURN(inference_runner_, CreateInferenceInterpreterDelegateRunner(
    model_packet,
    PacketAdopting<tflite::OpResolver>(std::move(op_resolver)),
    std::move(delegate),
    -1,
    &options.input_output_config(),
    false));

  // Update IoMapper with input/output tensor names from the TfLite model.
  io_mapper_ = std::make_unique<InferenceIoMapper>();
  return io_mapper_->UpdateIoMap(options.input_output_config(), inference_runner_->GetInputOutputTensorNames());
}

/// does not really use its CalculatorContext argument in the cascade of called functions
/// (only one of the chain of functions called from it merely used it only for performance tracing before)
absl::StatusOr<std::vector<Tensor>> InferenceCalculatorCpuImpl::Process(
    CalculatorContext* cc, const TensorSpan& tensor_span) {

  MP_ASSIGN_OR_RETURN(std::vector<Tensor> output_tensors, inference_runner_->Run(cc, tensor_span));

  return output_tensors;
}

absl::Status InferenceCalculatorCpuImpl::Close(CalculatorContext* cc) {
  inference_runner_ = nullptr;
  return absl::OkStatus();
}

}  // namespace api2
}  // namespace mediapipe_v01013_based
