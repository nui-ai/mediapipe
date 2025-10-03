// Copyright 2022 The MediaPipe Authors.
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

#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <fstream>

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
#include "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h"
#include "mediapipe/calculators/tflite/tflite_model_calculator.cc"
#include "mediapipe/calculators/tensor/inference_calculator_xnnpack.h"

std::unique_ptr<tflite::FlatBufferModel> LoadTFLiteModelFromFile(const std::string& model_path) {
    std::ifstream file(model_path, std::ios::binary | std::ios::ate);
    if (!file) {
        ABSL_LOG(ERROR) << "Failed to open model file: " << model_path;
        return nullptr;
    }
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::string buffer(size, '\0');
    if (!file.read(&buffer[0], size)) {
        ABSL_LOG(ERROR) << "Failed to read model file: " << model_path;
        return nullptr;
    }
    auto model = tflite::FlatBufferModel::BuildFromBuffer(buffer.data(), buffer.size());
    if (!model) {
        ABSL_LOG(ERROR) << "Failed to build tflite model from file: " << model_path;
        return nullptr;
    }
    ABSL_LOG(INFO) << "tflite model loaded from file: " << model_path;
    return model;
}

namespace mediapipe {
namespace api2 {

absl::Status InferenceCalculatorXnnpackImpl::UpdateContract(
    CalculatorContract* cc) {
  MP_RETURN_IF_ERROR(TensorContractCheck(cc));

  return absl::OkStatus();
}

absl::Status InferenceCalculatorXnnpackImpl::Open(CalculatorContext* cc) {
  MP_ASSIGN_OR_RETURN(inference_runner_, CreateInferenceRunner(cc));
  return InferenceCalculatorNodeImpl::UpdateIoMapping(
      cc, inference_runner_->GetInputOutputTensorNames());
}

absl::StatusOr<std::vector<Tensor>> InferenceCalculatorXnnpackImpl::Process(
    CalculatorContext* cc, const TensorSpan& tensor_span) {
  MP_ASSIGN_OR_RETURN(std::vector<Tensor> output_tensors,
                      inference_runner_->Run(cc, tensor_span));
  return output_tensors;
}

absl::Status InferenceCalculatorXnnpackImpl::Close(CalculatorContext* cc) {
  inference_runner_ = nullptr;
  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<InferenceRunner>>
InferenceCalculatorXnnpackImpl::CreateInferenceRunner(CalculatorContext* cc) {
  // Load model directly from file path, ignore input packets.
  std::unique_ptr<tflite::FlatBufferModel> raw_model = LoadTFLiteModelFromFile(GetModelPath());
  RET_CHECK(raw_model) << "Failed to load TfLite model from file.";
  TfLiteModelPtr model_ptr = TfLiteModelPtr(
      raw_model.release(), [](tflite::FlatBufferModel* model) {
        delete model;
      });
  auto model_packet = MakePacket<TfLiteModelPtr>(std::move(model_ptr));
  // Get op_resolver from SharedCalculatorState and pass directly.
  auto op_resolver_ptr = mediapipe::SharedCalculatorState::GetOpResolver();
  RET_CHECK(op_resolver_ptr != nullptr) << "OpResolver not set in SharedCalculatorState";
  const auto& calculator_opts =
      cc->Options<mediapipe::InferenceCalculatorOptions>();
  const int interpreter_num_threads = calculator_opts.cpu_num_thread();
  MP_ASSIGN_OR_RETURN(TfLiteDelegatePtr delegate, CreateDelegate(cc));
  return CreateInferenceInterpreterDelegateRunner(
      model_packet, op_resolver_ptr,
      std::move(delegate), interpreter_num_threads,
      &calculator_opts.input_output_config(),
      calculator_opts.delegate().xnnpack().enable_zero_copy_tensor_io());
}

absl::StatusOr<TfLiteDelegatePtr>
InferenceCalculatorXnnpackImpl::CreateDelegate(CalculatorContext* cc) {
  const auto& calculator_opts =
      cc->Options<mediapipe::InferenceCalculatorOptions>();
  auto opts_delegate = calculator_opts.delegate();
  if (!kDelegate(cc).IsEmpty()) {
    const mediapipe::InferenceCalculatorOptions::Delegate&
        input_side_packet_delegate = kDelegate(cc).Get();
    RET_CHECK(
        input_side_packet_delegate.has_xnnpack() ||
        input_side_packet_delegate.delegate_case() ==
            mediapipe::InferenceCalculatorOptions::Delegate::DELEGATE_NOT_SET)
        << "inference_calculator_cpu only supports delegate input side packet "
        << "for TFLite, XNNPack";
    opts_delegate.MergeFrom(input_side_packet_delegate);
  }
  const bool opts_has_delegate =
      calculator_opts.has_delegate() || !kDelegate(cc).IsEmpty();

  auto xnnpack_opts = TfLiteXNNPackDelegateOptionsDefault();
  xnnpack_opts.num_threads =
      GetXnnpackNumThreads(opts_has_delegate, opts_delegate);
  return TfLiteDelegatePtr(TfLiteXNNPackDelegateCreate(&xnnpack_opts),
                           &TfLiteXNNPackDelegateDelete);
}

}  // namespace api2
}  // namespace mediapipe
