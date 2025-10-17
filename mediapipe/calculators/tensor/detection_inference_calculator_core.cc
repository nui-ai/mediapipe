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


#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "mediapipe/calculators/tensor/detection_inference_calculator_core.h"
#include "mediapipe/calculators/tensor/inference_interpreter_delegate_runner_new.h"
#include "mediapipe/framework/deps/status_macros.h"
#include "absl/log/log.h"
#include "absl/strings/str_format.h"
#include "mediapipe/calculators/tensor/inference_calculator_utils.h"
#include "mediapipe/util/tflite/cpu_op_resolver.h"
#include "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h"

namespace mediapipe {
namespace api2 {


DetectionInferenceCalculatorCore::DetectionInferenceCalculatorCore(const std::string& model_path) {
  auto default_resources = CreateDefaultResources();
  auto model_packet_status = TfLiteModelLoader::LoadFromPath(*default_resources, model_path, false);
  if (!model_packet_status.ok()) {
    ABSL_LOG(ERROR) << "Failed to load model from path: " << model_path;
    throw std::runtime_error(model_packet_status.status().ToString());
  }
  auto model_packet = model_packet_status.value();
  ABSL_CHECK(!model_packet.IsEmpty());
  ABSL_LOG(INFO) << absl::StrFormat(
    "GetModelAsPacket successfully loaded model from path: %s. Model size: %ld bytes",
    model_path, model_packet.Get()->allocation()->bytes());

  auto op_resolver = std::make_unique<mediapipe::CpuOpResolver>();

  auto xnnpack_opts = TfLiteXNNPackDelegateOptionsDefault();
  xnnpack_opts.num_threads = 1;
  auto delegate = TfLiteDelegatePtr(TfLiteXNNPackDelegateCreate(&xnnpack_opts), &TfLiteXNNPackDelegateDelete);

  tflite::InterpreterBuilder interpreter_builder(*model_packet.Get(), *op_resolver);
  interpreter_builder.AddDelegate(delegate.get());
  interpreter_builder.SetNumThreads(-1);

  auto options = InferenceCalculatorOptions();

  auto runner_construction_status = CreateInferenceInterpreterDelegateRunner(
      model_packet,
      PacketAdopting<tflite::OpResolver>(std::move(op_resolver)),
      std::move(delegate),
      -1,
      &options.input_output_config(),
      false);
  if (!runner_construction_status.ok()) {
    ABSL_LOG(ERROR) << "Failed to create inference runner.";
    throw std::runtime_error(runner_construction_status.status().ToString());
  }
  inference_runner_ = std::move(runner_construction_status.value());
}

absl::StatusOr<std::vector<Tensor>> DetectionInferenceCalculatorCore::Process(const TensorSpan& tensor_span) {
  return inference_runner_->Run(tensor_span);
}

}  // namespace api2
}  // namespace mediapipe



