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

#include "mediapipe/calculators/tensor/inference_calculator_core.h"
#include "mediapipe/calculators/tensor/inference_interpreter_delegate_runner_new.h"
#include "mediapipe/framework/deps/status_macros.h"
#include "absl/log/log.h"
#include "absl/strings/str_format.h"
#include "mediapipe/calculators/tensor/inference_calculator_utils.h"
#include "mediapipe/util/tflite/cpu_op_resolver.h"
#include "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h"

namespace mediapipe {
namespace api2 {

/// a tensorflow interpreter, which is always using the XNNPACK delegate for CPU inference.
ModelInference::ModelInference(const std::string& model_path, int32_t XNNPackDelegate_threads) {

  // load the model
  auto default_resources = CreateDefaultResources();
  auto model_packet_status = TfLiteModelLoader::LoadFromPath(*default_resources, model_path, true);
  if (!model_packet_status.ok()) {
    ABSL_LOG(ERROR) << "failed to load model from path: " << model_path;
    throw std::runtime_error(model_packet_status.status().ToString());
  }
  auto model_packet = model_packet_status.value();
  ABSL_CHECK(!model_packet.IsEmpty());
  ABSL_LOG(INFO) << absl::StrFormat(
    "successfully loaded model from path: %s. Model size: %ld bytes",
    model_path, model_packet.Get()->allocation()->bytes());

  // use the mediapipe default CPU ops resolver
  auto op_resolver = std::make_unique<mediapipe::CpuOpResolver>();

  // use the XNNPACK delegate, which will use the requested number of threads
  auto xnnpack_opts = TfLiteXNNPackDelegateOptionsDefault();
  xnnpack_opts.num_threads = XNNPackDelegate_threads;
  auto delegate = TfLiteDelegatePtr(TfLiteXNNPackDelegateCreate(&xnnpack_opts), &TfLiteXNNPackDelegateDelete);

  // create a runner for the interpeter and delegate
  auto runner_construction_status = CreateInferenceInterpreterDelegateRunner(
      model_packet,
      PacketAdopting<tflite::OpResolver>(std::move(op_resolver)),
      std::move(delegate),
      &InferenceCalculatorOptions().input_output_config());
  if (!runner_construction_status.ok()) {
    ABSL_LOG(ERROR) << "Failed to create inference runner.";
    throw std::runtime_error(runner_construction_status.status().ToString());
  }
  inference_runner_ = std::move(runner_construction_status.value());
}

absl::StatusOr<std::vector<Tensor>> ModelInference::Process(const TensorSpan& tensor_span) {
  return inference_runner_->Run(tensor_span);
}

}  // namespace api2
}  // namespace mediapipe



