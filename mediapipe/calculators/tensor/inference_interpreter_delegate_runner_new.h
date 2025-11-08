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

#ifndef MEDIAPIPE_CALCULATORS_TENSOR_INFERENCE_INTERPRETER_DELEGATE_RUNNER_H_
#define MEDIAPIPE_CALCULATORS_TENSOR_INFERENCE_INTERPRETER_DELEGATE_RUNNER_H_

#include <memory>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "mediapipe/calculators/tensor/inference_calculator_utils.h"
#include "mediapipe/calculators/tensor/inference_feedback_manager.h"
#include "mediapipe/calculators/tensor/inference_io_mapper.h"
#include "mediapipe/calculators/tensor/tensor_span.h"
#include "mediapipe/calculators/tensor/tflite_delegate_ptr.h"
#include "mediapipe/framework/api2/packet.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/mediapipe_profiling.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status_macros.h"
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/interpreter_builder.h"
#include "tensorflow/lite/util.h"

#include "absl/status/statusor.h"
#include "mediapipe/calculators/tensor/inference_runner_new.h"
#include "mediapipe/calculators/tensor/tflite_delegate_ptr.h"
#include "mediapipe/framework/api2/packet.h"
#include "mediapipe/util/tflite/tflite_model_loader.h"
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/core/api/op_resolver.h"

namespace mediapipe_v01013_based {

using Interpreter = ::tflite::Interpreter;
using InterpreterBuilder = ::tflite::InterpreterBuilder;

class InferenceInterpreterDelegateRunner : public InferenceRunner {
public:
    InferenceInterpreterDelegateRunner(
        api2::Packet<TfLiteModelPtr> model,
        std::unique_ptr<Interpreter> interpreter, TfLiteDelegatePtr delegate,
        InputOutputTensorNames&& input_output_tensor_names,
        std::unique_ptr<InferenceFeedbackManager> feedback_manager)
        : model_(std::move(model)),
          delegate_(std::move(delegate)),
          interpreter_(std::move(interpreter)),
          input_output_tensor_names_(std::move(input_output_tensor_names)),
          feedback_manager_(std::move(feedback_manager)){}

    absl::StatusOr<std::vector<Tensor>> Run(const TensorSpan &tensor_span) override;

private:
    api2::Packet<TfLiteModelPtr> model_;
    TfLiteDelegatePtr delegate_;
    std::unique_ptr<Interpreter> interpreter_;
    InputOutputTensorNames input_output_tensor_names_;
    std::unique_ptr<InferenceFeedbackManager> feedback_manager_;

    // Copy output tensors from the interpreter always, because zero copy may cause a stability issue,
    // as seen in inline code comments from the mediapipe team around the use of this variable.
    bool enable_zero_copy_tensor_io_ = false;
};


// Creates inference runner which run inference using newly initialized
// interpreter and provided `delegate`.
//
// `delegate` can be nullptr, in that case newly initialized interpreter will
// use what is available by default.
// `input_output_config` optional config to enable feedback tensors.
//
// `enable_zero_copy_tensor_input` and `enable_zero_copy_tensor_output` enable
// zero copy tensor I/O using TfLite's custom allocator API.
// Note that `enable_zero_copy_tensor_input` requires *all* input tensors to be
// aligned to tflite::kDefaultTensorAlignment bytes.
// `enable_zero_copy_tensor_output` requires that the model has no duplicate
// output tensors (tensors with identical TfLite tensor indices) and no
// passthrough input->output tensors (input and output tensors with identical
// TfLite tensor indices).
absl::StatusOr<std::unique_ptr<InferenceRunner>>
CreateInferenceInterpreterDelegateRunner(
    api2::Packet<TfLiteModelPtr> model,
    api2::Packet<tflite::OpResolver> op_resolver, TfLiteDelegatePtr delegate,
    const mediapipe_v01013_based::InferenceCalculatorOptions::InputOutputConfig* input_output_config = nullptr,
    int interpreter_num_threads = 1);

}  // namespace mediapipe_v01013_based

#endif  // MEDIAPIPE_CALCULATORS_TENSOR_INFERENCE_INTERPRETER_DELEGATE_RUNNER_H_
