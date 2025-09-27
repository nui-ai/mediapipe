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

// In TensorFlow Lite (TFLite), op resolving refers to the process of mapping operation names (like Conv2D, Add, etc.) in a TFLite model
// to their actual implementations (kernels) in code. The OpResolver is responsible for registering and providing these implementations
// to the TFLite interpreter, so it knows how to execute each operation in the model. This is especially important for custom or
// non-standard ops that are not built into TFLite by default.
// In the context of TensorFlow Lite and TfLiteCustomOpResolverCalculatorNew, typical providers of custom ops may include:
// mediapipe (for MediaPipe-specific custom ops)
// + tensorflow/lite (for built-in and custom TFLite ops)
// + GPU libraries (such as OpenGL, Metal, or CUDA for GPU-accelerated ops)
// + Third-party libraries (for domain-specific or experimental ops)
// - These libraries implement the actual kernels and registration logic for the ops used in TFLite models.
// so this calculator basically configures tflite for being able to use the necessary kernels which our models use.

#include <memory>

#include "mediapipe/calculators/tflite/tflite_custom_op_resolver_calculator.pb.h"
#include "mediapipe/framework/api2/packet.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/util/tflite/cpu_op_resolver.h"
#include "mediapipe/util/tflite/op_resolver.h"
#include "tensorflow/lite/core/api/op_resolver.h"

namespace mediapipe {

namespace {
constexpr char kOpResolverTag[] = "OP_RESOLVER";
}  // namespace

// This calculator creates a custom op resolver as a side packet that can be
// used in TfLiteInferenceCalculator. Current custom op resolver supports the
// following custom op on CPU and GPU:
//   Convolution2DTransposeBias
//   MaxPoolArgmax
//   MaxUnpooling
//
// Usage examples:
//
// For using with TfliteInferenceCalculator:
// node {
//   calculator: "TfLiteCustomOpResolverCalculatorNew"
//   output_side_packet: "op_resolver"
//   node_options: {
//     [type.googleapis.com/mediapipe.TfLiteCustomOpResolverCalculatorOptions] {
//       use_gpu: true
//     }
//   }
// }
//
// For using with InferenceCalculatorNew:
// node {
//   calculator: "TfLiteCustomOpResolverCalculatorNew"
//   output_side_packet: "OP_RESOLVER:op_resolver"
//   node_options: {
//     [type.googleapis.com/mediapipe.TfLiteCustomOpResolverCalculatorOptions] {
//       use_gpu: true
//     }
//   }
// }
class TfLiteCustomOpResolverCalculatorNew : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc) {
    RET_CHECK(cc->OutputSidePackets().HasTag(kOpResolverTag));
    cc->OutputSidePackets().Tag(kOpResolverTag).Set<tflite::OpResolver>();
    return absl::OkStatus();
  }

  // sets a cpu operations resolver for tflite always,
  // through CpuOpResolver/MediaPipe_RegisterTfLiteOpResolver,
  // which provides implementations for some math operations
  // for tflite to use.
  absl::Status Open(CalculatorContext* cc) override {
    cc->SetOffset(TimestampDiff(0));

    std::unique_ptr<tflite::ops::builtin::BuiltinOpResolver> op_resolver;
    op_resolver = absl::make_unique<mediapipe::CpuOpResolver>();

    RET_CHECK(cc->OutputSidePackets().HasTag(kOpResolverTag));
    cc->OutputSidePackets()
        .Tag(kOpResolverTag)
        .Set(mediapipe::api2::PacketAdopting<tflite::OpResolver>(
            std::move(op_resolver)));
    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) override {
    return absl::OkStatus();
  }
};
REGISTER_CALCULATOR(TfLiteCustomOpResolverCalculatorNew);

}  // namespace mediapipe
