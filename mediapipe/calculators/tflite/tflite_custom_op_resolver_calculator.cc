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
// In the context of TensorFlow Lite and TfLiteCustomOpResolverCalculator, typical providers of custom ops may include:
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

namespace hand_tracking_mp_lean {

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
//   calculator: "TfLiteCustomOpResolverCalculator"
//   output_side_packet: "op_resolver"
//   node_options: {
//     [type.googleapis.com/mediapipe.TfLiteCustomOpResolverCalculatorOptions] {
//       use_gpu: true
//     }
//   }
// }
//
// For using with InferenceCalculator:
// node {
//   calculator: "TfLiteCustomOpResolverCalculator"
//   output_side_packet: "OP_RESOLVER:op_resolver"
//   node_options: {
//     [type.googleapis.com/mediapipe.TfLiteCustomOpResolverCalculatorOptions] {
//       use_gpu: true
//     }
//   }
// }
// this class does not dynamically decide TensorFlow ops resolving at runtime based on model contents or external input. It only chooses between two predefined resolvers (OpResolver for GPU or CpuOpResolver for CPU) based on the use_gpu option in its configuration. The actual set of supported ops is determined by the implementation of these resolver classes, not by anything dynamic in this calculator.
class TfLiteCustomOpResolverCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc) {
    if (cc->OutputSidePackets().HasTag(kOpResolverTag)) {
      cc->OutputSidePackets().Tag(kOpResolverTag).Set<tflite::OpResolver>();
    } else {
      cc->OutputSidePackets()
          .Index(0)
          .Set<tflite::ops::builtin::BuiltinOpResolver>();
    }
    return absl::OkStatus();
  }

  absl::Status Open(CalculatorContext* cc) override {
    cc->SetOffset(TimestampDiff(0));

    const TfLiteCustomOpResolverCalculatorOptions& options =
        cc->Options<TfLiteCustomOpResolverCalculatorOptions>();

    std::unique_ptr<tflite::ops::builtin::BuiltinOpResolver> op_resolver;
    if (options.use_gpu()) {
      op_resolver = absl::make_unique<hand_tracking_mp_lean::OpResolver>();
    } else {
      op_resolver = absl::make_unique<hand_tracking_mp_lean::CpuOpResolver>();
    }

    if (cc->OutputSidePackets().HasTag(kOpResolverTag)) {
      cc->OutputSidePackets()
          .Tag(kOpResolverTag)
          .Set(hand_tracking_mp_lean::api2::PacketAdopting<tflite::OpResolver>(
              std::move(op_resolver)));
    } else {
      cc->OutputSidePackets().Index(0).Set(Adopt(op_resolver.release()));
    }
    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) override {
    return absl::OkStatus();
  }
};
REGISTER_CALCULATOR(TfLiteCustomOpResolverCalculator);

}  // namespace hand_tracking_mp_lean
