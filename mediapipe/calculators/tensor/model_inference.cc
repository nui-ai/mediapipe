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

#include "mediapipe/calculators/tensor/model_inference.h"
#include "mediapipe/calculators/tensor/inference_interpreter_delegate_runner_new.h"
#include "absl/log/log.h"
#include "absl/strings/str_format.h"
#include "mediapipe/calculators/tensor/inference_calculator_utils.h"
#include "mediapipe/util/tflite/cpu_op_resolver.h"
#include "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h"

namespace mediapipe_v01013_based {
namespace api2 {

/// a tensorflow interpreter, which is always using the XNNPACK delegate for CPU inference.
/// see https://github.com/nui-ai/tflite-analysis/blob/8e6f4e7b4211dc78fa4c6d9c0f77aa783fdf669a/readme.md
/// as a starting point for performance analysis of any tflite model used.
ModelInference::ModelInference(const std::string& model_path, int32_t XNNPackDelegate_threads) {

  // load the model
  auto default_resources = CreateDefaultResources();
  auto model_packet_status = TfLiteModelLoader::LoadFromPath(*default_resources, model_path, false);
  if (!model_packet_status.ok()) {
    ABSL_LOG(ERROR) << "failed to load model from path: " << model_path;
    throw std::runtime_error(model_packet_status.status().ToString());
  }
  auto model_packet = model_packet_status.value();
  ABSL_CHECK(!model_packet.IsEmpty());
  ABSL_LOG(INFO) << absl::StrFormat(
    "successfully loaded model from path: %s. Model size: %ld bytes",
    model_path, model_packet.Get()->allocation()->bytes());

  // set the tflite interpreter to use the mediapipe default CPU ops resolver,
  // for any ops which are not claimed by our XNNPACK delegate, if any.
  // as seen in XNNPack logging, juxtapose with https://github.com/nui-ai/tflite-analysis,
  // XNNPack satisfies all ops which are included in our two model graphs:
  // the landmarks inference model has 165 operators, and we get in the log ―
  // "Replacing 165 out of 165 node(s) with delegate (TfLiteXNNPackDelegate) node".
  // so as long as that holds concurrency at the interpreter level is void for us.
  auto op_resolver = std::make_unique<mediapipe_v01013_based::CpuOpResolver>();

  // use the XNNPACK delegate, which will use the requested number of threads, but this is a little confusing
  // as we set the number of threads argument on both the XNNPack delegate here, and on the tflite interpreter
  // object later below, whereas the semantics of this argument can be different between tflite and XNNPack:
  // https://chatgpt.com/s/t_690f7be2ebe48191ac0fb8bbd27a218b.
  // last I recall tflite will only use concurrency if there are graph partitions, but in our case XNNPack
  // owns all graph operations of our models so the tflite interpreter should not use threads,
  // only the XNNPack delegate should or may, if helpful.
  auto xnnpack_opts = TfLiteXNNPackDelegateOptionsDefault();
  xnnpack_opts.num_threads = -1;
  auto delegate = TfLiteDelegatePtr(TfLiteXNNPackDelegateCreate(&xnnpack_opts), &TfLiteXNNPackDelegateDelete);

  /*
   * creates a mediapipe wrapper class called a runner, for the interpeter and delegate,
   * cpu only implementation for now, providing coverage for hardware well supported by XNNPACK.
   *
   * for the meaning of the interpreter_num_threads argument c.f. https://chatgpt.com/s/t_690f234671f481918a7c9ea096134dd5.
   * fiddling the number of threads instead of using the default seems to provide no speedup at all,
   * maybe it doesn't really take effect at our current versions of XNNPACK and TFLITE, or our tflite
   * graphs of the landmarks inference and palm detection modles don't benefit from parallelization.
   *
   * other paths to inference performance:
   *
   *   • pinning the inference steps to a processor as a way of forcing model weights (and XNNPACK/tflite code)
   *     cache residency higher may help, which is of course platform specific.
   *
   *   • enabling XNNPACK weights caching may slightly help with cache persistence as well,
   *     or it can just be marginal for that (https://chatgpt.com/c/68f05fb1-a444-8328-b1c0-53b1e57a99f1).
   *     it's really not any complicated code work to enable it, lest the unexpected.
   *
   *   • as evident in its logging XNNPACK is aware of cache sizes and may or may not adapt to them
   *     in some ways which should be read from its code prior to any tinkering outside of XNNPACK.
   *   • you may use our tflite-analysis code (https://github.com/nui-ai/tflite-analysis)
   *     to juxtapose model weights total memory size with the machine's CPU cache levels
   *     sharing architecture and sizes to estimate impact.
   *
   *   • performance (P) cores may or may not yield faster elapsed time than efficiency (E) cores on Intel x86-64
   *     Performance Hybrid Architecture, which most 12th Gen forward seem to be, and parallel to this actual processor
   *     speeds over time are on Intel x86-64 aggressively over-managed by the OS->Hardware and hardware-only throttling
   *     and cooling strategies which are partly user-configurable and partly version dependent.
   *     we have code recording processor speeds from python in https: https://github.com/nui-ai/core.
   *
   *   • remember that subscription into multi-threading typically effects overall performance in different ways
   *     in different scenarios as per the overall workload not just a single layer's concurrency so never optimize
   *     in a way which prevents a different optimization for other workloads ― keep all concurrency levels fully
   *     and recursively parameterizeable for flexible switching across workload scenarios and machine differences.
   *
   *   • as elsewhere noted, all else equal GPU inference flow can be faster than CPU even for real-time inference,
   *     which is aptly exemplified in the official browser demo as modern browser versions pass images from camera to GPU
   *     associated memory such that the classical GPU transfer time becomes a non-issue. an equivalent fast route from camera
   *     to inference on GPU can be tailored in OS and driver dependent ways (i.e. the browser driving of webcams is smarter
   *     in this than e.g. default use of the v4l2 UVC driver).
   *
   *     for a delineation refer to:
   *       https://chatgpt.com/s/t_690269a8c35481918dbe2dad7df757c8,
   *       https://chatgpt.com/s/t_6902657e23408191b6d9d368dc904205,
   *
   *     it should be noted that browser passing to GPU may be faster but it depends also on the GPU model, host memory
   *     speeds, and the browser api for camera video acquisition may not provide other needed affordances like frame
   *     rate stabilizing ones, and all other camera controls like gain, brightness, focus settings etc. to the same
   *     levels or by the same ways that a driver like v4l2 poorly does up to a webcam's effort to comply
   *     with them.
   */
  auto options = InferenceCalculatorOptions();
  auto runner_construction_status = CreateInferenceInterpreterDelegateRunner(
      model_packet,
      PacketAdopting<tflite::OpResolver>(std::move(op_resolver)),
      std::move(delegate),
      &options.input_output_config(),
      -1);
  if (!runner_construction_status.ok()) {
    ABSL_LOG(ERROR) << "failed to create the mediapipe inference runner object";
    throw std::runtime_error(runner_construction_status.status().ToString());
  }
  inference_runner_ = std::move(runner_construction_status.value());
}

absl::StatusOr<std::vector<Tensor>> ModelInference::Process(const TensorSpan& tensor_span) const {
  return inference_runner_->Run(tensor_span);}

}  // namespace api2
}  // namespace mediapipe_v01013_based



