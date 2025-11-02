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

#ifndef MEDIAPIPE_CALCULATORS_TENSOR_DETECTION_INFERENCE_CALCULATOR_CORE_H_
#define MEDIAPIPE_CALCULATORS_TENSOR_DETECTION_INFERENCE_CALCULATOR_CORE_H_

#include <memory>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "mediapipe/calculators/tensor/inference_runner_new.h"
#include "mediapipe/calculators/tensor/inference_calculator.h"
#include "mediapipe/calculators/tensor/inference_io_mapper.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/calculators/tensor/tensor_span.h"

namespace mediapipe_v01013_based {
namespace api2 {

class ModelInference {
 public:
  ModelInference(const std::string& model_path, int32_t XNNPackDelegate_threads=-1);
  ~ModelInference() = default;

  absl::StatusOr<std::vector<Tensor>> Process(const TensorSpan& tensor_span) const;

 private:
  std::unique_ptr<InferenceRunner> inference_runner_;
};

}  // namespace api2
}  // namespace mediapipe_v01013_based

#endif  // MEDIAPIPE_CALCULATORS_TENSOR_DETECTION_INFERENCE_CALCULATOR_CORE_H_
