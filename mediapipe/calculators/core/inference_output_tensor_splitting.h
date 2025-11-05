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

#ifndef MEDIAPIPE_CALCULATORS_CORE_INFERENCE_OUTPUT_TENSOR_SPLITTING_H_
#define MEDIAPIPE_CALCULATORS_CORE_INFERENCE_OUTPUT_TENSOR_SPLITTING_H_

#include <cstdint>
#include <type_traits>
#include <utility>
#include <vector>
#include <memory>
#include <algorithm>

#include "mediapipe/calculators/core/split_vector_calculator.pb.h"
#include "mediapipe/calculators/core/split_vector_calculator_core.h"
#include "mediapipe/framework/port/status.h"

namespace mediapipe_v01013_based {

// A small façade over split_vector_calculator_core.* that encapsulates
// initialization and processing logic for splitting vectors of inference
// outputs (or any T) based on options ranges. It is template-parameterized by
// element type T and whether elements are moved or copied.
// This class is intentionally unaware of CalculatorContext; callers provide and
// receive plain C++ data structures.
template <typename T, bool move_elements>
class InferenceOutputTensorSplitting {
 public:
  InferenceOutputTensorSplitting() = default;

  // Construct and initialize internal state from options.
  explicit InferenceOutputTensorSplitting(
      const ::mediapipe_v01013_based::SplitVectorCalculatorOptions& options) {
    // Initialize derived configuration from options (no CalculatorContext).
    element_only_ = options.element_only();
    combine_outputs_ = options.combine_outputs();

    max_range_end_ = -1;
    total_elements_ = 0;
    ranges_.reserve(options.ranges_size());
    for (const auto& range : options.ranges()) {
      ranges_.push_back({range.begin(), range.end()});
      max_range_end_ = std::max(max_range_end_, range.end());
      total_elements_ += range.end() - range.begin();
    }
  }

  // Run for copyable elements path. Input is const reference; outputs are
  // produced according to the configured options.
  absl::Status Run(
      const std::vector<T>& input,
      std::vector<std::unique_ptr<std::vector<T>>>* output_vectors,
      std::vector<T>* output_elements,
      std::unique_ptr<std::vector<T>>* combined_output) const {
    return ::mediapipe_v01013_based::ProcessCopyableElements<T>(
        input, ranges_, max_range_end_, total_elements_, element_only_,
        combine_outputs_, output_vectors, output_elements, combined_output);
  }

  // Run for movable elements path. Input is consumed/moved from.
  absl::Status Run(
      std::unique_ptr<std::vector<T>>* input_vector,
      std::vector<std::unique_ptr<std::vector<T>>>* output_vectors,
      std::vector<T>* output_elements,
      std::unique_ptr<std::vector<T>>* combined_output) const {
    return ::mediapipe_v01013_based::ProcessMovableElements<T>(
        input_vector, ranges_, max_range_end_, total_elements_, element_only_,
        combine_outputs_, output_vectors, output_elements, combined_output);
  }

  // Expose configuration to the calculator for packaging results.
  bool element_only() const { return element_only_; }
  bool combine_outputs() const { return combine_outputs_; }
  int range_count() const { return static_cast<int>(ranges_.size()); }

  // Static helper so callers don't need to include or call core directly.
  static absl::Status CheckRangesDontOverlap(
      const ::mediapipe_v01013_based::SplitVectorCalculatorOptions& options) {
    return ::mediapipe_v01013_based::CheckRangesDontOverlap(options);
  }

 private:
  std::vector<std::pair<int32_t, int32_t>> ranges_;
  int32_t max_range_end_ = -1;
  int32_t total_elements_ = 0;
  bool element_only_ = false;
  bool combine_outputs_ = false;
};

}  // namespace mediapipe_v01013_based

#endif  // MEDIAPIPE_CALCULATORS_CORE_INFERENCE_OUTPUT_TENSOR_SPLITTING_H_
