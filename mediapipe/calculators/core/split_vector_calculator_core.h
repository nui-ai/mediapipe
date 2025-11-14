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

#ifndef MEDIAPIPE_CALCULATORS_CORE_SPLIT_VECTOR_CALCULATOR_CORE_H_
#define MEDIAPIPE_CALCULATORS_CORE_SPLIT_VECTOR_CALCULATOR_CORE_H_

#include <cstdint>
#include <type_traits>
#include <vector>
#include <utility>

#include "mediapipe/calculators/core/split_vector_calculator.pb.h"
#include "mediapipe/framework/port/canonical_errors.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"

namespace hand_tracking_mp_lean {

// Extracts configuration from options and initializes the calculator's state
template <typename T>
absl::Status InitializeSplitVectorCalculator(
    const ::hand_tracking_mp_lean::SplitVectorCalculatorOptions& options,
    std::vector<std::pair<int32_t, int32_t>>* ranges,
    int32_t* max_range_end,
    int32_t* total_elements,
    bool* element_only,
    bool* combine_outputs) {
  *element_only = options.element_only();
  *combine_outputs = options.combine_outputs();

  *max_range_end = -1;
  *total_elements = 0;
  for (const auto& range : options.ranges()) {
    ranges->push_back({range.begin(), range.end()});
    *max_range_end = std::max(*max_range_end, range.end());
    *total_elements += range.end() - range.begin();
  }

  return absl::OkStatus();
}

// Check that ranges don't overlap (used when elements can't be copied or when using combine_outputs)
inline absl::Status CheckRangesDontOverlap(
    const ::hand_tracking_mp_lean::SplitVectorCalculatorOptions& options) {
  for (int i = 0; i < options.ranges_size() - 1; ++i) {
    for (int j = i + 1; j < options.ranges_size(); ++j) {
      const auto& range_0 = options.ranges(i);
      const auto& range_1 = options.ranges(j);
      if ((range_0.begin() >= range_1.begin() &&
           range_0.begin() < range_1.end()) ||
          (range_1.begin() >= range_0.begin() &&
           range_1.begin() < range_0.end())) {
        return absl::InvalidArgumentError(
            "Ranges must be non-overlapping when using combine_outputs "
            "option.");
      }
    }
  }
  return absl::OkStatus();
}

// Process the input vector for copyable elements
template <typename T>
absl::Status ProcessCopyableElements(
    const std::vector<T>& input,
    const std::vector<std::pair<int32_t, int32_t>>& ranges,
    int32_t max_range_end,
    int32_t total_elements,
    bool element_only,
    bool combine_outputs,
    std::vector<std::unique_ptr<std::vector<T>>>* output_vectors,
    std::vector<T>* output_elements,
    std::unique_ptr<std::vector<T>>* combined_output) {

  if (!std::is_copy_constructible<T>::value) {
    return absl::InternalError("Cannot copy non-copyable elements.");
  }

  RET_CHECK_GE(input.size(), max_range_end);

  if (combine_outputs) {
    *combined_output = absl::make_unique<std::vector<T>>();
    (*combined_output)->reserve(total_elements);
    for (int i = 0; i < ranges.size(); ++i) {
      auto elements = absl::make_unique<std::vector<T>>(
          input.begin() + ranges[i].first,
          input.begin() + ranges[i].second);
      (*combined_output)->insert((*combined_output)->end(), elements->begin(), elements->end());
    }
  } else {
    if (element_only) {
      output_elements->reserve(ranges.size());
      for (int i = 0; i < ranges.size(); ++i) {
        output_elements->push_back(input[ranges[i].first]);
      }
    } else {
      output_vectors->reserve(ranges.size());
      for (int i = 0; i < ranges.size(); ++i) {
        auto output = absl::make_unique<std::vector<T>>(
            input.begin() + ranges[i].first,
            input.begin() + ranges[i].second);
        output_vectors->push_back(std::move(output));
      }
    }
  }

  return absl::OkStatus();
}

// Process the input vector for movable elements
template <typename T>
absl::Status ProcessMovableElements(
    std::unique_ptr<std::vector<T>>* input_vector,
    const std::vector<std::pair<int32_t, int32_t>>& ranges,
    int32_t max_range_end,
    int32_t total_elements,
    bool element_only,
    bool combine_outputs,
    std::vector<std::unique_ptr<std::vector<T>>>* output_vectors,
    std::vector<T>* output_elements,
    std::unique_ptr<std::vector<T>>* combined_output) {

  if (!std::is_move_constructible<T>::value) {
    return absl::InternalError("Cannot move non-movable elements.");
  }

  RET_CHECK_GE((*input_vector)->size(), max_range_end);

  if (combine_outputs) {
    *combined_output = absl::make_unique<std::vector<T>>();
    (*combined_output)->reserve(total_elements);
    for (int i = 0; i < ranges.size(); ++i) {
      (*combined_output)->insert(
          (*combined_output)->end(),
          std::make_move_iterator((*input_vector)->begin() + ranges[i].first),
          std::make_move_iterator((*input_vector)->begin() + ranges[i].second));
    }
  } else {
    if (element_only) {
      output_elements->reserve(ranges.size());
      for (int i = 0; i < ranges.size(); ++i) {
        output_elements->push_back(std::move((*input_vector)->at(ranges[i].first)));
      }
    } else {
      output_vectors->reserve(ranges.size());
      for (int i = 0; i < ranges.size(); ++i) {
        auto output = absl::make_unique<std::vector<T>>();
        output->insert(
            output->end(),
            std::make_move_iterator((*input_vector)->begin() + ranges[i].first),
            std::make_move_iterator((*input_vector)->begin() + ranges[i].second));
        output_vectors->push_back(std::move(output));
      }
    }
  }

  return absl::OkStatus();
}

}  // namespace hand_tracking_mp_lean

#endif  // MEDIAPIPE_CALCULATORS_CORE_SPLIT_VECTOR_CALCULATOR_CORE_H_
