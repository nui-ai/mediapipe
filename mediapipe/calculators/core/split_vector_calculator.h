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

#ifndef MEDIAPIPE_CALCULATORS_CORE_SPLIT_VECTOR_CALCULATOR_H_
#define MEDIAPIPE_CALCULATORS_CORE_SPLIT_VECTOR_CALCULATOR_H_

#include <cstdint>
#include <type_traits>
#include <vector>

#include "mediapipe/calculators/core/split_vector_calculator.pb.h"
#include "mediapipe/calculators/core/split_vector_calculator_core.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/canonical_errors.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/util/resource_util.h"
#include "tensorflow/lite/error_reporter.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"

namespace mediapipe_v01013_based {

template <typename T>
using IsCopyable = std::enable_if_t<std::is_copy_constructible<T>::value, bool>;

template <typename T>
using IsNotCopyable =
    std::enable_if_t<!std::is_copy_constructible<T>::value, bool>;

template <typename T>
using IsMovable = std::enable_if_t<std::is_move_constructible<T>::value, bool>;

template <typename T>
using IsNotMovable =
    std::enable_if_t<!std::is_move_constructible<T>::value, bool>;

// Splits an input packet with std::vector<T> into multiple std::vector<T>
// output packets using the [begin, end) ranges specified in
// SplitVectorCalculatorOptions. If the option "element_only" is set to true,
// all ranges should be of size 1 and all outputs will be elements of type T. If
// "element_only" is false, ranges can be non-zero in size and all outputs will
// be of type std::vector<T>. If the option "combine_outputs" is set to true,
// only one output stream can be specified and all ranges of elements will be
// combined into one vector.
// To use this class for a particular type T, register a calculator using
// SplitVectorCalculator<T>.
template <typename T, bool move_elements>
class SplitVectorCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc) {
    RET_CHECK(cc->Inputs().NumEntries() == 1);
    RET_CHECK(cc->Outputs().NumEntries() != 0);

    cc->Inputs().Index(0).Set<std::vector<T>>();

    const auto& options =
        cc->Options<::mediapipe_v01013_based::SplitVectorCalculatorOptions>();

    if (!std::is_copy_constructible<T>::value || move_elements) {
      // Ranges of elements shouldn't overlap when the vector contains
      // non-copyable elements.
      RET_CHECK_OK(CheckRangesDontOverlap(options));
    }

    if (options.combine_outputs()) {
      RET_CHECK_EQ(cc->Outputs().NumEntries(), 1);
      cc->Outputs().Index(0).Set<std::vector<T>>();
      RET_CHECK_OK(CheckRangesDontOverlap(options));
    } else {
      if (cc->Outputs().NumEntries() != options.ranges_size()) {
        return absl::InvalidArgumentError(
            "The number of output streams should match the number of ranges "
            "specified in the CalculatorOptions.");
      }

      // Set the output types for each output stream.
      for (int i = 0; i < cc->Outputs().NumEntries(); ++i) {
        if (options.ranges(i).begin() < 0 || options.ranges(i).end() < 0 ||
            options.ranges(i).begin() >= options.ranges(i).end()) {
          return absl::InvalidArgumentError(
              "Indices should be non-negative and begin index should be less "
              "than the end index.");
        }
        if (options.element_only()) {
          if (options.ranges(i).end() - options.ranges(i).begin() != 1) {
            return absl::InvalidArgumentError(
                "Since element_only is true, all ranges should be of size 1.");
          }
          cc->Outputs().Index(i).Set<T>();
        } else {
          cc->Outputs().Index(i).Set<std::vector<T>>();
        }
      }
    }

    return absl::OkStatus();
  }

  absl::Status Open(CalculatorContext* cc) override {
    cc->SetOffset(TimestampDiff(0));

    const auto& options =
        cc->Options<::mediapipe_v01013_based::SplitVectorCalculatorOptions>();

    // Use extracted function to initialize calculator state
    return InitializeSplitVectorCalculator<T>(
        options, &ranges_, &max_range_end_, &total_elements_,
        &element_only_, &combine_outputs_);
  }

  absl::Status Process(CalculatorContext* cc) override {
    if (cc->Inputs().Index(0).IsEmpty()) return absl::OkStatus();

    if (move_elements) {
      return ProcessMovableElements<T>(cc);
    } else {
      return ProcessCopyableElements<T>(cc);
    }
  }

  template <typename U, IsCopyable<U> = true>
  absl::Status ProcessCopyableElements(CalculatorContext* cc) {
    const auto& input = cc->Inputs().Index(0).Get<std::vector<U>>();

    std::vector<std::unique_ptr<std::vector<T>>> output_vectors;
    std::vector<T> output_elements;
    std::unique_ptr<std::vector<T>> combined_output;

    // Use extracted function to process the input
    auto status = ::mediapipe_v01013_based::ProcessCopyableElements<T>(
        input, ranges_, max_range_end_, total_elements_,
        element_only_, combine_outputs_,
        &output_vectors, &output_elements, &combined_output);

    if (!status.ok()) return status;

    // Handle output based on configuration
    if (combine_outputs_) {
      cc->Outputs().Index(0).Add(combined_output.release(), cc->InputTimestamp());
    } else {
      if (element_only_) {
        for (int i = 0; i < ranges_.size(); ++i) {
          cc->Outputs().Index(i).AddPacket(
              MakePacket<U>(output_elements[i]).At(cc->InputTimestamp()));
        }
      } else {
        for (int i = 0; i < ranges_.size(); ++i) {
          cc->Outputs().Index(i).Add(output_vectors[i].release(), cc->InputTimestamp());
        }
      }
    }

    return absl::OkStatus();
  }

  template <typename U, IsNotCopyable<U> = true>
  absl::Status ProcessCopyableElements(CalculatorContext* cc) {
    return absl::InternalError("Cannot copy non-copyable elements.");
  }

  template <typename U, IsMovable<U> = true>
  absl::Status ProcessMovableElements(CalculatorContext* cc) {
    absl::StatusOr<std::unique_ptr<std::vector<U>>> input_status =
        cc->Inputs().Index(0).Value().Consume<std::vector<U>>();
    if (!input_status.ok()) return input_status.status();
    std::unique_ptr<std::vector<U>> input_vector =
        std::move(input_status).value();

    std::vector<std::unique_ptr<std::vector<T>>> output_vectors;
    std::vector<T> output_elements;
    std::unique_ptr<std::vector<T>> combined_output;

    // Use extracted function to process the input
    auto status = ::mediapipe_v01013_based::ProcessMovableElements<T>(
        &input_vector, ranges_, max_range_end_, total_elements_,
        element_only_, combine_outputs_,
        &output_vectors, &output_elements, &combined_output);

    if (!status.ok()) return status;

    // Handle output based on configuration
    if (combine_outputs_) {
      cc->Outputs().Index(0).Add(combined_output.release(), cc->InputTimestamp());
    } else {
      if (element_only_) {
        for (int i = 0; i < ranges_.size(); ++i) {
          cc->Outputs().Index(i).AddPacket(
              MakePacket<U>(std::move(output_elements[i]))
                  .At(cc->InputTimestamp()));
        }
      } else {
        for (int i = 0; i < ranges_.size(); ++i) {
          cc->Outputs().Index(i).Add(output_vectors[i].release(), cc->InputTimestamp());
        }
      }
    }

    return absl::OkStatus();
  }

  template <typename U, IsNotMovable<U> = true>
  absl::Status ProcessMovableElements(CalculatorContext* cc) {
    return absl::InternalError("Cannot move non-movable elements.");
  }

 private:
  std::vector<std::pair<int32_t, int32_t>> ranges_;
  int32_t max_range_end_ = -1;
  int32_t total_elements_ = 0;
  bool element_only_ = false;
  bool combine_outputs_ = false;
};

}  // namespace mediapipe_v01013_based

#endif  // MEDIAPIPE_CALCULATORS_CORE_SPLIT_VECTOR_CALCULATOR_H_
