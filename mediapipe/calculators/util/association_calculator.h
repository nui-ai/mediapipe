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

#ifndef MEDIAPIPE_CALCULATORS_UTIL_ASSOCIATION_CALCULATOR_H_
#define MEDIAPIPE_CALCULATORS_UTIL_ASSOCIATION_CALCULATOR_H_

#include <memory>
#include <vector>

#include "absl/log/absl_check.h"
#include "absl/memory/memory.h"
#include "mediapipe/calculators/util/association_calculator.pb.h"
#include "mediapipe/calculators/util/association_calculator_core.h"
#include "mediapipe/framework/calculator_context.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/collection_item_id.h"
#include "mediapipe/framework/port/canonical_errors.h"
#include "mediapipe/framework/port/rectangle.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/util/rectangle_util.h"

namespace hand_tracking_mp_lean {

// AssocationCalculator<T> accepts multiple inputs of vectors of type T that can
// be converted to Rectangle_f. The output is a vector of type T that contains
// elements from the input vectors that don't overlap with each other. When
// two elements overlap, the element that comes in from a later input stream
// is kept in the output. This association operation is useful for multiple
// instance inference pipelines in MediaPipe.
// If an input stream is tagged with "PREV" tag, IDs of overlapping elements
// from "PREV" input stream are propagated to the output. Elements in the "PREV"
// input stream that don't overlap with other elements are not added to the
// output. This stream is designed to take detections from previous timestamp,
// e.g. output of PreviousLoopbackCalculator to provide temporal association.
// See AssociationDetectionCalculator and AssociationNormRectCalculator for
// example uses.
template <typename T>
class AssociationCalculator : public CalculatorBase {
public:
  static absl::Status GetContract(CalculatorContract* cc) {
    // Atmost one input stream can be tagged with "PREV".
    RET_CHECK_LE(cc->Inputs().NumEntries("PREV"), 1);

    if (cc->Inputs().HasTag("PREV")) {
      RET_CHECK_GE(cc->Inputs().NumEntries(), 2);
    }

    for (CollectionItemId id = cc->Inputs().BeginId();
         id < cc->Inputs().EndId(); ++id) {
      cc->Inputs().Get(id).Set<std::vector<T>>();
         }

    cc->Outputs().Index(0).Set<std::vector<T>>();

    return absl::OkStatus();
  }

  absl::Status Open(CalculatorContext* cc) override {
    cc->SetOffset(TimestampDiff(0));

    has_prev_input_stream_ = cc->Inputs().HasTag("PREV");
    if (has_prev_input_stream_) {
      prev_input_stream_id_ = cc->Inputs().GetId("PREV", 0);
    }
    options_ = cc->Options<::hand_tracking_mp_lean::AssociationCalculatorOptions>();
    ABSL_CHECK_GE(options_.min_similarity_threshold(), 0);

    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) override {

    for (CollectionItemId stream_id = cc->Inputs().BeginId(); stream_id < cc->Inputs().EndId(); ++stream_id) {
      ABSL_LOG(INFO) << "stream_id: " << stream_id << " stream_name: " << *cc->Inputs().Get(stream_id).name_;
    }

    const auto& input_stream_1 = cc->Inputs().Get(cc->Inputs().BeginId());
    const auto& input_stream_2 = cc->Inputs().Get(cc->Inputs().EndId()-1);

    assert(*input_stream_1.name_ == std::string("hand_rects_from_palm_detections"));
    assert(*input_stream_2.name_ == std::string("prev_hand_rects_from_landmarks"));

    auto& hand_rects_from_palm_detection = input_stream_1.IsEmpty() ? std::vector<T>() : input_stream_1.Get<std::vector<T>>();
    auto& prev_hand_rects_from_landmarks = input_stream_2.IsEmpty() ? std::vector<T>() : input_stream_2.Get<std::vector<T>>();

    std::list<T> result_set;
    MP_ASSIGN_OR_RETURN(result_set,
      hand_tracking_mp_lean::IouFilterMerge(hand_rects_from_palm_detection, prev_hand_rects_from_landmarks, options_.min_similarity_threshold()));

    auto output = absl::make_unique<std::vector<T>>();
    for (auto it = result_set.begin(); it != result_set.end(); ++it) {
      output->push_back(*it);
    }
    cc->Outputs().Index(0).Add(output.release(), cc->InputTimestamp());

    return absl::OkStatus();
  }

protected:
  ::hand_tracking_mp_lean::AssociationCalculatorOptions options_;

  // if there is an input stream tagged with "PREV", set a boolean to true and store its ID.
  bool has_prev_input_stream_;
  CollectionItemId prev_input_stream_id_;

  virtual std::pair<bool, int> GetId(const T& input) { return {false, -1}; }

  virtual void SetId(T* input, int id) {}

  virtual absl::StatusOr<Rectangle_f> GetRectangle(const T& input) {
    return absl::OkStatus();
  }
};

};  // namespace hand_tracking_mp_lean

#endif  // MEDIAPIPE_CALCULATORS_UTIL_ASSOCIATION_CALCULATOR_H_
