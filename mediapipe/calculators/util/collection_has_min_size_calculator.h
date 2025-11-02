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

#ifndef MEDIAPIPE_CALCULATORS_UTIL_COLLECTION_HAS_MIN_SIZE_CALCULATOR_H_
#define MEDIAPIPE_CALCULATORS_UTIL_COLLECTION_HAS_MIN_SIZE_CALCULATOR_H_

#include <vector>

#include "mediapipe/calculators/util/collection_has_min_size_calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/canonical_errors.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"

namespace mediapipe_v01013_based {
  class ImageFrame;

  // Deterimines if an input iterable collection has a minimum size, specified
// in CollectionHasMinSizeCalculatorOptions. Example usage:
// node {
//   calculator: "IntVectorHasMinSizeCalculator"
//   input_stream: "ITERABLE:input_int_vector"
//   output_stream: "has_min_ints"
//  options {
//    [mediapipe.CollectionHasMinSizeCalculatorOptions.ext] {
//      min_size: 2
//    }
//   }
// }
// Optionally, uses a side packet to override `min_size` specified in the
// calculator options.
template <typename IterableT>
class CollectionHasMinSizeCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc) {
    RET_CHECK(cc->Inputs().HasTag("IMAGE"));
    RET_CHECK(cc->Inputs().HasTag("ITERABLE"));

    RET_CHECK_GE(
        cc->Options<::mediapipe_v01013_based::CollectionHasMinSizeCalculatorOptions>()
            .min_size(),
        0);

    cc->Inputs().Tag("ITERABLE").Set<IterableT>();
    cc->Inputs().Tag("IMAGE").Set<ImageFrame>();
    cc->Outputs().Index(0).Set<bool>();

    // Optional input side packet that determines `min_size_`.
    if (cc->InputSidePackets().NumEntries() > 0) {
      cc->InputSidePackets().Index(0).Set<int>();
    }
    return absl::OkStatus();
  }

  absl::Status Open(CalculatorContext* cc) override {
    cc->SetOffset(TimestampDiff(0));
    min_size_ =
        cc->Options<::mediapipe_v01013_based::CollectionHasMinSizeCalculatorOptions>()
            .min_size();
    // Override `min_size` if passed as side packet.
    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) override {
    ABSL_LOG(INFO) << "legacy process starting";

    // the condition has been modified as adaptation to no longer feeding off a Gate calculator because:
    //
    // 1. because without the Gate calculator suppressing our input stream if it were an empty packet ― this calculator is now called also in that case ―
    //    which previously it did not get called for. so modified here as adaptation to no longer feeding off a Gate calculator.
    //    instead of never being called when there was no packet in prev_hand_rects_from_landmarks ― now we are called also in that case.
    // 2. then, the behavior adapatation here is that this calculator will now trigger palm detection also if the input iterable is empty.
    //    this is required not because of the Gate calculator swallowed which this calculator feeds from ― but because of the specific fact
    //    that the Gate calculator FEEDING FROM the current one should also be swallowed.
    //
    // this bridges the emptinesss semantics across Gate calculators, to a pipeline flow which decides about passing forward in plain non Gate-calculator logic.
    //
    // so in summary, this adaptation is tailored to that very specific state of being sandwitched beteween two Gate calculators on the flow of the pipeline!
    // and is not the simple case of swalling a single Gate calculator.
    //
    // it's not even a generic recipe, because empty semantics hinges on Gate calculator Options such as empty_packets_as_allow:true and allow:true (!!)
    // so it's not a generic recipe but every Gate calc removal needs to be tailored to the Gate calculator Options values and overall pipeline context
    // (which nodes feed from it, which nodes feed into it, by their streams).
    bool has_min_size = (cc->Inputs().Tag("ITERABLE").IsEmpty() || cc->Inputs().Tag("ITERABLE").Get<IterableT>().size() < min_size_) ? false : true;
    ABSL_LOG(INFO) << "HeadCalculator legacy part has_min_size: " << has_min_size << " (min_size_ is " << min_size_ << ", input iterable size is " << (cc->Inputs().Tag("ITERABLE").IsEmpty() ? 0 : cc->Inputs().Tag("ITERABLE").Get<IterableT>().size()) << ")";
    cc->Outputs().Index(0).AddPacket(MakePacket<bool>(has_min_size).At(cc->InputTimestamp()));

    return absl::OkStatus();
  }

 protected:
  int min_size_ = 0;
};

}  // namespace mediapipe_v01013_based

#endif  // MEDIAPIPE_CALCULATORS_UTIL_COLLECTION_HAS_MIN_SIZE_CALCULATOR_H_
