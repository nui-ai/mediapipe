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

#include <vector>

#include "mediapipe/calculators/util/collection_has_min_size_calculator.h"
#include "mediapipe/calculators/tensor/image_to_tensor_utils.h"
#include "mediapipe/framework/api2/port.h"
#include "mediapipe/framework/port/status.h"

#include "mediapipe/framework/formats/classification.pb.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/rect.pb.h"

namespace mediapipe {

class HeadCalculator : public CollectionHasMinSizeCalculator<std::vector<mediapipe::NormalizedRect>> {
 public:
  absl::Status Open(CalculatorContext* cc) override {
    // Call base Open to set min_size_ from options or side packet.
    MP_RETURN_IF_ERROR(CollectionHasMinSizeCalculator::Open(cc));
    // Check shared config for "num_hands".
    min_size_ = GetSharedState().NUM_HANDS;
    return absl::OkStatus();
  }

  /// this function now replaces the following calculator's work, and no longer needs to inherit CollectionHasMinSizeCalculator's Process method.
  ///
  /// - NormalizedRectVectorHasMinSizeCalculator: Determines if an input vector of NormalizedRect has a size greater than or equal to the globally defined num_hands.
  /// - GateCalculator: Drops the incoming image if enough hands have already been identified from the previous image. Otherwise, passes the incoming image through to trigger a new round of palm detection.
  ///
  /// *  uses shared state as stream input and stream output
  /// ** algorithm imporvement note: this is a little coarse, but it works as a great baseline
  absl::Status Process(CalculatorContext* cc) override {

    ABSL_LOG(INFO) << "NormalizedRectVectorHasMinSizeCalculator starting to process";

    static constexpr api2::Input<api2::OneOf<mediapipe::Image, mediapipe::ImageFrame>>::Optional kIn{"IMAGE"};
    MP_ASSIGN_OR_RETURN(GetSharedState().image, GetInputImage(kIn(cc)));

    ABSL_LOG(INFO) << "image is null? " << (GetSharedState().image == nullptr);

    // trigger palm detection, as we don't have detection signal from the previous frame's landmarks processing for the amount of expected hands.
    // AssociationNormRectCalculator may later reconcile the rects from palm detection with those from landmarks tracking, but that is not this
    // calculator's concern.
    if (GetSharedState().prev_hand_rects_from_landmarks.size() < min_size_) {

      GetSharedState().palm_detection_image = GetSharedState().image;

      GetSharedState().ImageToTensorCalculatorOptions_output_tensor_width = 192;
      GetSharedState().ImageToTensorCalculatorOptions_output_tensor_height = 192;
      GetSharedState().ImageToTensorCalculatorOptions_keep_aspect_ratio = true;
      GetSharedState().ImageToTensorCalculatorOptions_float_range_min = 0.0f;
      GetSharedState().ImageToTensorCalculatorOptions_float_range_max = 1.0f;
      GetSharedState().ImageToTensorCalculatorOptions_border_mode = 1; // reuse the zero border mode value from the Options class enumeration for clarity
      ABSL_LOG(INFO) << "palm detection is being triggered";
    } else {
      // avoid applying hand detection as we have an amount of hand detections from the previous frame's landmarks processing
      // which is the same or more than the amount of hands our pipeline has been configured to track.
      GetSharedState().palm_detection_image = nullptr;
      ABSL_LOG(INFO) << "skipping palm detection as " << GetSharedState().prev_hand_rects_from_landmarks.size() << " hands have been detected from the previous frame's landmarks processing.";
      if (GetSharedState().prev_hand_rects_from_landmarks.size() > min_size_) {
        ABSL_LOG(INFO) << "number of hands detected from previous frame's landmarks (" << GetSharedState().prev_hand_rects_from_landmarks.size() << ") is larger than the number of hands being tracked (" << min_size_ << ") which is unexpected.";
      }
    }

    return CollectionHasMinSizeCalculator::Process(cc);
    return absl::OkStatus();
  }


};
REGISTER_CALCULATOR(HeadCalculator);

typedef CollectionHasMinSizeCalculator<
    std::vector<mediapipe::NormalizedLandmarkList>>
    NormalizedLandmarkListVectorHasMinSizeCalculator;
REGISTER_CALCULATOR(NormalizedLandmarkListVectorHasMinSizeCalculator);

typedef CollectionHasMinSizeCalculator<
    std::vector<mediapipe::ClassificationList>>
    ClassificationListVectorHasMinSizeCalculator;
REGISTER_CALCULATOR(ClassificationListVectorHasMinSizeCalculator);

}  // namespace mediapipe
