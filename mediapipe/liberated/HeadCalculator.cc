#include "HeadCalculator.h"
#include <vector>
#include "mediapipe/calculators/util/collection_has_min_size_calculator.h"
#include "mediapipe/calculators/tensor/image_to_tensor_utils.h"
#include "mediapipe/framework/api2/port.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/formats/classification.pb.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/calculators/util/collection_has_min_size_calculator.h"

namespace mediapipe {

  class HeadCalculator : public CollectionHasMinSizeCalculator<std::vector<NormalizedRect>> {
    public:
      absl::Status Open(CalculatorContext* cc) override {
        min_size_ = GetSharedState().NUM_HANDS;
        return absl::OkStatus();
      }

      /// Determines if an input vector of NormalizedRect has a size greater than or equal to the globally defined num_hands (previously done by NormalizedRectVectorHasMinSizeCalculator).
      /// proceeds to palm detection if not enough hands have been passed from the previous iteration of the pipeline.
      ///
      absl::Status Process(CalculatorContext* cc) override {

        ABSL_LOG(INFO) << "HeadCalculator starting to process";

        static constexpr api2::Input<api2::OneOf<Image, ImageFrame>>::Optional kIn{"IMAGE"};
        MP_ASSIGN_OR_RETURN(GetSharedState().image, GetInputImage(kIn(cc)));

        auto image = GetSharedState().image;
        ABSL_LOG(INFO) << "number of hands detected from previous frame's landmarks: " << GetSharedState().prev_hand_rects_from_landmarks.size();
        if (GetSharedState().prev_hand_rects_from_landmarks.size() < min_size_) {

          GetSharedState().palm_detection_image = GetSharedState().image;
          ABSL_LOG(INFO) << "palm detection is being triggered";

          image_to_tensor(&image);

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
      }


    };
  REGISTER_CALCULATOR(HeadCalculator);
}