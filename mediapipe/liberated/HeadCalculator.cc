#include "HeadCalculator.h"
#include <vector>
#include <memory>
#include <array>
#include "mediapipe/framework/memory_manager.h"
#include "mediapipe/framework/memory_manager_service.h"
#include "mediapipe/framework/api2/port.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/formats/classification.pb.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/calculators/tensor/image_to_tensor_utils.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/calculators/tensor/image_to_tensor_calculator_core.h"

namespace mediapipe {

  class HeadCalculator : public CalculatorBase {

    private:
      std::unique_ptr<api2::ImageToTensorCalculatorCore> image_to_tensor_core_;
      std::unique_ptr<ImageToTensorConverter> gpu_converter_;
      std::unique_ptr<ImageToTensorConverter> cpu_converter_;
      int min_size_ = 0;

      static constexpr api2::Output<std::vector<Tensor>>::Optional kOutTensors{"TENSORS"};
      static constexpr api2::Output<Tensor>::Optional kOutTensor{"TENSOR"};
      static constexpr api2::Output<std::array<float, 4>>::Optional kOutLetterboxPadding{"LETTERBOX_PADDING"};

    public:
      static absl::Status GetContract(CalculatorContract* cc) {
        RET_CHECK(cc->Inputs().HasTag("IMAGE"));
        RET_CHECK(cc->Inputs().HasTag("ITERABLE"));
        // Accept ImageFrame for IMAGE input; helper converts to Image if needed.
        cc->Inputs().Tag("IMAGE").Set<ImageFrame>();
        // Iterable input is a vector of NormalizedRect.
        cc->Inputs().Tag("ITERABLE").Set<std::vector<NormalizedRect>>();
        // Unnamed boolean output at index 0 (legacy behavior expected by graph).
        cc->Outputs().Index(0).Set<bool>();

        cc->Outputs().Tag("TENSORS").Set<std::vector<Tensor>>();
        cc->Outputs().Tag("LETTERBOX_PADDING").Set<std::array<float, 4>>();
        return absl::OkStatus();
      }

      absl::Status Open(CalculatorContext* cc) override {
        min_size_ = GetSharedState().NUM_HANDS;
        MemoryManager* memory_manager_ = nullptr;
        if (cc->Service(kMemoryManagerService).IsAvailable()) {
          memory_manager_ = &cc->Service(kMemoryManagerService).GetObject();
        }
        // Configure fixed 192x192 core options.
        auto options_ = ImageToTensorCalculatorOptions();
        options_.set_output_tensor_width(192);
        options_.set_output_tensor_height(192);
        options_.set_keep_aspect_ratio(true);
        options_.mutable_output_tensor_float_range()->set_min(0.0f);
        options_.mutable_output_tensor_float_range()->set_max(1.0f);
        options_.set_border_mode(mediapipe::ImageToTensorCalculatorOptions::BORDER_ZERO);
        auto params_ = GetOutputTensorParams(options_);
        int tensor_width = params_.output_width.value_or(0);
        int tensor_height = params_.output_height.value_or(0);
        image_to_tensor_core_ = std::make_unique<api2::ImageToTensorCalculatorCore>(
            options_, tensor_width, tensor_height, params_,
            gpu_converter_, cpu_converter_, memory_manager_);

        return absl::OkStatus();
      }

      /// Determines if an input vector of NormalizedRect has a size greater than or equal to the globally defined num_hands (previously done by NormalizedRectVectorHasMinSizeCalculator).
      /// proceeds to palm detection if not enough hands have been passed from the previous iteration of the pipeline.
      ///
      absl::Status Process(CalculatorContext* cc) override {

        ABSL_LOG(INFO) << "HeadCalculator starting to process";

        static constexpr api2::Input<api2::OneOf<Image, ImageFrame>>::Optional kIn{"IMAGE"};
        MP_ASSIGN_OR_RETURN(GetSharedState().image, GetInputImage(kIn(cc)));

        // Legacy has_min_size computation (mirrors CollectionHasMinSizeCalculator behavior)
        bool iterable_is_empty = cc->Inputs().Tag("ITERABLE").IsEmpty();
        int iterable_size = 0;
        if (!iterable_is_empty) {
          const auto& rects = cc->Inputs().Tag("ITERABLE").Get<std::vector<NormalizedRect>>();
          iterable_size = static_cast<int>(rects.size());
        }
        bool has_min_size = (iterable_is_empty || iterable_size < min_size_) ? false : true;
        ABSL_LOG(INFO) << "HeadCalculator has_min_size: " << has_min_size
                        << " (min_size_=" << min_size_
                        << ", iterable_size=" << iterable_size << ")";
        cc->Outputs().Index(0).AddPacket(MakePacket<bool>(has_min_size).At(cc->InputTimestamp()));

        std::shared_ptr<const mediapipe::Image> image = GetSharedState().image;
        ABSL_LOG(INFO) << "number of hands detected from previous frame's landmarks: " << GetSharedState().prev_hand_rects_from_landmarks.size();
        if (GetSharedState().prev_hand_rects_from_landmarks.size() < min_size_) {

          GetSharedState().palm_detection_image = GetSharedState().image;
          ABSL_LOG(INFO) << "palm detection is being triggered";

          api2::ImageToTensorCoreResult core_result;
          absl::optional<mediapipe::NormalizedRect> norm_rect = absl::nullopt;
          MP_RETURN_IF_ERROR(image_to_tensor_core_->Process(*image, norm_rect, &core_result));

          kOutLetterboxPadding(cc).Send(core_result.padding);
          auto result = std::make_unique<std::vector<Tensor>>();
          *result = std::move(core_result.tensors);
          kOutTensors(cc).Send(std::move(result));

        } else {

          // avoid applying hand detection as we have an amount of hand detections from the previous frame's landmarks processing
          // which is the same or more than the amount of hands our pipeline has been configured to track.
          GetSharedState().palm_detection_image = nullptr;
          ABSL_LOG(INFO) << "skipping palm detection as " << GetSharedState().prev_hand_rects_from_landmarks.size() << " hands have been detected from the previous frame's landmarks processing.";

          if (GetSharedState().prev_hand_rects_from_landmarks.size() > min_size_) {
            ABSL_LOG(INFO) << "number of hands detected from previous frame's landmarks (" << GetSharedState().prev_hand_rects_from_landmarks.size() << ") is larger than the number of hands being tracked (" << min_size_ << ") which is unexpected.";
          }
        }

        return absl::OkStatus();
      }


    };
  REGISTER_CALCULATOR(HeadCalculator);
}