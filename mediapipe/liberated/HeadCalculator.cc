#include "HeadCalculator.h"
#include <vector>
#include <memory>
#include <array>
#include "absl/log/absl_log.h"
#include "mediapipe/framework/memory_manager.h"
#include "mediapipe/framework/memory_manager_service.h"
#include "mediapipe/framework/api2/port.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/calculators/tensor/image_to_tensor_utils.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/calculators/tensor/image_to_tensor_calculator_core.h"
#include "mediapipe/liberated/liberated.h"

namespace mediapipe {

  class HeadCalculator : public CalculatorBase {

      std::unique_ptr<api2::ImageToTensorCalculatorCore> image_to_tensor_core_;
      std::unique_ptr<ImageToTensorConverter> gpu_converter_;
      std::unique_ptr<ImageToTensorConverter> cpu_converter_;
      std::unique_ptr<Liberated> liberated_;
      int max_hands_to_track = 0;

      static constexpr api2::Output<std::vector<Tensor>>::Optional kOutTensors{"TENSORS"};
      static constexpr api2::Output<Tensor>::Optional kOutTensor{"TENSOR"};
      static constexpr api2::Output<std::array<float, 4>>::Optional kOutLetterboxPadding{"LETTERBOX_PADDING"};

    public:
      static absl::Status GetContract(CalculatorContract* cc) {

        RET_CHECK(cc->Inputs().HasTag("IMAGE"));
        RET_CHECK(cc->Inputs().HasTag("ITERABLE"));
        cc->Inputs().Tag("IMAGE").Set<ImageFrame>();
        cc->Inputs().Tag("ITERABLE").Set<std::vector<NormalizedRect>>();

        cc->Outputs().Index(0).Set<bool>(); // Unnamed boolean output at index 0 (legacy behavior expected by graph).
        // cc->Outputs().Tag("TENSORS").Set<std::vector<Tensor>>();
        // cc->Outputs().Tag("LETTERBOX_PADDING").Set<std::array<float, 4>>();
        // cc->Outputs().Tag("ITERABLE").Set<NormalizedRect>();
        cc->Outputs().Tag("DETECTIONS").Set<std::vector<Detection>>();
        return absl::OkStatus();
      }

      absl::Status Open(CalculatorContext* cc) override {
        max_hands_to_track = GetSharedState().NUM_HANDS;
        MemoryManager* memory_manager_ = nullptr;
        if (cc->Service(kMemoryManagerService).IsAvailable()) {
          memory_manager_ = &cc->Service(kMemoryManagerService).GetObject();
        }
        liberated_ = std::make_unique<Liberated>(memory_manager_);
        return absl::OkStatus();
      }

      absl::Status Process(CalculatorContext* cc) override {

        ABSL_LOG(INFO) << "HeadCalculator starting to process";

        static constexpr api2::Input<api2::OneOf<Image, ImageFrame>>::Optional kIn{"IMAGE"};
        std::shared_ptr<const mediapipe::Image> image;
        MP_ASSIGN_OR_RETURN(image, GetInputImage(kIn(cc)));
        GetSharedState().image = image;

        bool iterable_is_empty = cc->Inputs().Tag("ITERABLE").IsEmpty();
        if (!iterable_is_empty) {
          const auto& rects = cc->Inputs().Tag("ITERABLE").Get<std::vector<NormalizedRect>>();
        }

        std::unique_ptr<std::vector<Detection>> partial_result;
        auto resultOrStatus = liberated_->Process(GetSharedState().prev_hand_rects_from_landmarks, image, max_hands_to_track);
        if (resultOrStatus.ok()) {
          auto output_detections = std::move(resultOrStatus.value());
          cc->Outputs()
            .Tag("DETECTIONS")
            .Add(output_detections.release(), cc->InputTimestamp());
          return absl::OkStatus();
        }
        else {
          return resultOrStatus.status();
        }
      }
    };

  REGISTER_CALCULATOR(HeadCalculator);
}