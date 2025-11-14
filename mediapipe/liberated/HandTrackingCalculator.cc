#include "HandTrackingCalculator.h"
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

namespace hand_tracking_mp_lean {

  // mediapipe Calculator performing the entire work of the former hand tracking pipeline (HandLandmarkTrackingCpu)
  // of mediapipe commit tag v0.10.13, which originally spanned many calculators and sub-graphs.
  // this implementation was one-to-one verified to yield the same results over long videos.
  class HandTrackingCalculator : public CalculatorBase {

      int max_hands_to_track = 3;

      std::unique_ptr<api2::ImageToTensorCalculatorCore> image_to_tensor_core_;
      std::unique_ptr<ImageToTensorConverter> gpu_converter_;
      std::unique_ptr<ImageToTensorConverter> cpu_converter_;
      std::unique_ptr<HandTrackingCore> liberated_;

      static constexpr api2::Output<std::vector<Tensor>>::Optional kOutTensors{"TENSORS"};
      static constexpr api2::Output<Tensor>::Optional kOutTensor{"TENSOR"};
      static constexpr api2::Output<std::array<float, 4>>::Optional kOutLetterboxPadding{"LETTERBOX_PADDING"};

    public:
      static absl::Status GetContract(CalculatorContract* cc) {

        RET_CHECK(cc->Inputs().HasTag("IMAGE"));
        cc->Inputs().Tag("IMAGE").Set<ImageFrame>();

        // cc->Outputs().Index(0).Set<bool>(); // untagged output stream 0
        cc->Outputs().Tag("LANDMARKS").Set<std::vector<NormalizedLandmarkList>>();
        cc->Outputs().Tag("WORLD_LANDMARKS").Set<std::vector<LandmarkList>>();
        cc->Outputs().Tag("HANDEDNESS").Set<std::vector<ClassificationList>>();

        // cc->Outputs().Index(1).Set<std::vector<NormalizedRect>>(); // untagged output stream 1
        // cc->Outputs().Tag("TENSORS").Set<std::vector<Tensor>>();
        // cc->Outputs().Tag("LETTERBOX_PADDING").Set<std::array<float, 4>>();
        // cc->Outputs().Tag("ITERABLE").Set<NormalizedRect>();
        // cc->Outputs().Tag("DETECTIONS").Set<std::vector<Detection>>();
        return absl::OkStatus();
      }

      absl::Status Open(CalculatorContext* cc) override {
        ABSL_LOG(INFO) << "tracking up to " << max_hands_to_track << " hands";

        MemoryManager* memory_manager_ = nullptr;
        if (cc->Service(kMemoryManagerService).IsAvailable()) {
          memory_manager_ = &cc->Service(kMemoryManagerService).GetObject();
        }
        liberated_ = std::make_unique<HandTrackingCore>(memory_manager_);
        return absl::OkStatus();
      }

      absl::Status Process(CalculatorContext* cc) override {

        static constexpr api2::Input<api2::OneOf<Image, ImageFrame>>::Optional kIn{"IMAGE"};
        std::shared_ptr<const hand_tracking_mp_lean::Image> image;
        MP_ASSIGN_OR_RETURN(image, GetInputImage(kIn(cc)));

        absl::StatusOr<std::unique_ptr<ImageHandTrackingAndInferenceResult>> resultOrStatus = liberated_->Process(image, max_hands_to_track);
        if (resultOrStatus.ok()) {
          cc->Outputs().Tag("LANDMARKS").Add(resultOrStatus.value()->viewport_landmarkss.release(), cc->InputTimestamp());
          cc->Outputs().Tag("WORLD_LANDMARKS").Add(resultOrStatus.value()->object_landmarkss.release(), cc->InputTimestamp());
          cc->Outputs().Tag("HANDEDNESS").Add(resultOrStatus.value()->handedness_classifications.release(), cc->InputTimestamp());
          return absl::OkStatus();
        }
        else {
          return resultOrStatus.status();
        }
      }
    };

  REGISTER_CALCULATOR(HandTrackingCalculator);
}