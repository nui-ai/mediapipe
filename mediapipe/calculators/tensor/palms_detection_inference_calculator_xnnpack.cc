// Copyright 2022 The MediaPipe Authors.
#include <string>
#include "mediapipe/calculators/tensor/inference_calculator_xnnpack.h"
#include "mediapipe/framework/calculator_registry.h"

namespace mediapipe {
namespace api2 {

class PalmsDetectionInferenceCalculatorXnnpackImpl : public InferenceCalculatorXnnpackImpl {
 protected:
  [[nodiscard]] std::string GetModelPath() const override {
    return std::string("mediapipe/modules/palm_detection/palm_detection_full.tflite");
  }
};

}  // namespace api2
}  // namespace mediapipe

REGISTER_CALCULATOR(mediapipe::api2::PalmsDetectionInferenceCalculatorXnnpackImpl);
