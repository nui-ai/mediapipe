// Copyright 2022 The MediaPipe Authors.
// #include "mediapipe/calculators/tensor/inference_calculator_xnnpack.cc"
#include <string>
#include "mediapipe/calculators/tensor/inference_calculator_xnnpack.h"
#include "mediapipe/framework/calculator_registry.h"

namespace mediapipe {
namespace api2 {

class HandLandmarksInferenceCalculatorXnnpackImpl : public InferenceCalculatorXnnpackImpl {
 protected:
  [[nodiscard]] std::string GetModelPath() const override {
    return std::string("mediapipe/modules/hand_landmark/hand_landmark_full.tflite");
  }
};

}  // namespace api2
}  // namespace mediapipe

REGISTER_CALCULATOR(mediapipe::api2::HandLandmarksInferenceCalculatorXnnpackImpl);
