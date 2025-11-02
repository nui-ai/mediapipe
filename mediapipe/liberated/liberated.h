#ifndef MEDIAPIPE_LIBERATED_H
#define MEDIAPIPE_LIBERATED_H
#include <memory>
#include "mediapipe/framework/memory_manager.h"
#include "mediapipe/calculators/tensor/image_to_tensor_utils.h"
#include "mediapipe/calculators/tensor/image_to_tensor_calculator_core.h"
#include "mediapipe/calculators/tensor/model_inference.h"
#include "mediapipe/calculators/tensor/tensors_to_detections_calculator_core.h"

namespace mediapipe_v01013_based {

class Liberated {
 public:

  explicit Liberated(MemoryManager* memory_manager);
  ~Liberated() = default;

  // Non-copyable, movable.
  Liberated(const Liberated&) = delete;
  Liberated& operator=(const Liberated&) = delete;
  Liberated(Liberated&&) = default;
  Liberated& operator=(Liberated&&) = default;

  [[nodiscard]] absl::StatusOr<std::unique_ptr<std::vector<Detection>>> Process(const std::vector<mediapipe_v01013_based::NormalizedRect> &prev_hand_rects_from_landmarks, std::shared_ptr<const mediapipe_v01013_based::Image> image, uint32_t max_hands_to_track) const;

 private:
  std::unique_ptr<api2::ImageToTensorCalculatorCore> image_to_tensor_core_;
  std::unique_ptr<ImageToTensorConverter> gpu_converter_;
  std::unique_ptr<ImageToTensorConverter> cpu_converter_;
  std::unique_ptr<api2::ModelInference> palm_detection_inference_;
  std::unique_ptr<api2::ConvertDetectionTensors> inference_filter_stage1_;
};

}
#endif