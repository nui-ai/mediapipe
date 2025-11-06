#ifndef MEDIAPIPE_LIBERATED_H
#define MEDIAPIPE_LIBERATED_H

#include <memory>

#include "mediapipe/framework/memory_manager.h"
#include "mediapipe/calculators/tensor/image_to_tensor_calculator_core.h"
#include "mediapipe/calculators/tensor/model_inference.h"
#include "mediapipe/calculators/tensor/tensors_to_detections_calculator_core.h"
#include "mediapipe/calculators/util/detections_to_rects_calculator_core.h"
#include "mediapipe/calculators/tensor/inference_runner.h"
#include "mediapipe/calculators/util/association_calculator_core.h"
#include "mediapipe/calculators/util/detection_letterbox_removal.h"
#include "mediapipe/calculators/util/rect_transformation_calculator_core.h"
#include "mediapipe/calculators/core/inference_output_tensor_splitting.h"
#include "mediapipe/calculators/tensor/tensors_to_floats_calculator_core.h"
#include "mediapipe/calculators/tensor/tensors_to_classification_calculator_core.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/formats/classification.pb.h"
#include "mediapipe/calculators/tensor/tensors_to_landmarks_calculator.pb.h"
#include "mediapipe/calculators/tensor/tensors_to_classification_calculator.pb.h"
#include "mediapipe/calculators/tensor/tensors_to_landmarks_calculator_core.h"
#include "mediapipe/calculators/util/landmark_letterbox_removal_calculator_core.h"
#include "mediapipe/calculators/util/landmark_projection_calculator_core.h"
#include "mediapipe/calculators/tensor/tensors_to_world_landmarks_calculator_core.h"
#include "mediapipe/calculators/util/world_landmark_projection_calculator_core.h"
#include "mediapipe/modules/hand_landmark/calculators/hand_landmarks_to_rect_calculator_core.h"

namespace mediapipe_v01013_based {
 class DetectionsToOrientedRects;

 class Liberated {
 public:

  explicit Liberated(MemoryManager* memory_manager);
  ~Liberated() = default;

  // Non-copyable, movable.
  Liberated(const Liberated&) = delete;
  Liberated& operator=(const Liberated&) = delete;
  Liberated(Liberated&&) = default;
  Liberated& operator=(Liberated&&) = default;

  [[nodiscard]] absl::StatusOr<std::unique_ptr<std::vector<NormalizedRect>>> Process(const std::vector<mediapipe_v01013_based::NormalizedRect> &prev_hand_rects_from_landmarks, std::shared_ptr<const mediapipe_v01013_based::Image> image, uint32_t max_hands_to_track) const;

 private:
  std::unique_ptr<api2::ImageToTensorCalculatorCore> image_to_tensor_core_;
  std::unique_ptr<ImageToTensorConverter> gpu_converter_;
  std::unique_ptr<ImageToTensorConverter> cpu_converter_;
  std::unique_ptr<api2::ModelInference> palm_detection_inference_;
  std::unique_ptr<api2::ConvertDetectionTensors> inference_filter_stage1_;
  std::unique_ptr<DetectionsToOrientedRects> palm_detection_to_oriented_palm_rect_;
  std::unique_ptr<RectTransformation> oriented_palm_rect_to_hand_rect_expander_;
  std::unique_ptr<api2::ImageToTensorCalculatorCore> sub_image_for_landmarks_inference_extractor_;
  std::unique_ptr<api2::ModelInference> landmarks_inference_;
  std::unique_ptr<InferenceOutputTensorSplitting<Tensor, false>> landmarks_inference_splitter_;
  api2::TensorsToClassificationConfig handedness_classification_config_;
  std::unique_ptr<api2::TensorsToLandmarksCore> landmarks_extractor_;
  std::unique_ptr<RectTransformation> expand_rect_for_next_frame_;
 };

}
#endif