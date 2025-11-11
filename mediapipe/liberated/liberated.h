#ifndef MEDIAPIPE_LIBERATED_H
#define MEDIAPIPE_LIBERATED_H

#include <memory>

#include "mediapipe/framework/memory_manager.h"
#include "mediapipe/calculators/tensor/image_to_tensor_calculator_core.h"
#include "mediapipe/calculators/tensor/model_inference.h"
#include "mediapipe/calculators/tensor/detections_extraction.h"
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
 class DetectionsToOrientedRects;  // forward declaration hack that will ultimately be unnecessary

 struct ImageHandTrackingAndInferenceResult {
  std::unique_ptr<std::vector<NormalizedLandmarkList>> viewport_landmarkss;
  std::unique_ptr<std::vector<LandmarkList>> object_landmarkss;
  std::unique_ptr<std::vector<ClassificationList>> handedness_classifications;
 };

 class Liberated {
 public:

  explicit Liberated(MemoryManager* memory_manager);

  ~Liberated() = default;

  // Non-copyable, movable.
  Liberated(const Liberated&) = delete;
  Liberated& operator=(const Liberated&) = delete;
  Liberated(Liberated&&) = default;
  Liberated& operator=(Liberated&&) = default;

  [[nodiscard]] absl::StatusOr<std::unique_ptr<ImageHandTrackingAndInferenceResult>> Process(std::shared_ptr<const mediapipe_v01013_based::Image> image, uint32_t max_hands_to_track);

 private:

  // meta-parameters should ultimately go here.
  // values we pass to below implementations which must mirror how
  // neural network models have been trained should not really be
  // moved here as they cannot be fiddled.
  float hand_presence_in_landmarks_inference_threshold_ = 0.5f;

  // step implementations, some of which need initialization and hence an object,
  // and some don't and are just free functions.
  std::unique_ptr<api2::ImageToTensorCalculatorCore> image_to_palm_detection_input_;
  std::unique_ptr<api2::ModelInference> palm_detection_inference_;
  std::unique_ptr<api2::DetectionsExtractionAndFiltering> palm_detection_inference_filter_;
  std::unique_ptr<DetectionsToOrientedRects> palm_detection_to_oriented_palm_rect_;
  std::unique_ptr<RectTransformation> oriented_palm_rect_to_hand_rect_expander_;
  std::unique_ptr<api2::ImageToTensorCalculatorCore> sub_image_for_landmarks_inference_extractor_;
  std::unique_ptr<api2::ModelInference> landmarks_inference_;
  std::unique_ptr<InferenceOutputTensorSplitting<Tensor, false>> landmarks_inference_splitter_;
  api2::TensorsToClassificationConfig handedness_classification_config_;
  std::unique_ptr<api2::TensorsToLandmarksCore> landmarks_extractor_;
  std::unique_ptr<RectTransformation> expand_rect_for_next_frame_;

  // objects which are owned from here but should be refactored to be owned by the ImageToTensorCalculatorCore objects themselves ...
  std::unique_ptr<ImageToTensorConverter> palm_detection_gpu_converter_;  // must have lifetime equal to image_to_palm_detection_input_, or be refactored to be owned by it
  std::unique_ptr<ImageToTensorConverter> palm_detection_cpu_converter_;  // must have lifetime equal to image_to_palm_detection_input_, or be refactored to be owned by it
  std::unique_ptr<ImageToTensorConverter> landmarks_inference_gpu_converter_;  // must have lifetime equal to sub_image_for_landmarks_inference_extractor_, or refactored to be owned by it
  std::unique_ptr<ImageToTensorConverter> landmarks_inference_cpu_converter_;  // must have lifetime equal to sub_image_for_landmarks_inference_extractor_, or refactored to be owned by it

  // state which replaces the former passing of information by
  // a framework loopback to the pipeline's head
  std::vector<NormalizedRect> hand_rects_from_previous_frame_;

  uint32_t call_counter_ = 0;

  // debug printing methods
  static void sub_image_for_landmarks_inference_debug_logging(api2::ImageToTensorCoreResult *extracted_sub_image_struct);
  static void sub_image_padding_debug_logging(api2::ImageToTensorCoreResult* extracted_sub_image_struct);
  static void landmarks_inference_debug_logging(std::vector<Tensor> landmarks_inference_output_tensors);


 };

}
#endif