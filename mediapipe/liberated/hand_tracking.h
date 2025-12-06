#ifndef MEDIAPIPE_LIBERATED_H
#define MEDIAPIPE_LIBERATED_H

#include <memory>
#include <optional>

#include "mediapipe/framework/memory_manager.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/formats/classification.pb.h"

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
#include "mediapipe/calculators/tensor/tensors_to_landmarks_calculator.pb.h"
#include "mediapipe/calculators/tensor/tensors_to_classification_calculator.pb.h"
#include "mediapipe/calculators/tensor/tensors_to_landmarks_calculator_core.h"
#include "mediapipe/calculators/util/landmark_letterbox_removal_calculator_core.h"
#include "mediapipe/calculators/util/landmark_projection_calculator_core.h"
#include "mediapipe/calculators/tensor/tensors_to_world_landmarks_calculator_core.h"
#include "mediapipe/calculators/util/world_landmark_projection_calculator_core.h"
#include "mediapipe/modules/hand_landmark/calculators/hand_landmarks_to_rect_calculator_core.h"

namespace hand_tracking_mp_lean {
 class DetectionsToOrientedRects;  // forward declaration hack that will ultimately be unnecessary

 /// additional debug/analysis info per palm detection used to derive a hand rect
 struct DetectionDetails {
  /// A full image-normalized rectangle.
  /// Coordinates are normalized to image size: x in [0,1] left→right, y in [0,1] top→bottom.
  /// Rotation is in radians, counter-clockwise, normalized to [-pi, pi).
  struct RectValues {
    float x_center;
    float y_center;
    float width;
    float height;
    float rotation;
  };
  float palm_detection_score;  // confidence score from palm detection (SSD) for this detection
  float hand_presence_score;   // presence score from the landmarks inference output corresponding to this hand
  // Each rectangle is now optional and will only be populated if such a rectangle was actually built.
  std::optional<RectValues> detected;                 // initial axis-aligned palm detection rectangle
  std::optional<RectValues> oriented;                 // oriented rectangle derived from the detection (pre-expansion)
  std::optional<RectValues> expanded;                 // rectangle after expansion used for landmarks inference input
  std::optional<RectValues> hand_rect_for_next_frame; // rectangle predicted from current landmarks for use in next frame (after its own expansion step)
  };

 /// hand tracking result type for a single input image used as hand tracking input.
 ///
 /// it is currently structured such that each sub-field is populated per hand graduating the hand tracking processing in a call to HandTrackingCore::Process,
 /// so to get just one hand's results out of potentially many ― you have to collect its elements from across each of the sub-fields ― i.e. for the 3rd hand,
 /// you have to take viewport_landmarkss[2], object_landmarkss[2], handedness_classifications[2].
 ///
 /// this just follows the original pipeline's output shape and can be easily flipped to return a vector of hands.
 /// when we start tracking hand identity, this structure is even more bound to change.
 struct ImageHandTrackingAndInferenceResult {
  std::unique_ptr<std::vector<NormalizedLandmarkList>> viewport_landmarkss;
  std::unique_ptr<std::vector<LandmarkList>> object_landmarkss;
  std::unique_ptr<std::vector<ClassificationList>> handedness_classifications;
  std::unique_ptr<std::vector<DetectionDetails>> detection_details;
 };

 class HandTrackingCore {
 public:

  explicit HandTrackingCore(uint32_t max_hands_to_track, const std::string* models_path = nullptr);

  ~HandTrackingCore() = default;

  // Non-copyable, movable.
  HandTrackingCore(const HandTrackingCore&) = delete;
  HandTrackingCore& operator=(const HandTrackingCore&) = delete;
  HandTrackingCore(HandTrackingCore&&) = default;
  HandTrackingCore& operator=(HandTrackingCore&&) = default;

  [[nodiscard]] absl::StatusOr<std::unique_ptr<ImageHandTrackingAndInferenceResult>> Process(const std::shared_ptr<const hand_tracking_mp_lean::Image>& image);

 private:

  // meta-parameters should ultimately go here.
  // values we pass to below implementations which must mirror how
  // neural network models have been trained should not really be
  // moved here as they cannot be fiddled.
  float hand_presence_in_landmarks_inference_threshold_ = 0.5f;
  uint32_t max_hands_to_track_;

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
  static void image_debug_logging(api2::ImageToTensorCoreResult *image_struct);
  static void sub_image_for_landmarks_inference_debug_logging(api2::ImageToTensorCoreResult *extracted_sub_image_struct);
  static void sub_image_padding_debug_logging(api2::ImageToTensorCoreResult* extracted_sub_image_struct);
  static void landmarks_inference_debug_logging(std::vector<Tensor> landmarks_inference_output_tensors);


 };

}
#endif