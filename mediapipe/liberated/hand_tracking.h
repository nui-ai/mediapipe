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

 // forward declaration which can be replaced by more build wiring
 class DetectionsToOrientedRects;

 /// rectangle normalized back to the original image coordinates:
 /// position & size are normalized to image size: x in [0,1] left→right, y in [0,1] top→bottom.
 /// rotation is in radians, counter-clockwise, normalized to [-pi, pi).
 struct RectGeometry {
  float x_center;
  float y_center;
  float width;
  float height;
  float rotation;
 };

 /// additional debug/analysis info per palm detection used to derive a hand rect
 struct DetectionInformation {

  // confidence score from palm detection (SSD) for this detection
  float palm_detection_score = 0.0f;

  // initial axes-aligned palm detection rectangle
  std::optional<RectGeometry> detected;

  // oriented palm rectangle derived from it
  std::optional<RectGeometry> oriented;

  // rectangle heuristically expanded to surely capture the entire hand, intended for the landmarks inference step
  std::optional<RectGeometry> expanded;
};

 /// hand tracking result type for a single input image comprising both bottom-line data,
 /// and process-transparency data.
 ///
 /// 1. Per Graduated Hand Information (this is bottom-line output data)
 ///
 ///    this is information from the current call to HandTracking::Process, per hand which made it
 ///    all-the-way through the landmarks inference model invocation stage of HandTracking::Process,
 ///
 ///      • its presence score from that model
 ///      • its viewport landmarks from that model
 ///      • its object landmarks from that model
 ///      • its handedness classification from that model
 ///      • a rectangle heuristically fit to the hand from its landmarks (chiefly used for constructing the next field)
 ///      • a rectangle heuristically derived for the capture of the same hand on the next call
 ///
 ///    this per hand information is currently structured such that each one of the above information constituents is populated per hand,
 ///    the order of the hands being the same across the different fields to which these information constituents from above split to.
 ///    so the hands are implicitly indexed by their order in the sub-field vectors, and this indexing is uniform across those fields.
 ///
 ///    this shape follows the original pipeline's output shape and can be easily flipped and "transposed" to return a vector of hands.
 ///    when we start tracking hand identity, this structure is even more bound to change.
 ///
 /// 2. Palm Detection Derived Information (this is judicious process-transparency data)
 ///
 ///    this is extra information per palm detection from the current call to HandTracking::Process,
 ///    whereby the palm detection step yielding this information is not always employed by
 ///    HandTracking::Process, hence this information may be either present or abscent.
 ///
 ///    these detections are not one-to-one to the per-hand information, not in association by any indexing
 ///    and not even in total quantity! this is because detections are not always carried out, AND also because
 ///    the detections when carried out, are merged-and-filtered against and with the rectangles heuristically
 ///    derived above from landmarks inference, before HandTracking::Process decides which ones to apply
 ///    hand inference to. hence they are not alignable to the resulting per-hand information other
 ///    than within the flow implemented within HandTracking::Process itself.
 ///
 ///    during foture work incorporating hand identity in the tracking, such associations may arise
 ///    as applicable to any specific flow trajectory being implemented (by research) for
 ///    identity-enabled tracking.
 ///
 ///    this is of course partial tracking information, since it does not go into exporting the various
 ///    filtering stages which the detections undergo within and after the detection inference model's invocation!
 ///
 struct ImageHandTrackingResult {

  ImageHandTrackingResult()
    : viewport_landmarkss(std::make_unique<std::vector<NormalizedLandmarkList>>()),
      object_landmarkss(std::make_unique<std::vector<LandmarkList>>()),
      handedness_classifications(std::make_unique<std::vector<ClassificationList>>()),
      landmarks_derived_hand_presence_scores(std::make_unique<std::vector<float>>()),
      landmarks_based_rectangles(std::make_unique<std::vector<RectGeometry>>()),
      detections_information(std::make_unique<std::vector<DetectionInformation>>()) {}

  // bottom line output
  std::unique_ptr<std::vector<NormalizedLandmarkList>> viewport_landmarkss;
  std::unique_ptr<std::vector<LandmarkList>> object_landmarkss;
  std::unique_ptr<std::vector<ClassificationList>> handedness_classifications;
  std::unique_ptr<std::vector<float>> landmarks_derived_hand_presence_scores;

  // judicious (i.e. partial, selective) process-transparency information
  std::unique_ptr<std::vector<RectGeometry>> landmarks_based_rectangles;
  std::unique_ptr<std::vector<DetectionInformation>> detections_information;
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

  [[nodiscard]] absl::StatusOr<std::unique_ptr<ImageHandTrackingResult>> Process(const std::shared_ptr<const hand_tracking_mp_lean::Image>& image);

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
  std::unique_ptr<RectTransformation> landmarks_derived_hand_rect_expander_;

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