#ifndef MEDIAPIPE_LIBERATED_FACE_TRACKING_H_
#define MEDIAPIPE_LIBERATED_FACE_TRACKING_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "mediapipe/calculators/tensor/detections_extraction.h"
#include "mediapipe/calculators/tensor/image_to_tensor_calculator_core.h"
#include "mediapipe/calculators/tensor/model_inference.h"
#include "mediapipe/calculators/tensor/tensors_to_landmarks_calculator_core.h"
#include "mediapipe/calculators/util/detections_to_rects_calculator_core.h"
#include "mediapipe/calculators/util/rect_transformation_calculator_core.h"
#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/framework/memory_manager.h"

namespace hand_tracking_mp_lean {

// Runtime choices corresponding to the side packets and calculator parameters
// of the v0.10.13 FaceLandmarkFrontCpu graph.
struct FaceTrackingOptions {
  std::uint32_t max_faces = 1;
  bool use_previous_landmarks = true;
  bool with_attention = false;
  float min_detection_confidence = 0.5f;
  float min_tracking_confidence = 0.5f;
  int xnnpack_num_threads = 1;
};

// Result for one input image. Face indices agree across the first three fields
// within this result, but are not persistent identities across frames.
struct ImageFaceTrackingResult {
  std::vector<NormalizedLandmarkList> face_landmarks;
  std::vector<float> face_presence_scores;
  std::vector<NormalizedRect> face_rects_from_landmarks;

  // Detection-derived diagnostics are populated only on frames where the
  // detector ran. Tracking normally skips detection while enough previous
  // landmark-derived ROIs remain valid.
  bool face_detector_ran = false;
  std::vector<Detection> face_detections;
  std::vector<NormalizedRect> face_rects_from_detections;
};

// Direct, CalculatorGraph-free implementation of the v0.10.13
// FaceLandmarkFrontCpu graph. It retains MediaPipe-derived low-level image,
// tensor, protobuf, preprocessing, and TFLite/XNNPACK components, exactly as
// HandTrackingCore does.
class FaceTrackingCore {
 public:
  explicit FaceTrackingCore(
      FaceTrackingOptions options = {},
      const std::string* models_path = nullptr);
  ~FaceTrackingCore() = default;

  FaceTrackingCore(const FaceTrackingCore&) = delete;
  FaceTrackingCore& operator=(const FaceTrackingCore&) = delete;
  FaceTrackingCore(FaceTrackingCore&&) = delete;
  FaceTrackingCore& operator=(FaceTrackingCore&&) = delete;

  [[nodiscard]] absl::StatusOr<std::unique_ptr<ImageFaceTrackingResult>>
  Process(const std::shared_ptr<const Image>& image);

  void Reset();

 private:
  struct LandmarkInference {
    NormalizedLandmarkList landmarks;
    float presence_score = 0.0f;
  };

  absl::StatusOr<std::vector<NormalizedRect>> DetectFaces(
      const Image& image, ImageFaceTrackingResult* result);
  absl::StatusOr<std::optional<LandmarkInference>> InferLandmarks(
      const Image& image, const NormalizedRect& face_rect);
  absl::StatusOr<NormalizedRect> FaceRectFromLandmarks(
      const NormalizedLandmarkList& landmarks, int image_width,
      int image_height) const;

  FaceTrackingOptions options_;
  MemoryManager memory_manager_;

  // Converter lifetimes must cover the ImageToTensorCalculatorCore instances,
  // which retain references to these unique_ptr slots.
  std::unique_ptr<ImageToTensorConverter> detector_gpu_converter_;
  std::unique_ptr<ImageToTensorConverter> detector_cpu_converter_;
  std::unique_ptr<ImageToTensorConverter> landmark_gpu_converter_;
  std::unique_ptr<ImageToTensorConverter> landmark_cpu_converter_;

  std::unique_ptr<api2::ImageToTensorCalculatorCore>
      image_to_detector_input_;
  std::unique_ptr<api2::ModelInference> detector_inference_;
  std::unique_ptr<api2::DetectionsExtractionAndFiltering>
      detector_output_decoder_;
  std::unique_ptr<DetectionsToOrientedRects>
      detection_to_oriented_face_rect_;
  std::unique_ptr<RectTransformation> detection_face_rect_expander_;

  std::unique_ptr<api2::ImageToTensorCalculatorCore>
      face_roi_to_landmark_input_;
  std::unique_ptr<api2::ModelInference> landmark_inference_;
  std::unique_ptr<api2::TensorsToLandmarksCore> mesh_decoder_;
  std::unique_ptr<api2::TensorsToLandmarksCore> lips_decoder_;
  std::unique_ptr<api2::TensorsToLandmarksCore> left_eye_decoder_;
  std::unique_ptr<api2::TensorsToLandmarksCore> right_eye_decoder_;
  std::unique_ptr<api2::TensorsToLandmarksCore> iris_decoder_;
  std::unique_ptr<DetectionsToOrientedRects>
      landmarks_to_oriented_face_rect_;
  std::unique_ptr<RectTransformation> landmark_face_rect_expander_;

  std::vector<NormalizedRect> face_rects_from_previous_frame_;
};

}  // namespace hand_tracking_mp_lean

#endif  // MEDIAPIPE_LIBERATED_FACE_TRACKING_H_
