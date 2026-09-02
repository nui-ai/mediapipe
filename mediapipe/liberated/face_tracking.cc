#include "mediapipe/liberated/face_tracking.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <list>
#include <stdexcept>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/types/optional.h"
#include "mediapipe/calculators/tensor/image_to_tensor_calculator.pb.h"
#include "mediapipe/calculators/tensor/image_to_tensor_utils.h"
#include "mediapipe/calculators/tensor/tensors_to_floats_calculator.pb.h"
#include "mediapipe/calculators/tensor/tensors_to_floats_calculator_core.h"
#include "mediapipe/calculators/tensor/tensors_to_landmarks_calculator.pb.h"
#include "mediapipe/calculators/util/landmark_projection_calculator_core.h"
#include "mediapipe/calculators/util/association_calculator_core.h"
#include "mediapipe/calculators/util/rect_transformation_calculator.pb.h"
#include "mediapipe/framework/formats/location_data.pb.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status_macros.h"

namespace hand_tracking_mp_lean {
namespace {

constexpr int kDetectorInputSize = 128;
constexpr int kLandmarkInputSize = 192;
constexpr int kMeshLandmarkCount = 468;
constexpr int kAttentionLandmarkCount = 478;
constexpr int kLipsLandmarkCount = 80;
constexpr int kEyeLandmarkCount = 71;
constexpr int kIrisLandmarkCount = 5;

constexpr std::array<int, kLipsLandmarkCount> kLipsMapping = {
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
    185, 40, 39, 37, 0, 267, 269, 270, 409,
    78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
    191, 80, 81, 82, 13, 312, 311, 310, 415,
    76, 77, 90, 180, 85, 16, 315, 404, 320, 307, 306,
    184, 74, 73, 72, 11, 302, 303, 304, 408,
    62, 96, 89, 179, 86, 15, 316, 403, 319, 325, 292,
    183, 42, 41, 38, 12, 268, 271, 272, 407};

constexpr std::array<int, kEyeLandmarkCount> kLeftEyeMapping = {
    33, 7, 163, 144, 145, 153, 154, 155, 133,
    246, 161, 160, 159, 158, 157, 173,
    130, 25, 110, 24, 23, 22, 26, 112, 243,
    247, 30, 29, 27, 28, 56, 190,
    226, 31, 228, 229, 230, 231, 232, 233, 244,
    113, 225, 224, 223, 222, 221, 189,
    35, 124, 46, 53, 52, 65,
    143, 111, 117, 118, 119, 120, 121, 128, 245,
    156, 70, 63, 105, 66, 107, 55, 193};

constexpr std::array<int, kEyeLandmarkCount> kRightEyeMapping = {
    263, 249, 390, 373, 374, 380, 381, 382, 362,
    466, 388, 387, 386, 385, 384, 398,
    359, 255, 339, 254, 253, 252, 256, 341, 463,
    467, 260, 259, 257, 258, 286, 414,
    446, 261, 448, 449, 450, 451, 452, 453, 464,
    342, 445, 444, 443, 442, 441, 413,
    265, 353, 276, 283, 282, 295,
    372, 340, 346, 347, 348, 349, 350, 357, 465,
    383, 300, 293, 334, 296, 336, 285, 417};

constexpr std::array<int, kIrisLandmarkCount> kLeftIrisMapping = {
    468, 469, 470, 471, 472};
constexpr std::array<int, kIrisLandmarkCount> kRightIrisMapping = {
    473, 474, 475, 476, 477};

constexpr std::array<int, 16> kLeftIrisZAverageIndices = {
    33, 7, 163, 144, 145, 153, 154, 155,
    133, 246, 161, 160, 159, 158, 157, 173};
constexpr std::array<int, 16> kRightIrisZAverageIndices = {
    263, 249, 390, 373, 374, 380, 381, 382,
    362, 466, 388, 387, 386, 385, 384, 398};

float ProjectX(float x, float y, const std::array<float, 16>& matrix) {
  return x * matrix[0] + y * matrix[1] + matrix[3];
}

float ProjectY(float x, float y, const std::array<float, 16>& matrix) {
  return x * matrix[4] + y * matrix[5] + matrix[7];
}

absl::Status ProjectDetection(const std::array<float, 16>& matrix,
                              Detection* detection) {
  auto* location = detection->mutable_location_data();
  RET_CHECK_EQ(location->format(), LocationData::RELATIVE_BOUNDING_BOX);

  for (int i = 0; i < location->relative_keypoints_size(); ++i) {
    auto* keypoint = location->mutable_relative_keypoints(i);
    const float x = keypoint->x();
    const float y = keypoint->y();
    keypoint->set_x(ProjectX(x, y, matrix));
    keypoint->set_y(ProjectY(x, y, matrix));
  }

  auto* box = location->mutable_relative_bounding_box();
  const float xmin = box->xmin();
  const float ymin = box->ymin();
  const float xmax = xmin + box->width();
  const float ymax = ymin + box->height();
  const std::array<std::pair<float, float>, 4> corners = {{
      {xmin, ymin}, {xmax, ymin}, {xmax, ymax}, {xmin, ymax}}};

  float projected_xmin = std::numeric_limits<float>::max();
  float projected_ymin = std::numeric_limits<float>::max();
  float projected_xmax = std::numeric_limits<float>::lowest();
  float projected_ymax = std::numeric_limits<float>::lowest();
  for (const auto& [x, y] : corners) {
    const float projected_x = ProjectX(x, y, matrix);
    const float projected_y = ProjectY(x, y, matrix);
    projected_xmin = std::min(projected_xmin, projected_x);
    projected_ymin = std::min(projected_ymin, projected_y);
    projected_xmax = std::max(projected_xmax, projected_x);
    projected_ymax = std::max(projected_ymax, projected_y);
  }
  box->set_xmin(projected_xmin);
  box->set_ymin(projected_ymin);
  box->set_width(projected_xmax - projected_xmin);
  box->set_height(projected_ymax - projected_ymin);
  return absl::OkStatus();
}

absl::StatusOr<std::vector<Detection>> ProjectDetections(
    const std::vector<Detection>& detections,
    const std::array<float, 16>& matrix) {
  std::vector<Detection> projected;
  projected.reserve(detections.size());
  for (const auto& detection : detections) {
    Detection output = detection;
    MP_RETURN_IF_ERROR(ProjectDetection(matrix, &output));
    projected.push_back(std::move(output));
  }
  return projected;
}

absl::StatusOr<Detection> LandmarksToDetection(
    const NormalizedLandmarkList& landmarks) {
  RET_CHECK_GT(landmarks.landmark_size(), 0);

  Detection detection;
  auto* location = detection.mutable_location_data();
  float xmin = std::numeric_limits<float>::max();
  float ymin = std::numeric_limits<float>::max();
  float xmax = std::numeric_limits<float>::lowest();
  float ymax = std::numeric_limits<float>::lowest();

  for (const auto& landmark : landmarks.landmark()) {
    xmin = std::min(xmin, landmark.x());
    ymin = std::min(ymin, landmark.y());
    xmax = std::max(xmax, landmark.x());
    ymax = std::max(ymax, landmark.y());
    auto* keypoint = location->add_relative_keypoints();
    keypoint->set_x(landmark.x());
    keypoint->set_y(landmark.y());
  }

  location->set_format(LocationData::RELATIVE_BOUNDING_BOX);
  auto* box = location->mutable_relative_bounding_box();
  box->set_xmin(xmin);
  box->set_ymin(ymin);
  box->set_width(xmax - xmin);
  box->set_height(ymax - ymin);
  return detection;
}

absl::StatusOr<NormalizedLandmarkList> DecodeLandmarkTensor(
    Tensor tensor, api2::TensorsToLandmarksCore* decoder) {
  std::vector<Tensor> tensors;
  tensors.emplace_back(std::move(tensor));
  NormalizedLandmarkList landmarks;
  MP_RETURN_IF_ERROR(
      decoder->OutputTensorsToLandmarks(tensors, &landmarks));
  return landmarks;
}

absl::StatusOr<float> DecodePresenceTensor(Tensor tensor) {
  std::vector<Tensor> tensors;
  tensors.emplace_back(std::move(tensor));
  TensorsToFloatsCalculatorOptions options;
  options.set_activation(TensorsToFloatsCalculatorOptions::SIGMOID);
  auto decoded =
      tensors_to_floats_calculator_core::HandPresenceExtract(tensors, options);
  MP_RETURN_IF_ERROR(decoded.status);
  RET_CHECK_EQ(decoded.num_values, 1);
  return decoded.output_floats->at(0);
}

template <std::size_t N>
absl::Status OverlayXY(
    const NormalizedLandmarkList& source,
    const std::array<int, N>& mapping,
    NormalizedLandmarkList* target) {
  RET_CHECK_EQ(source.landmark_size(), N);
  for (std::size_t i = 0; i < N; ++i) {
    RET_CHECK_LT(mapping[i], target->landmark_size());
    auto* output = target->mutable_landmark(mapping[i]);
    output->set_x(source.landmark(i).x());
    output->set_y(source.landmark(i).y());
  }
  return absl::OkStatus();
}

template <std::size_t N>
float AverageZ(const NormalizedLandmarkList& landmarks,
               const std::array<int, N>& indices) {
  double sum = 0.0;
  for (int index : indices) {
    sum += landmarks.landmark(index).z();
  }
  return static_cast<float>(sum / N);
}

absl::StatusOr<NormalizedLandmarkList> RefineAttentionLandmarks(
    const NormalizedLandmarkList& mesh,
    const NormalizedLandmarkList& lips,
    const NormalizedLandmarkList& left_eye,
    const NormalizedLandmarkList& right_eye,
    const NormalizedLandmarkList& left_iris,
    const NormalizedLandmarkList& right_iris) {
  RET_CHECK_EQ(mesh.landmark_size(), kMeshLandmarkCount);
  RET_CHECK_EQ(lips.landmark_size(), kLipsLandmarkCount);
  RET_CHECK_EQ(left_eye.landmark_size(), kEyeLandmarkCount);
  RET_CHECK_EQ(right_eye.landmark_size(), kEyeLandmarkCount);
  RET_CHECK_EQ(left_iris.landmark_size(), kIrisLandmarkCount);
  RET_CHECK_EQ(right_iris.landmark_size(), kIrisLandmarkCount);

  NormalizedLandmarkList refined;
  for (int i = 0; i < kAttentionLandmarkCount; ++i) {
    refined.add_landmark();
  }

  // The legacy LandmarksRefinementCalculator copies mesh X/Y/Z first, then
  // replaces only X/Y for lips and eyes.
  for (int i = 0; i < kMeshLandmarkCount; ++i) {
    auto* output = refined.mutable_landmark(i);
    output->set_x(mesh.landmark(i).x());
    output->set_y(mesh.landmark(i).y());
    output->set_z(mesh.landmark(i).z());
  }
  MP_RETURN_IF_ERROR(OverlayXY(lips, kLipsMapping, &refined));
  MP_RETURN_IF_ERROR(OverlayXY(left_eye, kLeftEyeMapping, &refined));
  MP_RETURN_IF_ERROR(OverlayXY(right_eye, kRightEyeMapping, &refined));

  const float left_iris_z =
      AverageZ(refined, kLeftIrisZAverageIndices);
  const float right_iris_z =
      AverageZ(refined, kRightIrisZAverageIndices);
  MP_RETURN_IF_ERROR(OverlayXY(left_iris, kLeftIrisMapping, &refined));
  MP_RETURN_IF_ERROR(OverlayXY(right_iris, kRightIrisMapping, &refined));
  for (int index : kLeftIrisMapping) {
    refined.mutable_landmark(index)->set_z(left_iris_z);
  }
  for (int index : kRightIrisMapping) {
    refined.mutable_landmark(index)->set_z(right_iris_z);
  }
  return refined;
}

}  // namespace

FaceTrackingCore::FaceTrackingCore(
    FaceTrackingOptions options, const std::string* assets_path)
    : options_(std::move(options)) {
  if (options_.max_faces == 0) {
    throw std::invalid_argument("max_faces must be greater than zero");
  }
  if (!(options_.min_detection_confidence > 0.0f &&
        options_.min_detection_confidence < 1.0f)) {
    throw std::invalid_argument(
        "min_detection_confidence must be between zero and one");
  }
  if (!(options_.min_tracking_confidence > 0.0f &&
        options_.min_tracking_confidence < 1.0f)) {
    throw std::invalid_argument(
        "min_tracking_confidence must be between zero and one");
  }

  if (options_.estimate_pose) {
    auto estimator = FaceGeometryEstimator::Create(
        options_.vertical_fov_degrees, assets_path);
    if (!estimator.ok()) {
      throw std::runtime_error(estimator.status().ToString());
    }
    face_geometry_estimator_ = std::move(estimator.value());
  }

  ImageToTensorCalculatorOptions detector_options;
  detector_options.set_output_tensor_width(kDetectorInputSize);
  detector_options.set_output_tensor_height(kDetectorInputSize);
  detector_options.set_keep_aspect_ratio(true);
  detector_options.mutable_output_tensor_float_range()->set_min(-1.0f);
  detector_options.mutable_output_tensor_float_range()->set_max(1.0f);
  detector_options.set_border_mode(
      ImageToTensorCalculatorOptions::BORDER_ZERO);
  const auto detector_params = GetOutputTensorParams(detector_options);
  image_to_detector_input_ =
      std::make_unique<api2::ImageToTensorCalculatorCore>(
          detector_options, kDetectorInputSize, kDetectorInputSize,
          detector_params, detector_gpu_converter_, detector_cpu_converter_,
          &memory_manager_);

  detector_inference_ = std::make_unique<api2::ModelInference>(
      "mediapipe/modules/face_detection/face_detection_short_range.tflite",
      assets_path, options_.xnnpack_num_threads);
  detector_output_decoder_ =
      std::make_unique<api2::DetectionsExtractionAndFiltering>(
          options_.min_detection_confidence,
          api2::DetectionModel::kFaceShortRange);
  detection_to_oriented_face_rect_ =
      std::make_unique<DetectionsToOrientedRects>(0, 1, 0.0f);

  RectTransformationCalculatorOptions detected_face_expansion;
  detected_face_expansion.set_scale_x(1.5f);
  detected_face_expansion.set_scale_y(1.5f);
  detected_face_expansion.set_square_long(true);
  detection_face_rect_expander_ =
      std::make_unique<RectTransformation>(detected_face_expansion);

  ImageToTensorCalculatorOptions landmark_options;
  landmark_options.set_output_tensor_width(kLandmarkInputSize);
  landmark_options.set_output_tensor_height(kLandmarkInputSize);
  landmark_options.mutable_output_tensor_float_range()->set_min(0.0f);
  landmark_options.mutable_output_tensor_float_range()->set_max(1.0f);
  const auto landmark_params = GetOutputTensorParams(landmark_options);
  face_roi_to_landmark_input_ =
      std::make_unique<api2::ImageToTensorCalculatorCore>(
          landmark_options, kLandmarkInputSize, kLandmarkInputSize,
          landmark_params, landmark_gpu_converter_, landmark_cpu_converter_,
          &memory_manager_);

  const std::string landmark_model =
      options_.with_attention
          ? "mediapipe/modules/face_landmark/"
            "face_landmark_with_attention.tflite"
          : "mediapipe/modules/face_landmark/face_landmark.tflite";
  landmark_inference_ = std::make_unique<api2::ModelInference>(
      landmark_model, assets_path, options_.xnnpack_num_threads);

  TensorsToLandmarksCalculatorOptions decode_options;
  mesh_decoder_ = std::make_unique<api2::TensorsToLandmarksCore>(
      kLandmarkInputSize, kLandmarkInputSize,
      decode_options.visibility_activation(),
      decode_options.presence_activation(), 1.0f, kMeshLandmarkCount);
  if (options_.with_attention) {
    lips_decoder_ = std::make_unique<api2::TensorsToLandmarksCore>(
        kLandmarkInputSize, kLandmarkInputSize,
        decode_options.visibility_activation(),
        decode_options.presence_activation(), 1.0f, kLipsLandmarkCount);
    left_eye_decoder_ = std::make_unique<api2::TensorsToLandmarksCore>(
        kLandmarkInputSize, kLandmarkInputSize,
        decode_options.visibility_activation(),
        decode_options.presence_activation(), 1.0f, kEyeLandmarkCount);
    right_eye_decoder_ = std::make_unique<api2::TensorsToLandmarksCore>(
        kLandmarkInputSize, kLandmarkInputSize,
        decode_options.visibility_activation(),
        decode_options.presence_activation(), 1.0f, kEyeLandmarkCount);
    iris_decoder_ = std::make_unique<api2::TensorsToLandmarksCore>(
        kLandmarkInputSize, kLandmarkInputSize,
        decode_options.visibility_activation(),
        decode_options.presence_activation(), 1.0f, kIrisLandmarkCount);
  }

  landmarks_to_oriented_face_rect_ =
      std::make_unique<DetectionsToOrientedRects>(33, 263, 0.0f);
  RectTransformationCalculatorOptions tracked_face_expansion;
  tracked_face_expansion.set_scale_x(1.5f);
  tracked_face_expansion.set_scale_y(1.5f);
  tracked_face_expansion.set_square_long(true);
  landmark_face_rect_expander_ =
      std::make_unique<RectTransformation>(tracked_face_expansion);
}

void FaceTrackingCore::Reset() {
  face_rects_from_previous_frame_.clear();
}

absl::StatusOr<std::vector<NormalizedRect>> FaceTrackingCore::DetectFaces(
    const Image& image, ImageFaceTrackingResult* result) {
  result->face_detector_ran = true;
  api2::ImageToTensorCoreResult detector_input;
  MP_RETURN_IF_ERROR(image_to_detector_input_->Process(
      image, absl::nullopt, &detector_input));

  MP_ASSIGN_OR_RETURN(
      std::vector<Tensor> detector_outputs,
      detector_inference_->Process(MakeTensorSpan(detector_input.tensors)));
  MP_ASSIGN_OR_RETURN(
      auto decoded,
      detector_output_decoder_->Extract(detector_outputs));
  MP_ASSIGN_OR_RETURN(
      auto filtered,
      detector_output_decoder_->Filter(*decoded));
  MP_ASSIGN_OR_RETURN(
      std::vector<Detection> projected,
      ProjectDetections(*filtered, detector_input.matrix));

  if (projected.size() > options_.max_faces) {
    projected.resize(options_.max_faces);
  }
  result->face_detections = projected;

  std::vector<NormalizedRect> face_rects;
  std::vector<Rect> unused_rects;
  MP_RETURN_IF_ERROR(
      detection_to_oriented_face_rect_->OrientedRectsFromDetections(
          projected, std::make_pair(image.width(), image.height()),
          &face_rects, &unused_rects));
  for (auto& face_rect : face_rects) {
    detection_face_rect_expander_->ExpandNormalizedRect(
        &face_rect, image.width(), image.height());
  }
  result->face_rects_from_detections = face_rects;
  return face_rects;
}

absl::StatusOr<std::optional<FaceTrackingCore::LandmarkModelResult>>
FaceTrackingCore::InferLandmarks(
    const Image& image, const NormalizedRect& face_rect) {
  api2::ImageToTensorCoreResult landmark_input;
  MP_RETURN_IF_ERROR(face_roi_to_landmark_input_->Process(
      image, face_rect, &landmark_input));

  MP_ASSIGN_OR_RETURN(
      std::vector<Tensor> outputs,
      landmark_inference_->Process(MakeTensorSpan(landmark_input.tensors)));
  const int expected_output_count = options_.with_attention ? 7 : 2;
  RET_CHECK_EQ(outputs.size(), expected_output_count);

  MP_ASSIGN_OR_RETURN(
      const float presence_score,
      DecodePresenceTensor(std::move(outputs[expected_output_count - 1])));
  if (presence_score < options_.min_tracking_confidence) {
    return std::nullopt;
  }

  NormalizedLandmarkList roi_landmarks;
  if (!options_.with_attention) {
    MP_ASSIGN_OR_RETURN(
        roi_landmarks,
        DecodeLandmarkTensor(std::move(outputs[0]), mesh_decoder_.get()));
  } else {
    MP_ASSIGN_OR_RETURN(
        auto mesh,
        DecodeLandmarkTensor(std::move(outputs[0]), mesh_decoder_.get()));
    MP_ASSIGN_OR_RETURN(
        auto lips,
        DecodeLandmarkTensor(std::move(outputs[1]), lips_decoder_.get()));
    MP_ASSIGN_OR_RETURN(
        auto left_eye,
        DecodeLandmarkTensor(std::move(outputs[2]), left_eye_decoder_.get()));
    MP_ASSIGN_OR_RETURN(
        auto right_eye,
        DecodeLandmarkTensor(std::move(outputs[3]), right_eye_decoder_.get()));
    MP_ASSIGN_OR_RETURN(
        auto left_iris,
        DecodeLandmarkTensor(std::move(outputs[4]), iris_decoder_.get()));
    MP_ASSIGN_OR_RETURN(
        auto right_iris,
        DecodeLandmarkTensor(std::move(outputs[5]), iris_decoder_.get()));
    MP_ASSIGN_OR_RETURN(
        roi_landmarks,
        RefineAttentionLandmarks(
            mesh, lips, left_eye, right_eye, left_iris, right_iris));
  }

  LandmarkModelResult inference;
  inference.presence_score = presence_score;
  ToViewportCoordinates(
      roi_landmarks, &face_rect, &inference.landmarks);
  return inference;
}

absl::StatusOr<NormalizedRect> FaceTrackingCore::FaceRectFromLandmarks(
    const NormalizedLandmarkList& landmarks, int image_width,
    int image_height) const {
  MP_ASSIGN_OR_RETURN(Detection detection, LandmarksToDetection(landmarks));
  std::vector<Detection> detections;
  detections.push_back(std::move(detection));
  std::vector<NormalizedRect> face_rects;
  std::vector<Rect> unused_rects;
  MP_RETURN_IF_ERROR(
      landmarks_to_oriented_face_rect_->OrientedRectsFromDetections(
          detections, std::make_pair(image_width, image_height),
          &face_rects, &unused_rects));
  RET_CHECK_EQ(face_rects.size(), 1);
  landmark_face_rect_expander_->ExpandNormalizedRect(
      &face_rects[0], image_width, image_height);
  return face_rects[0];
}

absl::StatusOr<std::unique_ptr<ImageFaceTrackingResult>>
FaceTrackingCore::Process(const std::shared_ptr<const Image>& image) {
  RET_CHECK(image != nullptr);
  RET_CHECK_GT(image->width(), 0);
  RET_CHECK_GT(image->height(), 0);

  auto result = std::make_unique<ImageFaceTrackingResult>();
  std::vector<NormalizedRect> detected_face_rects;
  std::vector<NormalizedRect> previous_face_rects;
  if (options_.use_previous_landmarks) {
    previous_face_rects = face_rects_from_previous_frame_;
  }

  if (!options_.use_previous_landmarks ||
      previous_face_rects.size() < options_.max_faces) {
    MP_ASSIGN_OR_RETURN(
        detected_face_rects, DetectFaces(*image, result.get()));
  }

  MP_ASSIGN_OR_RETURN(
      std::list<NormalizedRect> merged_face_rects,
      IouFilterMerge(
          detected_face_rects, previous_face_rects, 0.5f));

  face_rects_from_previous_frame_.clear();
  for (const auto& face_rect : merged_face_rects) {
    MP_ASSIGN_OR_RETURN(
        auto inference, InferLandmarks(*image, face_rect));
    if (!inference.has_value()) {
      continue;
    }

    MP_ASSIGN_OR_RETURN(
        NormalizedRect next_face_rect,
        FaceRectFromLandmarks(
            inference->landmarks, image->width(), image->height()));
    std::optional<FacePoseTransform> pose_transform;
    if (face_geometry_estimator_ != nullptr) {
      MP_ASSIGN_OR_RETURN(
          pose_transform,
          face_geometry_estimator_->Estimate(
              inference->landmarks, image->width(), image->height()));
    }

    FaceInference face;
    face.landmarks = std::move(inference->landmarks);
    face.presence_score = inference->presence_score;
    face.rect_from_landmarks = next_face_rect;
    face.pose_transform = std::move(pose_transform);
    result->faces.push_back(std::move(face));
    if (options_.use_previous_landmarks) {
      face_rects_from_previous_frame_.push_back(next_face_rect);
    }
  }

  return result;
}

}  // namespace hand_tracking_mp_lean
