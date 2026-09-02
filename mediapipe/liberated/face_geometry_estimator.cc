#include "mediapipe/liberated/face_geometry_estimator.h"

#include <algorithm>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "mediapipe/framework/formats/matrix_data.pb.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status_macros.h"
#include "mediapipe/tasks/cc/vision/face_geometry/libs/geometry_pipeline.h"
#include "mediapipe/tasks/cc/vision/face_geometry/libs/geometry_pipeline_metadata_loader.h"
#include "mediapipe/tasks/cc/vision/face_geometry/proto/environment.pb.h"
#include "mediapipe/tasks/cc/vision/face_geometry/proto/face_geometry.pb.h"

namespace hand_tracking_mp_lean {
namespace {

constexpr int kBaseLandmarkCount = 468;
constexpr int kAttentionLandmarkCount = 478;
constexpr char kGeometryMetadataPath[] =
    "mediapipe/tasks/cc/vision/face_geometry/data/"
    "geometry_pipeline_metadata_landmarks.binarypb";

// Resolves the geometry asset with the same optional runtime root used by the
// tracker model files.
std::string ResolveAssetPath(const std::string* assets_path,
                             const std::string& relative_path) {
  return assets_path == nullptr ? relative_path
                                : *assets_path + "/" + relative_path;
}

}  // namespace

class FaceGeometryEstimator::Impl {
 public:
  explicit Impl(
      std::unique_ptr<tasks::vision::face_geometry::GeometryPipeline> pipeline)
      : pipeline_(std::move(pipeline)) {}

  std::unique_ptr<tasks::vision::face_geometry::GeometryPipeline> pipeline_;
};

absl::StatusOr<std::unique_ptr<FaceGeometryEstimator>>
FaceGeometryEstimator::Create(float vertical_fov_degrees,
                              const std::string* assets_path) {
  // The binary geometry asset combines the 468-vertex canonical face, its mesh
  // topology and UV coordinates, and the smaller weighted landmark basis used
  // by the pose fit. Loading it here keeps those static inputs out of the
  // per-frame path.
  MP_ASSIGN_OR_RETURN(
      auto metadata,
      tasks::vision::face_geometry::ReadGeometryPipelineMetadata(
          ResolveAssetPath(assets_path, kGeometryMetadataPath)),
      _ << "Failed to read the canonical face and pose-fitting data!");

  tasks::vision::face_geometry::proto::Environment environment;
  environment.set_origin_point_location(
      tasks::vision::face_geometry::proto::TOP_LEFT_CORNER);
  // GeometryPipeline uses a virtual perspective camera to turn normalized
  // screen X/Y and relative Z into its runtime metric coordinate system. The
  // caller supplies vertical field of view; frame dimensions supplied to
  // Estimate later determine the corresponding horizontal extent.
  auto* camera = environment.mutable_perspective_camera();
  camera->set_vertical_fov_degrees(vertical_fov_degrees);
  camera->set_near(1.0f);
  camera->set_far(10000.0f);

  MP_ASSIGN_OR_RETURN(
      auto pipeline,
      tasks::vision::face_geometry::CreateGeometryPipeline(environment,
                                                           metadata),
      _ << "Failed to create face geometry pipeline!");
  return std::unique_ptr<FaceGeometryEstimator>(new FaceGeometryEstimator(
      std::make_unique<Impl>(std::move(pipeline))));
}

FaceGeometryEstimator::FaceGeometryEstimator(
    std::unique_ptr<Impl> implementation)
    : implementation_(std::move(implementation)) {}

FaceGeometryEstimator::~FaceGeometryEstimator() = default;

absl::StatusOr<std::optional<FacePoseTransform>>
FaceGeometryEstimator::Estimate(const NormalizedLandmarkList& landmarks,
                                int frame_width, int frame_height) const {
  RET_CHECK(landmarks.landmark_size() == kBaseLandmarkCount ||
            landmarks.landmark_size() == kAttentionLandmarkCount)
      << "Face pose input must contain 468 Base or 478 Attention landmarks!";

  // The Attention decoder has already replaced the Base lip and eye X/Y values
  // before this method runs. Preserve those refined values, but omit the ten
  // iris points because the configured canonical mesh has 468 vertices.
  NormalizedLandmarkList base_landmarks;
  for (int index = 0; index < kBaseLandmarkCount; ++index) {
    *base_landmarks.add_landmark() = landmarks.landmark(index);
  }

  MP_ASSIGN_OR_RETURN(
      auto geometry,
      implementation_->pipeline_->EstimateFaceGeometry(
          {base_landmarks}, frame_width, frame_height),
      _ << "Failed to estimate face geometry!");
  if (geometry.empty()) {
    return std::nullopt;
  }
  RET_CHECK_EQ(geometry.size(), 1)
      << "Single-face geometry estimation returned multiple results!";

  // FaceGeometry also contains the reconstructed canonical-local metric mesh.
  // The liberated tracking ABI currently needs only the pose which maps that
  // local mesh into runtime metric space, so it does not copy the mesh per frame.
  const auto& matrix = geometry.front().pose_transform_matrix();
  RET_CHECK_EQ(matrix.rows(), 4);
  RET_CHECK_EQ(matrix.cols(), 4);
  RET_CHECK_EQ(matrix.layout(), MatrixData::COLUMN_MAJOR)
      << "Face pose transform must use column-major storage!";
  RET_CHECK_EQ(matrix.packed_data_size(), 16);

  FacePoseTransform transform;
  std::copy(matrix.packed_data().begin(), matrix.packed_data().end(),
            transform.canonical_to_runtime_column_major.begin());
  return transform;
}

}  // namespace hand_tracking_mp_lean
