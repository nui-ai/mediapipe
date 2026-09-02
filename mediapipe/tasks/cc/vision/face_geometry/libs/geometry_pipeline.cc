// Copyright 2023 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "mediapipe/tasks/cc/vision/face_geometry/libs/geometry_pipeline.h"

#include <cmath>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "Eigen/Core"
#include "absl/memory/memory.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/matrix.h"
#include "mediapipe/framework/formats/matrix_data.pb.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/port/status_macros.h"
#include "mediapipe/framework/port/statusor.h"
#include "mediapipe/tasks/cc/vision/face_geometry/libs/mesh_3d_utils.h"
#include "mediapipe/tasks/cc/vision/face_geometry/libs/procrustes_solver.h"
#include "mediapipe/tasks/cc/vision/face_geometry/libs/validation_utils.h"
#include "mediapipe/tasks/cc/vision/face_geometry/proto/environment.pb.h"
#include "mediapipe/tasks/cc/vision/face_geometry/proto/face_geometry.pb.h"
#include "mediapipe/tasks/cc/vision/face_geometry/proto/geometry_pipeline_metadata.pb.h"
#include "mediapipe/tasks/cc/vision/face_geometry/proto/mesh_3d.pb.h"

namespace hand_tracking_mp_lean::tasks::vision::face_geometry {
namespace {

// Holds the image-aspect-specific bounds of the configured virtual camera.
struct PerspectiveCameraFrustum {
  // NOTE: all arguments must be validated prior to calling this constructor.
  PerspectiveCameraFrustum(const proto::PerspectiveCamera& perspective_camera,
                           int frame_width, int frame_height) {
    static constexpr float kDegreesToRadians = 3.14159265358979323846f / 180.f;

    // The virtual camera is at the origin. Its vertical field of view determines
    // the near-plane height at the configured near distance; the input image's
    // aspect ratio then determines the near-plane width. Screen X/Y coordinates
    // are mapped onto this rectangle before perspective unprojection.
    const float height_at_near =
        2.f * perspective_camera.near() *
        std::tan(0.5f * kDegreesToRadians *
                 perspective_camera.vertical_fov_degrees());

    const float width_at_near = frame_width * height_at_near / frame_height;

    left = -0.5f * width_at_near;
    right = 0.5f * width_at_near;
    bottom = -0.5f * height_at_near;
    top = 0.5f * height_at_near;
    near = perspective_camera.near();
    far = perspective_camera.far();
  }

  float left;
  float right;
  float bottom;
  float top;
  float near;
  float far;
};

// Performs one face's screen-to-metric reconstruction and canonical-face fit.
class ScreenToMetricSpaceConverter {
 public:
  ScreenToMetricSpaceConverter(
      proto::OriginPointLocation origin_point_location,  //
      proto::InputSource input_source,                   //
      Eigen::Matrix3Xf&& canonical_metric_landmarks,     //
      Eigen::VectorXf&& landmark_weights,                //
      std::unique_ptr<ProcrustesSolver> procrustes_solver)
      : origin_point_location_(origin_point_location),
        input_source_(input_source),
        canonical_metric_landmarks_(std::move(canonical_metric_landmarks)),
        landmark_weights_(std::move(landmark_weights)),
        procrustes_solver_(std::move(procrustes_solver)) {}

  // Reconstructs a runtime metric cloud and its canonical-to-runtime transform.
  //
  // The network input provides image-normalized X/Y and relative Z whose scale
  // follows image X. It does not provide camera-space depth. The conversion must
  // therefore infer the missing scale before perspective X/Y can be recovered:
  //
  // 1. Map normalized X/Y onto the virtual camera's near plane. Map Z with the
  //    same horizontal scale so its weak-perspective relation to X is retained.
  // 2. Fit the canonical face directly to that projected cloud. The fitted
  //    uniform scale supplies a first estimate of the depth conversion without
  //    yet treating relative Z as an absolute perspective depth.
  // 3. Recenter and rescale Z with that estimate, perspective-unproject X/Y,
  //    and fit again. This second scale measures the multiplicative correction
  //    revealed by the provisional perspective reconstruction.
  // 4. Repeat the Z conversion and unprojection with the product of both scale
  //    estimates. The result is the final runtime metric landmark cloud.
  // 5. Fit the canonical face to that cloud with the configured landmark
  //    weights. This produces the canonical-to-runtime pose transform.
  // 6. Apply the inverse transform to the runtime cloud. The returned mesh is
  //    thereby expressed in the canonical face's local metric frame, while the
  //    separately returned pose matrix places that mesh back in runtime space.
  //
  // Screen landmarks use a left-handed depth convention. Canonical, provisional,
  // and final metric landmarks use a right-handed convention, so each projected
  // screen cloud crosses that coordinate-system boundary by negating Z once.
  absl::Status Convert(
      const hand_tracking_mp_lean::NormalizedLandmarkList& screen_landmark_list,  //
      const PerspectiveCameraFrustum& pcf,                            //
      hand_tracking_mp_lean::LandmarkList& metric_landmark_list,                  //
      Eigen::Matrix4f& pose_transform_mat) const {
    RET_CHECK_EQ(screen_landmark_list.landmark_size(),
                 canonical_metric_landmarks_.cols())
        << "The number of landmarks doesn't match the number passed upon "
           "initialization!";

    Eigen::Matrix3Xf screen_landmarks;
    ConvertLandmarkListToEigenMatrix(screen_landmark_list, screen_landmarks);

    // ProjectXY maps X/Y to the virtual near-plane rectangle and applies that
    // rectangle's horizontal scale to relative Z. The mean projected Z is an
    // arbitrary network depth origin, not the face's camera-space distance.
    ProjectXY(pcf, screen_landmarks);
    const float depth_offset = screen_landmarks.row(2).mean();

    // First fit: use near-plane X/Y and relative Z as a weak-perspective cloud.
    // Perspective unprojection needs a depth for every point, so it cannot run
    // until this fit has supplied an initial canonical-to-runtime scale.
    Eigen::Matrix3Xf intermediate_landmarks(screen_landmarks);
    ChangeHandedness(intermediate_landmarks);

    MP_ASSIGN_OR_RETURN(const float first_iteration_scale,
                        EstimateScale(intermediate_landmarks),
                        _ << "Failed to estimate first iteration scale!");

    // Provisional reconstruction: the first scale turns centered relative Z
    // into a depth estimate. Unprojection then moves each near-plane X/Y point
    // along its camera ray to that estimated depth.
    intermediate_landmarks = screen_landmarks;
    MoveAndRescaleZ(pcf, depth_offset, first_iteration_scale,
                    intermediate_landmarks);
    UnprojectXY(pcf, intermediate_landmarks);
    ChangeHandedness(intermediate_landmarks);

    // For face detection input landmarks, re-write Z-coord from the canonical
    // landmarks.
    if (input_source_ == proto::InputSource::FACE_DETECTION_PIPELINE) {
      Eigen::Matrix4f intermediate_pose_transform_mat;
      MP_RETURN_IF_ERROR(procrustes_solver_->SolveWeightedOrthogonalProblem(
          canonical_metric_landmarks_, intermediate_landmarks,
          landmark_weights_, intermediate_pose_transform_mat))
          << "Failed to estimate pose transform matrix!";

      intermediate_landmarks.row(2) =
          (intermediate_pose_transform_mat *
           canonical_metric_landmarks_.colwise().homogeneous())
              .row(2);
    }
    MP_ASSIGN_OR_RETURN(const float second_iteration_scale,
                        EstimateScale(intermediate_landmarks),
                        _ << "Failed to estimate second iteration scale!");

    // The second fit measures the scale correction introduced by provisional
    // unprojection. Applying both multiplicative estimates to the original
    // projected landmarks produces the final runtime metric cloud.
    const float total_scale = first_iteration_scale * second_iteration_scale;
    MoveAndRescaleZ(pcf, depth_offset, total_scale, screen_landmarks);
    UnprojectXY(pcf, screen_landmarks);
    ChangeHandedness(screen_landmarks);

    // From this point onward, the matrix contains right-handed runtime metric
    // coordinates rather than normalized screen coordinates.
    Eigen::Matrix3Xf& metric_landmarks = screen_landmarks;

    MP_RETURN_IF_ERROR(procrustes_solver_->SolveWeightedOrthogonalProblem(
        canonical_metric_landmarks_, metric_landmarks, landmark_weights_,
        pose_transform_mat))
        << "Failed to estimate pose transform matrix!";

    // For face detection input landmarks, re-write Z-coord from the canonical
    // landmarks and run the pose transform estimation again.
    if (input_source_ == proto::InputSource::FACE_DETECTION_PIPELINE) {
      metric_landmarks.row(2) =
          (pose_transform_mat *
           canonical_metric_landmarks_.colwise().homogeneous())
              .row(2);

      MP_RETURN_IF_ERROR(procrustes_solver_->SolveWeightedOrthogonalProblem(
          canonical_metric_landmarks_, metric_landmarks, landmark_weights_,
          pose_transform_mat))
          << "Failed to estimate pose transform matrix!";
    }

    // Separate non-rigid face shape from global placement: applying the inverse
    // fitted similarity transform moves the runtime cloud into the canonical
    // face's local coordinate frame. The forward matrix remains the face pose.
    metric_landmarks = (pose_transform_mat.inverse() *
                        metric_landmarks.colwise().homogeneous())
                           .topRows(3);

    ConvertEigenMatrixToLandmarkList(metric_landmarks, metric_landmark_list);

    return absl::OkStatus();
  }

 private:
  void ProjectXY(const PerspectiveCameraFrustum& pcf,
                 Eigen::Matrix3Xf& landmarks) const {
    float x_scale = pcf.right - pcf.left;
    float y_scale = pcf.top - pcf.bottom;
    float x_translation = pcf.left;
    float y_translation = pcf.bottom;

    if (origin_point_location_ == proto::OriginPointLocation::TOP_LEFT_CORNER) {
      // The virtual camera uses bottom-up Y, whereas a top-left image origin
      // supplies downward-growing normalized Y.
      landmarks.row(1) = 1.f - landmarks.row(1).array();
    }

    // Map normalized X/Y onto the virtual camera's near-plane rectangle. Z is
    // relative rather than normalized, but the landmark model defines it in the
    // same scale as X, so it receives the near plane's horizontal scale too.
    landmarks =
        landmarks.array().colwise() * Eigen::Array3f(x_scale, y_scale, x_scale);
    landmarks.colwise() += Eigen::Vector3f(x_translation, y_translation, 0.f);
  }

  absl::StatusOr<float> EstimateScale(Eigen::Matrix3Xf& landmarks) const {
    Eigen::Matrix4f transform_mat;
    MP_RETURN_IF_ERROR(procrustes_solver_->SolveWeightedOrthogonalProblem(
        canonical_metric_landmarks_, landmarks, landmark_weights_,
        transform_mat))
        << "Failed to estimate canonical-to-runtime landmark set transform!";

    // A similarity transform's upper 3x3 block is sR. Every rotation column has
    // unit length, so the first transformed basis column has norm s.
    return transform_mat.col(0).norm();
  }

  static void MoveAndRescaleZ(const PerspectiveCameraFrustum& pcf,
                              float depth_offset, float scale,
                              Eigen::Matrix3Xf& landmarks) {
    // Subtract the network's arbitrary common depth offset, add the virtual near
    // distance, and divide that complete depth expression by the current
    // canonical-to-runtime scale estimate.
    landmarks.row(2) =
        (landmarks.array().row(2) - depth_offset + pcf.near) / scale;
  }

  static void UnprojectXY(const PerspectiveCameraFrustum& pcf,
                          Eigen::Matrix3Xf& landmarks) {
    // X/Y currently describe where each camera ray intersects the near plane.
    // Similar triangles move that point along the ray to its estimated Z by the
    // ratio Z / near.
    landmarks.row(0) =
        landmarks.row(0).cwiseProduct(landmarks.row(2)) / pcf.near;
    landmarks.row(1) =
        landmarks.row(1).cwiseProduct(landmarks.row(2)) / pcf.near;
  }

  static void ChangeHandedness(Eigen::Matrix3Xf& landmarks) {
    // Screen relative depth and the runtime metric camera convention point Z in
    // opposite directions. X and Y already use the intended metric orientation.
    landmarks.row(2) *= -1.f;
  }

  static void ConvertLandmarkListToEigenMatrix(
      const hand_tracking_mp_lean::NormalizedLandmarkList& landmark_list,
      Eigen::Matrix3Xf& eigen_matrix) {
    eigen_matrix = Eigen::Matrix3Xf(3, landmark_list.landmark_size());
    for (int i = 0; i < landmark_list.landmark_size(); ++i) {
      const auto& landmark = landmark_list.landmark(i);
      eigen_matrix(0, i) = landmark.x();
      eigen_matrix(1, i) = landmark.y();
      eigen_matrix(2, i) = landmark.z();
    }
  }

  static void ConvertEigenMatrixToLandmarkList(
      const Eigen::Matrix3Xf& eigen_matrix,
      hand_tracking_mp_lean::LandmarkList& landmark_list) {
    landmark_list.Clear();

    for (int i = 0; i < eigen_matrix.cols(); ++i) {
      auto& landmark = *landmark_list.add_landmark();
      landmark.set_x(eigen_matrix(0, i));
      landmark.set_y(eigen_matrix(1, i));
      landmark.set_z(eigen_matrix(2, i));
    }
  }

  const proto::OriginPointLocation origin_point_location_;
  const proto::InputSource input_source_;
  Eigen::Matrix3Xf canonical_metric_landmarks_;
  Eigen::VectorXf landmark_weights_;

  std::unique_ptr<ProcrustesSolver> procrustes_solver_;
};

// Owns immutable camera and canonical data shared by independent frame calls.
class GeometryPipelineImpl : public GeometryPipeline {
 public:
  GeometryPipelineImpl(
      const proto::PerspectiveCamera& perspective_camera,  //
      const proto::Mesh3d& canonical_mesh,                 //
      uint32_t canonical_mesh_vertex_size,                 //
      uint32_t canonical_mesh_num_vertices,
      uint32_t canonical_mesh_vertex_position_offset,
      std::unique_ptr<ScreenToMetricSpaceConverter> space_converter)
      : perspective_camera_(perspective_camera),
        canonical_mesh_(canonical_mesh),
        canonical_mesh_vertex_size_(canonical_mesh_vertex_size),
        canonical_mesh_num_vertices_(canonical_mesh_num_vertices),
        canonical_mesh_vertex_position_offset_(
            canonical_mesh_vertex_position_offset),
        space_converter_(std::move(space_converter)) {}

  absl::StatusOr<std::vector<proto::FaceGeometry>> EstimateFaceGeometry(
      const std::vector<hand_tracking_mp_lean::NormalizedLandmarkList>&
          multi_face_landmarks,
      int frame_width, int frame_height) const override {
    MP_RETURN_IF_ERROR(ValidateFrameDimensions(frame_width, frame_height))
        << "Invalid frame dimensions!";

    // Frame dimensions determine the virtual camera's aspect ratio. The same
    // resulting frustum applies to every face reconstructed from this frame.
    PerspectiveCameraFrustum pcf(perspective_camera_, frame_width,
                                 frame_height);

    std::vector<proto::FaceGeometry> multi_face_geometry;

    for (const hand_tracking_mp_lean::NormalizedLandmarkList& screen_face_landmarks :
         multi_face_landmarks) {
      // A cloud with negligible image-space extent cannot determine a stable
      // scale or rotation. Omit that face rather than emitting an arbitrary fit.
      if (IsScreenLandmarkListTooCompact(screen_face_landmarks)) {
        continue;
      }

      // Convert the neural output into a canonical-local metric mesh and obtain
      // the forward transform which places that mesh in runtime metric space.
      hand_tracking_mp_lean::LandmarkList metric_face_landmarks;
      Eigen::Matrix4f pose_transform_mat;
      MP_RETURN_IF_ERROR(space_converter_->Convert(screen_face_landmarks, pcf,
                                                   metric_face_landmarks,
                                                   pose_transform_mat))
          << "Failed to convert landmarks from the screen to the metric space!";

      proto::FaceGeometry face_geometry;
      proto::Mesh3d* mutable_mesh = face_geometry.mutable_mesh();
      // Reuse the canonical topology and UV coordinates, then replace only its
      // canonical XYZ values with this face's canonical-local reconstruction.
      mutable_mesh->CopyFrom(canonical_mesh_);
      for (int i = 0; i < canonical_mesh_num_vertices_; ++i) {
        uint32_t vertex_buffer_offset = canonical_mesh_vertex_size_ * i +
                                        canonical_mesh_vertex_position_offset_;

        mutable_mesh->set_vertex_buffer(vertex_buffer_offset,
                                        metric_face_landmarks.landmark(i).x());
        mutable_mesh->set_vertex_buffer(vertex_buffer_offset + 1,
                                        metric_face_landmarks.landmark(i).y());
        mutable_mesh->set_vertex_buffer(vertex_buffer_offset + 2,
                                        metric_face_landmarks.landmark(i).z());
      }
      // The pose matrix maps homogeneous canonical-local column vectors into
      // the runtime metric coordinate system used by the virtual camera.
      hand_tracking_mp_lean::MatrixDataProtoFromMatrix(
          pose_transform_mat, face_geometry.mutable_pose_transform_matrix());

      multi_face_geometry.push_back(face_geometry);
    }

    return multi_face_geometry;
  }

 private:
  static bool IsScreenLandmarkListTooCompact(
      const hand_tracking_mp_lean::NormalizedLandmarkList& screen_landmarks) {
    float mean_x = 0.f;
    float mean_y = 0.f;
    for (int i = 0; i < screen_landmarks.landmark_size(); ++i) {
      const auto& landmark = screen_landmarks.landmark(i);
      mean_x += (landmark.x() - mean_x) / static_cast<float>(i + 1);
      mean_y += (landmark.y() - mean_y) / static_cast<float>(i + 1);
    }

    float max_sq_dist = 0.f;
    for (const auto& landmark : screen_landmarks.landmark()) {
      const float d_x = landmark.x() - mean_x;
      const float d_y = landmark.y() - mean_y;
      max_sq_dist = std::max(max_sq_dist, d_x * d_x + d_y * d_y);
    }

    static constexpr float kIsScreenLandmarkListTooCompactThreshold = 1e-3f;
    return std::sqrt(max_sq_dist) <= kIsScreenLandmarkListTooCompactThreshold;
  }

  const proto::PerspectiveCamera perspective_camera_;
  const proto::Mesh3d canonical_mesh_;
  const uint32_t canonical_mesh_vertex_size_;
  const uint32_t canonical_mesh_num_vertices_;
  const uint32_t canonical_mesh_vertex_position_offset_;

  std::unique_ptr<ScreenToMetricSpaceConverter> space_converter_;
};

}  // namespace

absl::StatusOr<std::unique_ptr<GeometryPipeline>> CreateGeometryPipeline(
    const proto::Environment& environment,
    const proto::GeometryPipelineMetadata& metadata) {
  MP_RETURN_IF_ERROR(ValidateEnvironment(environment))
      << "Invalid environment!";
  MP_RETURN_IF_ERROR(ValidateGeometryPipelineMetadata(metadata))
      << "Invalid input-source, canonical-face, or fitting-basis configuration!";

  const auto& canonical_mesh = metadata.canonical_mesh();
  RET_CHECK(HasVertexComponent(canonical_mesh.vertex_type(),
                               VertexComponent::POSITION))
      << "Canonical face mesh must have the `POSITION` vertex component!";
  RET_CHECK(HasVertexComponent(canonical_mesh.vertex_type(),
                               VertexComponent::TEX_COORD))
      << "Canonical face mesh must have the `TEX_COORD` vertex component!";

  uint32_t canonical_mesh_vertex_size =
      GetVertexSize(canonical_mesh.vertex_type());
  uint32_t canonical_mesh_num_vertices =
      canonical_mesh.vertex_buffer_size() / canonical_mesh_vertex_size;
  uint32_t canonical_mesh_vertex_position_offset =
      GetVertexComponentOffset(canonical_mesh.vertex_type(),
                               VertexComponent::POSITION)
          .value();

  // Store every canonical vertex as one point-cloud column. The weight vector
  // has the same indexing as the 468 canonical vertices; vertices outside the
  // configured fitting basis retain weight zero and do not affect the pose fit.
  Eigen::Matrix3Xf canonical_metric_landmarks =
      Eigen::Matrix3Xf::Zero(3, canonical_mesh_num_vertices);
  Eigen::VectorXf landmark_weights =
      Eigen::VectorXf::Zero(canonical_mesh_num_vertices);

  for (int i = 0; i < canonical_mesh_num_vertices; ++i) {
    uint32_t vertex_buffer_offset =
        canonical_mesh_vertex_size * i + canonical_mesh_vertex_position_offset;

    canonical_metric_landmarks(0, i) =
        canonical_mesh.vertex_buffer(vertex_buffer_offset);
    canonical_metric_landmarks(1, i) =
        canonical_mesh.vertex_buffer(vertex_buffer_offset + 1);
    canonical_metric_landmarks(2, i) =
        canonical_mesh.vertex_buffer(vertex_buffer_offset + 2);
  }

  for (const proto::WeightedLandmarkRef& wlr :
       metadata.procrustes_landmark_basis()) {
    uint32_t landmark_id = wlr.landmark_id();
    landmark_weights(landmark_id) = wlr.weight();
  }

  std::unique_ptr<GeometryPipeline> result =
      absl::make_unique<GeometryPipelineImpl>(
          environment.perspective_camera(), canonical_mesh,
          canonical_mesh_vertex_size, canonical_mesh_num_vertices,
          canonical_mesh_vertex_position_offset,
          absl::make_unique<ScreenToMetricSpaceConverter>(
              environment.origin_point_location(),
              metadata.input_source() == proto::InputSource::DEFAULT
                  ? proto::InputSource::FACE_LANDMARK_PIPELINE
                  : metadata.input_source(),
              std::move(canonical_metric_landmarks),
              std::move(landmark_weights),
              CreateFloatPrecisionProcrustesSolver()));

  return result;
}

}  // namespace hand_tracking_mp_lean::tasks::vision::face_geometry
