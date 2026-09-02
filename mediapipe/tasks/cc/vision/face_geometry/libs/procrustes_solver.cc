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

#include "mediapipe/tasks/cc/vision/face_geometry/libs/procrustes_solver.h"

#include <cmath>
#include <memory>

#include "Eigen/Dense"
#include "absl/memory/memory.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/port/status_macros.h"
#include "mediapipe/framework/port/statusor.h"

namespace hand_tracking_mp_lean::tasks::vision::face_geometry {
namespace {

// Implements the weighted similarity fit with single-precision Eigen matrices.
class FloatPrecisionProcrustesSolver : public ProcrustesSolver {
 public:
  FloatPrecisionProcrustesSolver() = default;

  absl::Status SolveWeightedOrthogonalProblem(
      const Eigen::Matrix3Xf& source_points,  //
      const Eigen::Matrix3Xf& target_points,  //
      const Eigen::VectorXf& point_weights,
      Eigen::Matrix4f& transform_mat) const override {
    // Validate inputs.
    MP_RETURN_IF_ERROR(ValidateInputPoints(source_points, target_points))
        << "Failed to validate weighted orthogonal problem input points!";
    MP_RETURN_IF_ERROR(
        ValidatePointWeights(source_points.cols(), point_weights))
        << "Failed to validate weighted orthogonal problem point weights!";

    // Scaling each residual by sqrt(weight) makes its squared norm contribute
    // exactly `weight * squared_residual` to the least-squares objective.
    Eigen::VectorXf sqrt_weights = ExtractSquareRoot(point_weights);

    MP_RETURN_IF_ERROR(InternalSolveWeightedOrthogonalProblem(
        source_points, target_points, sqrt_weights, transform_mat))
        << "Failed to solve the WEOP problem!";

    return absl::OkStatus();
  }

 private:
  static constexpr float kAbsoluteErrorEps = 1e-9f;

  static absl::Status ValidateInputPoints(
      const Eigen::Matrix3Xf& source_points,
      const Eigen::Matrix3Xf& target_points) {
    RET_CHECK_GT(source_points.cols(), 0)
        << "The number of source points must be positive!";

    RET_CHECK_EQ(source_points.cols(), target_points.cols())
        << "The number of source and target points must be equal!";

    return absl::OkStatus();
  }

  static absl::Status ValidatePointWeights(
      int num_points, const Eigen::VectorXf& point_weights) {
    RET_CHECK_GT(point_weights.size(), 0)
        << "The number of point weights must be positive!";

    RET_CHECK_EQ(point_weights.size(), num_points)
        << "The number of points and point weights must be equal!";

    float total_weight = 0.f;
    for (int i = 0; i < num_points; ++i) {
      RET_CHECK_GE(point_weights(i), 0.f)
          << "Each point weight must be non-negative!";

      total_weight += point_weights(i);
    }

    RET_CHECK_GT(total_weight, kAbsoluteErrorEps)
        << "The total point weight is too small!";

    return absl::OkStatus();
  }

  static Eigen::VectorXf ExtractSquareRoot(
      const Eigen::VectorXf& point_weights) {
    Eigen::VectorXf sqrt_weights(point_weights);
    for (int i = 0; i < sqrt_weights.size(); ++i) {
      sqrt_weights(i) = std::sqrt(sqrt_weights(i));
    }

    return sqrt_weights;
  }

  // Combines a 3x3 rotation-and-scale matrix and a 3x1 translation vector into
  // a single 4x4 transformation matrix.
  static Eigen::Matrix4f CombineTransformMatrix(const Eigen::Matrix3f& r_and_s,
                                                const Eigen::Vector3f& t) {
    Eigen::Matrix4f result = Eigen::Matrix4f::Identity();
    result.leftCols(3).topRows(3) = r_and_s;
    result.col(3).topRows(3) = t;

    return result;
  }

  // `sources` and `targets` store corresponding 3D points in matrix columns.
  // This method finds scale s, proper rotation R, and translation t which
  // minimize sum_i w_i * ||s R sources_i + t - targets_i||^2.
  //
  // Multiplication by sqrt(w_i) converts that weighted objective into an
  // ordinary matrix least-squares norm. Weighted centering first removes the
  // translation term. An SVD of the centered cross-covariance then supplies the
  // best proper rotation. With rotation fixed, scalar projection supplies the
  // best uniform scale, and the weighted mean residual supplies translation.
  //
  // The derivation follows Section 2.4 of:
  // D. Akca, Generalized Procrustes analysis and its applications
  // in photogrammetry, 2003, https://doi.org/10.3929/ethz-a-004656648
  //
  // The paper stores points in rows and uses W_p = Q^T Q. Here points are
  // columns, W_p is diagonal, and Q is `diag(sqrt_weights)`. The implemented
  // matrix expressions are therefore transposes of the paper's equations.
  //
  // Note: the output `transform_mat` argument is used instead of `StatusOr<>`
  // return type in order to avoid Eigen memory alignment issues. Details:
  // https://eigen.tuxfamily.org/dox/group__TopicStructHavingEigenMembers.html
  static absl::Status InternalSolveWeightedOrthogonalProblem(
      const Eigen::Matrix3Xf& sources, const Eigen::Matrix3Xf& targets,
      const Eigen::VectorXf& sqrt_weights, Eigen::Matrix4f& transform_mat) {
    // Multiply point column i by sqrt(w_i). A later squared Frobenius norm then
    // gives point i its requested weight w_i.
    Eigen::Matrix3Xf weighted_sources =
        sources.array().rowwise() * sqrt_weights.array().transpose();
    Eigen::Matrix3Xf weighted_targets =
        targets.array().rowwise() * sqrt_weights.array().transpose();

    // sqrt(w_i)^2 recovers each original weight.
    float total_weight = sqrt_weights.cwiseProduct(sqrt_weights).sum();

    // Compute the source centroid as sum_i(w_i * source_i) / sum_i(w_i).
    // Centering the weighted source cloud removes translation from the rotation
    // and scale subproblem. The sqrt weights remain on the centered columns.
    Eigen::Matrix3Xf twice_weighted_sources =
        weighted_sources.array().rowwise() * sqrt_weights.array().transpose();
    Eigen::Vector3f source_center_of_mass =
        twice_weighted_sources.rowwise().sum() / total_weight;
    Eigen::Matrix3Xf centered_weighted_sources =
        weighted_sources - source_center_of_mass * sqrt_weights.transpose();

    // This product is the weighted cross-covariance between target points and
    // centered source points. Its SVD determines the rotation which maximizes
    // their alignment without permitting a mirror reflection.
    Eigen::Matrix3f rotation;
    MP_RETURN_IF_ERROR(ComputeOptimalRotation(
        weighted_targets * centered_weighted_sources.transpose(), rotation))
        << "Failed to compute the optimal rotation!";
    MP_ASSIGN_OR_RETURN(
        float scale,
        ComputeOptimalScale(centered_weighted_sources, weighted_sources,
                            weighted_targets, rotation),
        _ << "Failed to compute the optimal scale!");

    // The homogeneous transform stores sR in its upper 3x3 block.
    Eigen::Matrix3f rotation_and_scale = scale * rotation;

    // Once scale and rotation are fixed, translation is the weighted mean of
    // `target_i - s R source_i`. Both clouds already contain one sqrt(weight),
    // so multiplying their residual by sqrt(weight) once more supplies weight.
    const auto pointwise_diffs =
        weighted_targets - rotation_and_scale * weighted_sources;
    const auto weighted_pointwise_diffs =
        pointwise_diffs.array().rowwise() * sqrt_weights.array().transpose();
    Eigen::Vector3f translation =
        weighted_pointwise_diffs.rowwise().sum() / total_weight;

    transform_mat = CombineTransformMatrix(rotation_and_scale, translation);

    return absl::OkStatus();
  }

  // Computes the proper rotation from the weighted cross-covariance matrix.
  //
  // Note: the output `rotation` argument is used instead of `StatusOr<>`
  // return type in order to avoid Eigen memory alignment issues. Details:
  // https://eigen.tuxfamily.org/dox/group__TopicStructHavingEigenMembers.html
  static absl::Status ComputeOptimalRotation(
      const Eigen::Matrix3f& design_matrix, Eigen::Matrix3f& rotation) {
    RET_CHECK_GT(design_matrix.norm(), kAbsoluteErrorEps)
        << "Design matrix norm is too small!";

    Eigen::JacobiSVD<Eigen::Matrix3f> svd(
        design_matrix, Eigen::ComputeFullU | Eigen::ComputeFullV);

    Eigen::Matrix3f postrotation = svd.matrixU();
    Eigen::Matrix3f prerotation = svd.matrixV().transpose();

    // U V^T is the unconstrained optimal orthogonal alignment. If its
    // determinant is negative, it contains a reflection. Flip U's column for
    // the least singular direction to obtain the nearest proper rotation with
    // determinant +1. This is the constrained orthogonal Procrustes correction
    // described in section 4.6 of Gower and Dijksterhuis, Procrustes Problems.
    if (postrotation.determinant() * prerotation.determinant() <
        static_cast<float>(0)) {
      postrotation.col(2) *= static_cast<float>(-1);
    }

    rotation = postrotation * prerotation;
    return absl::OkStatus();
  }

  static absl::StatusOr<float> ComputeOptimalScale(
      const Eigen::Matrix3Xf& centered_weighted_sources,
      const Eigen::Matrix3Xf& weighted_sources,
      const Eigen::Matrix3Xf& weighted_targets,
      const Eigen::Matrix3f& rotation) {
    // Project the centered source cloud through the chosen rotation. The ratio
    // of its correlation with the target cloud to its own squared magnitude is
    // the least-squares uniform scale for that fixed rotation.
    const auto rotated_centered_weighted_sources =
        rotation * centered_weighted_sources;
    // A coefficient-wise product and sum evaluates the required inner products
    // without constructing the larger trace expressions from the derivation.
    float numerator =
        rotated_centered_weighted_sources.cwiseProduct(weighted_targets).sum();
    float denominator =
        centered_weighted_sources.cwiseProduct(weighted_sources).sum();

    RET_CHECK_GT(denominator, kAbsoluteErrorEps)
        << "Scale expression denominator is too small!";
    RET_CHECK_GT(numerator / denominator, kAbsoluteErrorEps)
        << "Scale is too small!";

    return numerator / denominator;
  }
};

}  // namespace

std::unique_ptr<ProcrustesSolver> CreateFloatPrecisionProcrustesSolver() {
  return absl::make_unique<FloatPrecisionProcrustesSolver>();
}

}  // namespace hand_tracking_mp_lean::tasks::vision::face_geometry
