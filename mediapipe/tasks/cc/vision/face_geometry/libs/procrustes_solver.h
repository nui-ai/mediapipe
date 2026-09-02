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

#ifndef MEDIAPIPE_TASKS_CC_VISION_FACE_GEOMETRY_LIBS_PROCRUSTES_SOLVER_H_
#define MEDIAPIPE_TASKS_CC_VISION_FACE_GEOMETRY_LIBS_PROCRUSTES_SOLVER_H_

#include <memory>

#include "Eigen/Dense"
#include "mediapipe/framework/port/status.h"

namespace hand_tracking_mp_lean::tasks::vision::face_geometry {

// Fits one weighted 3D point cloud to another with a similarity transform.
// This is the Weighted Extended Orthogonal Procrustes (WEOP) problem defined in
// Section 2.4 of
// https://doi.org/10.3929/ethz-a-004656648.
//
// Each matrix column is one 3D point. The returned 4x4 matrix acts on
// homogeneous column vectors and maps source points to target points. Its upper
// 3x3 block contains one uniform scale multiplied by a proper rotation; its
// rightmost column contains translation. The solution minimizes the sum of each
// point's squared residual multiplied by that point's weight.
class ProcrustesSolver {
 public:
  virtual ~ProcrustesSolver() = default;

  // Fits `source_points` to `target_points` under `point_weights`.
  //
  // All `source_points`, `target_points` and `point_weights` must define the
  // same number of points. Elements of `point_weights` must be non-negative.
  //
  // A too small diameter of either of the point clouds will likely lead to
  // numerical instabilities and failure to estimate the transformation.
  //
  // A too small point cloud total weight will likely lead to numerical
  // instabilities and failure to estimate the transformation too.
  //
  // Small point coordinate deviation for either of the point cloud will likely
  // result in a failure as it will make the solution very unstable if possible.
  //
  // Note: the output `transform_mat` argument is used instead of `StatusOr<>`
  // return type in order to avoid Eigen memory alignment issues. Details:
  // https://eigen.tuxfamily.org/dox/group__TopicStructHavingEigenMembers.html
  virtual absl::Status SolveWeightedOrthogonalProblem(
      const Eigen::Matrix3Xf& source_points,  //
      const Eigen::Matrix3Xf& target_points,  //
      const Eigen::VectorXf& point_weights,   //
      Eigen::Matrix4f& transform_mat) const = 0;
};

std::unique_ptr<ProcrustesSolver> CreateFloatPrecisionProcrustesSolver();

}  // namespace hand_tracking_mp_lean::tasks::vision::face_geometry

#endif  // MEDIAPIPE_TASKS_CC_VISION_FACE_GEOMETRY_LIBS_PROCRUSTES_SOLVER_H_
