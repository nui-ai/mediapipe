#ifndef MEDIAPIPE_LIBERATED_FACE_GEOMETRY_ESTIMATOR_H_
#define MEDIAPIPE_LIBERATED_FACE_GEOMETRY_ESTIMATOR_H_

#include <array>
#include <memory>
#include <optional>
#include <string>

#include "absl/status/statusor.h"
#include "mediapipe/framework/formats/landmark.pb.h"

namespace hand_tracking_mp_lean {

// Maps the canonical face into the runtime metric coordinate system.
//
// The column-major 4x4 matrix acts on homogeneous column vectors. Its upper 3x3
// block contains uniform scale multiplied by a proper rotation, and its
// rightmost column contains translation.
struct FacePoseTransform {
  std::array<float, 16> canonical_to_runtime_column_major;
};

// Derives face pose from accepted screen landmarks for one camera configuration.
//
// Creation loads the canonical face and weighted fitting basis once. Each call
// then runs MediaPipe's graph-independent GeometryPipeline for one face and
// retains the fitted pose transform from its full geometry result.
class FaceGeometryEstimator {
 public:
  static absl::StatusOr<std::unique_ptr<FaceGeometryEstimator>> Create(
      float vertical_fov_degrees, const std::string* assets_path);

  ~FaceGeometryEstimator();

  FaceGeometryEstimator(const FaceGeometryEstimator&) = delete;
  FaceGeometryEstimator& operator=(const FaceGeometryEstimator&) = delete;

  // Fits pose to a Base or Attention landmark set in input-image coordinates.
  // Attention's ten iris landmarks have no vertices in the configured
  // canonical mesh, so the fit uses the final refined first 468 landmarks.
  // Numerically compact input has no stable pose and returns `std::nullopt`
  // without failing the frame.
  absl::StatusOr<std::optional<FacePoseTransform>> Estimate(
      const NormalizedLandmarkList& landmarks, int frame_width,
      int frame_height) const;

 private:
  class Impl;

  explicit FaceGeometryEstimator(std::unique_ptr<Impl> implementation);

  std::unique_ptr<Impl> implementation_;
};

}  // namespace hand_tracking_mp_lean

#endif  // MEDIAPIPE_LIBERATED_FACE_GEOMETRY_ESTIMATOR_H_
