// Pure C types for graph-free face tracking configuration and results.
#ifndef MEDIAPIPE_EXAMPLES_DESKTOP_FACE_TRACKING_C_TYPES_H_
#define MEDIAPIPE_EXAMPLES_DESKTOP_FACE_TRACKING_C_TYPES_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Runtime configuration for one stateful face tracker. Boolean values use
// zero for false and any non-zero value for true across the C ABI.
typedef struct FaceTrackingOptionsC {
  uint32_t max_faces;
  uint8_t use_previous_landmarks;
  uint8_t with_attention;
  float min_detection_confidence;
  float min_tracking_confidence;
  int32_t xnnpack_num_threads;
  // Non-zero runs face geometry for every accepted face and requests its pose.
  uint8_t estimate_pose;
  // Virtual camera vertical field of view. This must be finite and within
  // (0, 180) degrees when estimate_pose is non-zero; otherwise it is unused.
  float vertical_fov_degrees;
} FaceTrackingOptionsC;

// One image-normalized face landmark. x and y use the input viewport; z uses
// the model's normalized depth scale.
typedef struct FaceLandmarkC {
  float x;
  float y;
  float z;
  float visibility;
  float presence;
} FaceLandmarkC;

// Image-normalized rectangle with counter-clockwise rotation in radians.
typedef struct FaceRectC {
  float x_center;
  float y_center;
  float width;
  float height;
  float rotation;
} FaceRectC;

// Canonical-face to runtime-metric similarity transform. The upper 3x3 block
// contains uniform scale multiplied by a proper rotation, and the rightmost
// column contains translation. Values use column-major storage and the matrix
// acts on homogeneous column vectors.
typedef struct FacePoseTransformC {
  float canonical_to_runtime_column_major[16];
} FacePoseTransformC;

// Bottom-line inference for one face. landmark_count is 468 for the base
// model and 478 for the attention model.
typedef struct FaceInferenceC {
  FaceLandmarkC* landmarks;
  size_t landmark_count;
  float presence_score;
  FaceRectC rect_from_landmarks;
  // Non-zero means pose_transform contains the fit for this face. Zero means
  // pose was disabled or this accepted face was too compact for a stable fit.
  uint8_t has_pose_transform;
  FacePoseTransformC pose_transform;
} FaceInferenceC;

// Results for one input image. Face array positions do not represent
// persistent identities across images.
typedef struct FaceTrackingResultC {
  FaceInferenceC* faces;
  size_t face_count;
  uint8_t face_detector_ran;
} FaceTrackingResultC;

#ifdef __cplusplus
}
#endif

#endif  // MEDIAPIPE_EXAMPLES_DESKTOP_FACE_TRACKING_C_TYPES_H_
