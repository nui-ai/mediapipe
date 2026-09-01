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

// Bottom-line inference for one face. landmark_count is 468 for the base
// model and 478 for the attention model.
typedef struct FaceInferenceC {
  FaceLandmarkC* landmarks;
  size_t landmark_count;
  float presence_score;
  FaceRectC rect_from_landmarks;
} FaceInferenceC;

// Results for one input image. Face ordering is consistent among fields in
// this result but does not represent persistent identities across images.
typedef struct FaceTrackingResultC {
  FaceInferenceC* faces;
  size_t face_count;
  uint8_t face_detector_ran;
} FaceTrackingResultC;

#ifdef __cplusplus
}
#endif

#endif  // MEDIAPIPE_EXAMPLES_DESKTOP_FACE_TRACKING_C_TYPES_H_
