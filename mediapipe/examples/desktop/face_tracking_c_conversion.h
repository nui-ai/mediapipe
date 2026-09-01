#ifndef MEDIAPIPE_EXAMPLES_DESKTOP_FACE_TRACKING_C_CONVERSION_H_
#define MEDIAPIPE_EXAMPLES_DESKTOP_FACE_TRACKING_C_CONVERSION_H_

#include <string>

#include "mediapipe/examples/desktop/face_tracking_c_types.h"
#include "mediapipe/liberated/face_tracking.h"

// Deep-copies one C++ result into C-owned storage. The caller must pass a
// zero-initialized output and destroy it with face_tracking_result_destroy.
int ConvertFaceTrackingResultToC(
    const hand_tracking_mp_lean::ImageFaceTrackingResult& cpp_result,
    FaceTrackingResultC* c_result,
    void (*set_last_error)(const std::string& error));

#endif  // MEDIAPIPE_EXAMPLES_DESKTOP_FACE_TRACKING_C_CONVERSION_H_
