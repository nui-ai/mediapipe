#ifndef HAND_TRACKING_C_CONVERSION_H_
#define HAND_TRACKING_C_CONVERSION_H_

#include "mediapipe/examples/desktop/hand_tracking_c_types.h"
#include "mediapipe/liberated/hand_tracking.h"
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

int ConvertCppResultToCNestedStruct(const hand_tracking_mp_lean::ImageHandTrackingAndInferenceResult& cpp_result, HandTrackingResultC* c_result_out, void (*set_last_error)(const std::string& err));

#ifdef __cplusplus
}
#endif

#endif // HAND_TRACKING_C_CONVERSION_H_
