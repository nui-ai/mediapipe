// C API for the pipeline liberated redux of the hand tracking pipeline, to be used in other languages (e.g. rust).

#ifndef NO_PIPELINE_C_API_H_
#define NO_PIPELINE_C_API_H_

#include "mediapipe/liberated/liberated_core.h"
#include "mediapipe/framework/deps/safe_int.h"
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque handle for the C++ HandsPipelineOperator instance,
// being returned by the create function and taken as input
// by the other below API functions
typedef void* HandTrackingCoreOpaqueHandle;

// C api creation of an instance of the underlying C++ class
HandTrackingCoreOpaqueHandle hand_tracking_core_create(uint max_hands_to_track);

int hand_tracking_core_process(
    HandTrackingCoreOpaqueHandle opaque_handle,
    const std::shared_ptr<const hand_tracking_mp_lean::Image>& image);

/// function implementations for the above signatures return non-zero and set a "last_error"
/// string value if they fail. This function should be used by the caller, to retrieve that
/// last message for the caller, in case the caller gets a non-zero return value
/// from any of the above API-exposed functions of this C API.
const char* hand_tracking_get_last_error();

// finalize the instance of the underlying C++ class
int hand_tracking_core_finalize(HandTrackingCoreOpaqueHandle opaque_handle);

#ifdef __cplusplus
}
#endif

#endif  // NO_PIPELINE_C_API_H_
