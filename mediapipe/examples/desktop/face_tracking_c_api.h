// C ABI for the CalculatorGraph-free face tracking implementation.
#ifndef MEDIAPIPE_EXAMPLES_DESKTOP_FACE_TRACKING_C_API_H_
#define MEDIAPIPE_EXAMPLES_DESKTOP_FACE_TRACKING_C_API_H_

#include <stddef.h>
#include <stdint.h>

#include "mediapipe/examples/desktop/face_tracking_c_types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef void* FaceTrackingCoreOpaqueHandle;

// Creates one stateful tracker. A null options pointer selects the C++ API's
// defaults; a null assets_path resolves repository-relative runtime assets.
FaceTrackingCoreOpaqueHandle face_tracking_core_create(
    const FaceTrackingOptionsC* options, const char* assets_path);

// Processes one borrowed RGB8 sRGB image. The caller retains the pixel memory
// for the duration of this call and owns result_out on success.
int face_tracking_core_process(FaceTrackingCoreOpaqueHandle opaque_handle,
                               const uint8_t* data, size_t width,
                               size_t height, size_t row_stride,
                               FaceTrackingResultC** result_out);

// Clears landmark-derived ROIs retained from preceding images.
int face_tracking_core_reset(FaceTrackingCoreOpaqueHandle opaque_handle);

// Destroys one tracker created by face_tracking_core_create.
int face_tracking_core_finalize(FaceTrackingCoreOpaqueHandle opaque_handle);

// Destroys a result and all nested allocations returned by process.
void face_tracking_result_destroy(FaceTrackingResultC* result);

// Returns the latest error for this thread. The pointer remains owned by the
// library and is invalidated by the next face API call on the same thread.
const char* face_tracking_get_last_error(void);

// Returns a semantic version whose major identifies the C struct ABI.
const char* face_tracking_core_version(void);

#ifdef __cplusplus
}
#endif

#endif  // MEDIAPIPE_EXAMPLES_DESKTOP_FACE_TRACKING_C_API_H_
