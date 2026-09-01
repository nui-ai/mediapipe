#include "mediapipe/examples/desktop/face_tracking_c_types.h"

#include <cstdlib>

extern "C" void face_tracking_result_destroy(FaceTrackingResultC* result) {
  if (result == nullptr) {
    return;
  }
  if (result->faces != nullptr) {
    for (size_t index = 0; index < result->face_count; ++index) {
      std::free(result->faces[index].landmarks);
    }
    std::free(result->faces);
  }
  std::free(result);
}
