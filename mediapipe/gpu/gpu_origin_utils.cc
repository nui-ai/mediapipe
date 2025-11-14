#include "mediapipe/gpu/gpu_origin_utils.h"

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"

namespace hand_tracking_mp_lean {

absl::StatusOr<bool> IsGpuOriginAtBottom(hand_tracking_mp_lean::GpuOrigin::Mode origin) {
  switch (origin) {
    case hand_tracking_mp_lean::GpuOrigin::TOP_LEFT:
      return false;
    case hand_tracking_mp_lean::GpuOrigin::DEFAULT:
    case hand_tracking_mp_lean::GpuOrigin::CONVENTIONAL:
      // TOP_LEFT on Metal, BOTTOM_LEFT on OpenGL.
#ifdef __APPLE__
      return false;
#else
      return true;
#endif
    default:
      return absl::InvalidArgumentError(
          absl::StrFormat("Unhandled GPU origin %i", origin));
  }
}

}  // namespace hand_tracking_mp_lean
