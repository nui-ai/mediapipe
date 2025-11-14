#ifndef MEDIAPIPE_GPU_GPU_ORIGIN_UTILS_H_
#define MEDIAPIPE_GPU_GPU_ORIGIN_UTILS_H_

#include "absl/status/statusor.h"
#include "mediapipe/gpu/gpu_origin.pb.h"

namespace hand_tracking_mp_lean {

// Returns true if with the given GpuOrigin value the origin would start at the
// bottom, or an error if the GpuOrigin is unknown.
absl::StatusOr<bool> IsGpuOriginAtBottom(hand_tracking_mp_lean::GpuOrigin::Mode origin);

}  // namespace hand_tracking_mp_lean

#endif  // MEDIAPIPE_GPU_GPU_ORIGIN_UTILS_H_
