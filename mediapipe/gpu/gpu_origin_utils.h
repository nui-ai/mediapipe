#ifndef MEDIAPIPE_GPU_GPU_ORIGIN_UTILS_H_
#define MEDIAPIPE_GPU_GPU_ORIGIN_UTILS_H_

#include "absl/status/statusor.h"
#include "mediapipe/gpu/gpu_origin.pb.h"

namespace mediapipe_v01013_based {

// Returns true if with the given GpuOrigin value the origin would start at the
// bottom, or an error if the GpuOrigin is unknown.
absl::StatusOr<bool> IsGpuOriginAtBottom(mediapipe_v01013_based::GpuOrigin::Mode origin);

}  // namespace mediapipe_v01013_based

#endif  // MEDIAPIPE_GPU_GPU_ORIGIN_UTILS_H_
