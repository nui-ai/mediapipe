#include "mediapipe/framework/port/statusor.h"

namespace hand_tracking_mp_lean {

StatusOr<std::string> GetDefaultTraceLogDirectory() {
  return "/data/local/tmp";
}

}  // namespace hand_tracking_mp_lean
