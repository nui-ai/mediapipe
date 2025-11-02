#include "mediapipe/framework/port/statusor.h"

namespace mediapipe_v01013_based {

StatusOr<std::string> GetDefaultTraceLogDirectory() {
  return "/data/local/tmp";
}

}  // namespace mediapipe_v01013_based
