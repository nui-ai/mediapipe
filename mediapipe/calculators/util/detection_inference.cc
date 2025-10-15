#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "mediapipe/framework/api2/node.h"
#include "mediapipe/framework/api2/packet.h"
#include "mediapipe/framework/api2/port.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/status_macros.h"
#include "mediapipe/framework/resources.h"
#include "mediapipe/calculators/util/detection_inference.h"

namespace mediapipe {

  absl::Status DetectionInference::GetContract(CalculatorContract* cc) {
    return absl::OkStatus();
  }

  absl::Status DetectionInference::Open(CalculatorContext* cc) {
    return absl::OkStatus();
  }

  absl::Status DetectionInference::Process(CalculatorContext* cc) {
    return absl::OkStatus();
  }

  MEDIAPIPE_REGISTER_NODE(DetectionInference)

}