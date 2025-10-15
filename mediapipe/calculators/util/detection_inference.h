#include "absl/types/optional.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_options.pb.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"

namespace mediapipe {

class DetectionInference : public CalculatorBase {
public:
  static absl::Status GetContract(CalculatorContract* cc);
  absl::Status Open(CalculatorContext* cc);
  absl::Status Process(CalculatorContext* cc);
};

}  // namespace mediapipe

