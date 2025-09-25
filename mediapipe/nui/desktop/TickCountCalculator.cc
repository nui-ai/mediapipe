#include <cstdlib>
#include "absl/log/absl_log.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_base.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/nui/desktop/SharedCalculatorState.h"

// MediaPipe calculators should be in the mediapipe namespace
namespace mediapipe {

class TickCountCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc) {
    // Accept the IMAGE input stream, which is necessary in order for this calculator to be actually invoked,
    // and obviously will only get invoked once per image (as it only appears in the pipeline once).
    cc->Inputs().Tag("IMAGE").Set<ImageFrame>();
    cc->Outputs().Tag("COUNTING_DONE").Set<bool>();
    return absl::OkStatus();
  }

  absl::Status Open(CalculatorContext* cc) override {
    SharedCalculatorState::ResetCounter();
    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) override {
    ABSL_LOG(INFO) << "pipeline processing frame number: " << SharedCalculatorState::GetCounter();
    SharedCalculatorState::IncrementCounter();
    cc->Outputs().Tag("COUNTING_DONE").Add(new bool(true), cc->InputTimestamp());
    return absl::OkStatus();
  }

};

REGISTER_CALCULATOR(TickCountCalculator);

}