#include <cstdlib>
#include "absl/log/absl_log.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_base.h"
#include "mediapipe/framework/formats/image_frame.h"

// MediaPipe calculators should be in the mediapipe namespace
namespace mediapipe {

class TickCountCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc) {
    // Accept the IMAGE input stream, which is necessary in order for this calculator to be actually invoked,
    // and obviously will only get invoked once per image (as it only appears in the pipeline once).
    cc->Inputs().Tag("IMAGE").Set<ImageFrame>();
    // later, maybe output the tick count for every calculator to consume,
    // if we don't have this calculator increment a global counter for them
    // to directly access.
    // cc->Outputs().Tag("TICK_COUNT").Set<int>();
    return absl::OkStatus();
  }

  absl::Status Open(CalculatorContext* cc) override {
    counter_ = 0;
    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) override {
    ABSL_LOG(INFO) << "pipeline processing frame number: " << counter_;
    ++counter_;
    // Optionally, output the tick count
    // cc->Outputs().Tag("TICK_COUNT").Add(new int(counter_), cc->InputTimestamp());
    return absl::OkStatus();
  }

 private:
  int counter_ = 0;
};

// Register without namespace prefix
REGISTER_CALCULATOR(TickCountCalculator);

}  // namespace mediapipe
