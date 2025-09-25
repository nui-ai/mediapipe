#include <cstdlib>
#include "absl/log/absl_log.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_base.h"

// MediaPipe calculators should be in the mediapipe namespace
namespace mediapipe {

class TickCountCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc) {
    // Output the tick count
    cc->Outputs().Tag("TICK_COUNT").Set<int>();
    return absl::OkStatus();
  }

  absl::Status Open(CalculatorContext* cc) override {
    counter_ = 0;
    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) override {
    ++counter_;
    ABSL_LOG(INFO) << "Tick count: " << counter_;
    cc->Outputs().Tag("TICK_COUNT").Add(new int(counter_), cc->InputTimestamp());
    return absl::OkStatus();
  }

 private:
  int counter_ = 0;
};

// Register without namespace prefix
REGISTER_CALCULATOR(TickCountCalculator);

}  // namespace mediapipe
