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
    // Accept IMAGE input stream
    cc->Inputs().Tag("IMAGE").Set<ImageFrame>();
    // Optionally, output the tick count
    // cc->Outputs().Tag("TICK_COUNT").Set<int>();
    return absl::OkStatus();
  }

  absl::Status Open(CalculatorContext* cc) override {
    counter_ = 0;
    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) override {
    ++counter_;
    // Removed division by zero
    ABSL_LOG(INFO) << "Tick count: " << counter_;
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
