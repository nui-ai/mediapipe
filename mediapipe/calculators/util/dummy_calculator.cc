#include "dummy_calculator.h"

#include <utility>
#include "absl/status/status.h"
#include "mediapipe/framework/api2/packet.h"

namespace mediapipe_v01013_based::api2 {

  absl::Status DummyCalculator::Open(CalculatorContext* cc) {
    return absl::OkStatus();
  }

  absl::Status DummyCalculator::Process(CalculatorContext* cc) {
    if (kInTensors(cc).IsEmpty()) return absl::OkStatus();

    // Move-only safe: takes ownership of the vector<Tensor> from the packet.
    MP_ASSIGN_OR_RETURN(auto tensors, kInTensors(cc).Consume());
    // tensors is now std::vector<Tensor>

    // ... run inference, possibly producing `tensors_out` ...
    if (kOutTensors(cc).IsConnected()) {
      kOutTensors(cc).Send(std::move(tensors));  // or std::move(tensors_out)
    }
    return absl::OkStatus();
  }

  MEDIAPIPE_REGISTER_NODE(DummyCalculator);

}  // namespace mediapipe_v01013_based::api2


