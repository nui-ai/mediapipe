#ifndef MEDIAPIPE_CALCULATORS_DETECTION_INFERENCE_H
#define MEDIAPIPE_CALCULATORS_DETECTION_INFERENCE_H

#include <vector>
#include "absl/status/status.h"
#include "mediapipe/framework/api2/node.h"
#include "mediapipe/framework/formats/tensor.h"

namespace mediapipe::api2 {

  // this is the newer mediapipe internal api way of defining a calculator not the older one which inherits CalculatorBase.
  // it is supposed to have many advantages: more strongly typed, Less boilerplate, more compile-time checking,
  // cleaner move-semantics (Consume()), fewer gotchas than legacy use of GetContract(). As per ChatGPT.
  class DummyCalculator : public Node {
  public:

    // define the calculator contract via a macro not a GetContract method
    static constexpr Input<std::vector<Tensor>>             kInTensors{"TENSORS"};
    static constexpr Output<std::vector<Tensor>>::Optional  kOutTensors{"TENSORS"};
    MEDIAPIPE_NODE_INTERFACE(DummyCalculator, kInTensors, kOutTensors);

    absl::Status Open(CalculatorContext* cc) override;
    absl::Status Process(CalculatorContext* cc) override;
  };

}  // namespace mediapipe::api2

#endif  // MEDIAPIPE_CALCULATORS_DETECTION_INFERENCE_H
