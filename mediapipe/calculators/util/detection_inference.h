#ifndef MEDIAPIPE_CALCULATORS_DETECTION_INFERENCE_H
#define MEDIAPIPE_CALCULATORS_DETECTION_INFERENCE_H

#include <vector>
#include "absl/status/status.h"
#include "mediapipe/framework/api2/node.h"
#include "mediapipe/framework/formats/tensor.h"

namespace mediapipe::api2 {

  class DetectionInference : public Node {
  public:
    // Required ports? Use Input<> / Output<>.
    // Optional ports? Use Input<>::Optional / Output<>::Optional.
    static constexpr Input<std::vector<Tensor>>             kInTensors{"TENSORS"};
    static constexpr Output<std::vector<Tensor>>::Optional  kOutTensors{"TENSORS"};

    MEDIAPIPE_NODE_INTERFACE(DetectionInference, kInTensors, kOutTensors);

    absl::Status Open(CalculatorContext* cc) override;
    absl::Status Process(CalculatorContext* cc) override;
  };

}  // namespace mediapipe::api2

#endif  // MEDIAPIPE_CALCULATORS_DETECTION_INFERENCE_H
