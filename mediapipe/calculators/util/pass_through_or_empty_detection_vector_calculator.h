#ifndef MEDIAPIPE_CALCULATORS_UTILS_PASS_THROUGH_OR_EMPTY_DETECTION_VECTOR_CALCULATOR_H_
#define MEDIAPIPE_CALCULATORS_UTILS_PASS_THROUGH_OR_EMPTY_DETECTION_VECTOR_CALCULATOR_H_

#include <vector>

#include "mediapipe/framework/api2/builder.h"
#include "mediapipe/framework/api2/node.h"
#include "mediapipe/framework/api2/packet.h"
#include "mediapipe/framework/api2/port.h"
#include "mediapipe/framework/formats/detection.pb.h"

namespace mediapipe_v01013_based {

// Calculator to pass through input vector of detections if packet is not empty,
// otherwise - outputing a new empty vector. So, instead of empty packet you get
// a packet containing empty vector.
//
// Example:
// node {
//   calculator: "PassThroughOrEmptyDetectionVectorCalculator"
//   input_stream: "TICK:tick"
//   input_stream: "VECTOR:input_detections"
//   output_stream: "VECTOR:output_detections"
// }
class PassThroughOrEmptyDetectionVectorCalculator
    : public mediapipe_v01013_based::api2::NodeIntf {
 public:
  static constexpr mediapipe_v01013_based::api2::Input<std::vector<mediapipe_v01013_based::Detection>>
      kInputVector{"VECTOR"};
  static constexpr mediapipe_v01013_based::api2::Input<mediapipe_v01013_based::api2::AnyType> kTick{
      "TICK"};
  static constexpr mediapipe_v01013_based::api2::Output<std::vector<mediapipe_v01013_based::Detection>>
      kOutputVector{"VECTOR"};

  MEDIAPIPE_NODE_INTERFACE(
      ::mediapipe_v01013_based::PassThroughOrEmptyDetectionVectorCalculator, kInputVector,
      kTick, kOutputVector);
};

template <typename TickT>
api2::builder::Stream<std::vector<mediapipe_v01013_based::Detection>>
PassThroughOrEmptyDetectionVector(
    api2::builder::Stream<std::vector<mediapipe_v01013_based::Detection>> detections,
    api2::builder::Stream<TickT> tick, mediapipe_v01013_based::api2::builder::Graph& graph) {
  auto& node =
      graph.AddNode("mediapipe.PassThroughOrEmptyDetectionVectorCalculator");
  detections.ConnectTo(
      node[PassThroughOrEmptyDetectionVectorCalculator::kInputVector]);
  tick.ConnectTo(node[PassThroughOrEmptyDetectionVectorCalculator::kTick]);
  return node[PassThroughOrEmptyDetectionVectorCalculator::kOutputVector];
}

}  // namespace mediapipe_v01013_based

#endif  // MEDIAPIPE_CALCULATORS_UTILS_PASS_THROUGH_OR_EMPTY_DETECTION_VECTOR_CALCULATOR_H_
