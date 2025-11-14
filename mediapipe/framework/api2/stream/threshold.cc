#include "mediapipe/framework/api2/stream/threshold.h"

#include "mediapipe/calculators/util/thresholding_calculator.pb.h"
#include "mediapipe/framework/api2/builder.h"

namespace hand_tracking_mp_lean::api2::builder {

Stream<bool> IsOverThreshold(Stream<float> value, double threshold,
                             hand_tracking_mp_lean::api2::builder::Graph& graph) {
  auto& node = graph.AddNode("ThresholdingCalculator");
  auto& node_opts = node.GetOptions<hand_tracking_mp_lean::ThresholdingCalculatorOptions>();
  node_opts.set_threshold(threshold);
  value.ConnectTo(node.In("FLOAT"));
  return node.Out("FLAG").Cast<bool>();
}

}  // namespace hand_tracking_mp_lean::api2::builder
