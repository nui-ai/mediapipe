#include "mediapipe/framework/api2/stream/segmentation_smoothing.h"

#include "mediapipe/calculators/image/segmentation_smoothing_calculator.pb.h"
#include "mediapipe/framework/api2/builder.h"
#include "mediapipe/framework/formats/image.h"

namespace hand_tracking_mp_lean::api2::builder {

Stream<Image> SmoothSegmentationMask(Stream<Image> mask,
                                     Stream<Image> previous_mask,
                                     float combine_with_previous_ratio,
                                     Graph& graph) {
  auto& smoothing_node = graph.AddNode("SegmentationSmoothingCalculator");
  auto& smoothing_node_opts =
      smoothing_node
          .GetOptions<hand_tracking_mp_lean::SegmentationSmoothingCalculatorOptions>();
  smoothing_node_opts.set_combine_with_previous_ratio(
      combine_with_previous_ratio);
  mask.ConnectTo(smoothing_node.In("MASK"));
  previous_mask.ConnectTo(smoothing_node.In("MASK_PREVIOUS"));
  return smoothing_node.Out("MASK_SMOOTHED").Cast<Image>();
}

}  // namespace hand_tracking_mp_lean::api2::builder
