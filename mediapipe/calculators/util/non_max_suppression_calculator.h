#ifndef MEDIAPIPE_CALCULATORS_UTIL_NON_MAX_SUPPRESSION_CALCULATOR_H_
#define MEDIAPIPE_CALCULATORS_UTIL_NON_MAX_SUPPRESSION_CALCULATOR_H_

#include <memory>
#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/calculators/util/non_max_suppression_calculator.pb.h"

namespace mediapipe {

using Detections = std::vector<::mediapipe::Detection>;
class NonMaxSuppressionCalculatorOptions;

std::unique_ptr<Detections> FilterDetectionsByNonMaximumSuppression(
    const Detections& input_detections,
    const NonMaxSuppressionCalculatorOptions& options,
    bool has_dimensions = false,
    int frame_width = 0,
    int frame_height = 0);

}  // namespace mediapipe

#endif  // MEDIAPIPE_CALCULATORS_UTIL_NON_MAX_SUPPRESSION_CALCULATOR_H_

