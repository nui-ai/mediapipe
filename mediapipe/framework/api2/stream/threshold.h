#ifndef MEDIAPIPE_FRAMEWORK_API2_STREAM_THRESHOLD_H_
#define MEDIAPIPE_FRAMEWORK_API2_STREAM_THRESHOLD_H_

#include "mediapipe/framework/api2/builder.h"

namespace mediapipe_v01013_based::api2::builder {

Stream<bool> IsOverThreshold(Stream<float> value, double threshold,
                             Graph& graph);

}  // namespace mediapipe_v01013_based::api2::builder

#endif  // MEDIAPIPE_FRAMEWORK_API2_STREAM_THRESHOLD_H_
