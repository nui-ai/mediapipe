#ifndef MEDIAPIPE_FRAMEWORK_API2_STREAM_TENSOR_TO_JOINTS_H_
#define MEDIAPIPE_FRAMEWORK_API2_STREAM_TENSOR_TO_JOINTS_H_

#include "mediapipe/framework/api2/builder.h"
#include "mediapipe/framework/formats/body_rig.pb.h"
#include "mediapipe/framework/formats/tensor.h"

namespace hand_tracking_mp_lean::api2::builder {

// Updates @graph to convert @tensor to a JointList skipping first @start_index
// values of a @tensor.
Stream<hand_tracking_mp_lean::JointList> ConvertTensorToJointsAtIndex(Stream<Tensor> tensor,
                                                          const int num_joints,
                                                          const int start_index,
                                                          Graph& graph);

// Updates @graph to convert @tensor to a JointList.
inline Stream<::hand_tracking_mp_lean::JointList> ConvertTensorToJoints(
    Stream<Tensor> tensor, const int num_joints, Graph& graph) {
  return ConvertTensorToJointsAtIndex(tensor, num_joints, /*start_index=*/0,
                                      graph);
}

}  // namespace hand_tracking_mp_lean::api2::builder

#endif  // MEDIAPIPE_FRAMEWORK_API2_STREAM_TENSOR_TO_JOINTS_H_
