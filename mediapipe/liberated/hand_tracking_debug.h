#pragma once
#include <vector>
#include "mediapipe/liberated/hand_tracking.h"

namespace hand_tracking_mp_lean {

void image_debug_logging(api2::ImageToTensorCoreResult *image_struct);
void sub_image_for_landmarks_inference_debug_logging(api2::ImageToTensorCoreResult *extracted_sub_image_struct);
void sub_image_padding_debug_logging(api2::ImageToTensorCoreResult* extracted_sub_image_struct);
void landmarks_inference_debug_logging(std::vector<Tensor> landmarks_inference_output_tensors);

}

