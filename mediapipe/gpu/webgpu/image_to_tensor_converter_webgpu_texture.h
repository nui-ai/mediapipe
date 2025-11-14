#ifndef MEDIAPIPE_GPU_WEBGPU_IMAGE_TO_TENSOR_CONVERTER_WEBGPU_TEXTURE_H_
#define MEDIAPIPE_GPU_WEBGPU_IMAGE_TO_TENSOR_CONVERTER_WEBGPU_TEXTURE_H_

#include <memory>

#include "mediapipe/calculators/tensor/image_to_tensor_converter.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/statusor.h"

namespace hand_tracking_mp_lean {

// Creates image to tensor (represented as WebGPU texture) converter.
// Note: Node::UpdateContract invocation must precede converter creation.
absl::StatusOr<std::unique_ptr<ImageToTensorConverter>>
CreateImageToWebGpuTextureTensorConverter(CalculatorContext* cc);

}  // namespace hand_tracking_mp_lean

#endif  // MEDIAPIPE_GPU_WEBGPU_IMAGE_TO_TENSOR_CONVERTER_WEBGPU_TEXTURE_H_
