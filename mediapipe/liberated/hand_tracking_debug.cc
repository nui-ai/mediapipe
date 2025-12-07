#include "mediapipe/liberated/hand_tracking_debug.h"
#include <iostream>
#include <algorithm>
#include <functional>

namespace hand_tracking_mp_lean {

void image_debug_logging(api2::ImageToTensorCoreResult *image_struct) {
  ABSL_LOG(INFO) << "image padding: ";
  std::cout << image_struct->padding[0] << " "
      << image_struct->padding[1] << " "
      << image_struct->padding[2] << " "
      << image_struct->padding[3] << std::endl;

  ABSL_LOG(INFO) << "image matrix: ";
  for (const auto& val : image_struct->matrix) {
    std::cout << val << " ";
  }
  std::cout << std::endl;

  ABSL_LOG(INFO) << "image middle section values: ";
  for (const auto& tensor: image_struct->tensors) {
    const auto& vals = tensor.GetCpuReadView().buffer<float>();
    for (int i = static_cast<int>(192*192*3*0.4); i < static_cast<int>(192*192*3*0.402); ++i) {
      std::cout << vals[i] << " ";
    }
  }
  std::cout << std::endl;

  std::cout << "image hash: ";
  std::size_t hash = 0;
  for (const auto& tensor : image_struct->tensors) {
    const auto& vals = tensor.GetCpuReadView().buffer<float>();
    for (int i = 0; i < tensor.shape().num_elements(); ++i) {
      hash ^= std::hash<float>{}(vals[i]) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
    }
  }
  std::cout << std::hex << hash << std::dec << " " << std::endl;
}

void sub_image_for_landmarks_inference_debug_logging(api2::ImageToTensorCoreResult *extracted_sub_image_struct) {
  ABSL_LOG(INFO) << "sub image padding: ";
  std::cout << extracted_sub_image_struct->padding[0] << " "
      << extracted_sub_image_struct->padding[1] << " "
      << extracted_sub_image_struct->padding[2] << " "
      << extracted_sub_image_struct->padding[3] << std::endl;

  ABSL_LOG(INFO) << "sub image matrix: ";
  for (const auto& val : extracted_sub_image_struct->matrix) {
    std::cout << val << " ";
  }
  std::cout << std::endl;

  ABSL_LOG(INFO) << "sub image first few values: ";
  for (const auto& tensor: extracted_sub_image_struct->tensors) {
    const auto& vals = tensor.GetCpuReadView().buffer<float>();
    for (int i = 0; i <  224*3; ++i) {
      std::cout << vals[i] << " ";
    }
  }
  std::cout << std::endl;

  std::cout << "sub image hash: ";
  std::size_t hash = 0;
  for (const auto& tensor : extracted_sub_image_struct->tensors) {
    const auto& vals = tensor.GetCpuReadView().buffer<float>();
    for (int i = 0; i < tensor.shape().num_elements(); ++i) {
      hash ^= std::hash<float>{}(vals[i]) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
    }
  }
  std::cout << std::hex << hash << std::dec << " " << std::endl;
}

void sub_image_padding_debug_logging(api2::ImageToTensorCoreResult* extracted_sub_image_struct) {
  if (std::any_of(extracted_sub_image_struct->padding.begin(), extracted_sub_image_struct->padding.end(),
                  [](float v) { return v > 0.0001f; })) {
    ABSL_LOG(INFO) << "non-zero letterbox padding: "
                        << extracted_sub_image_struct->padding[0] << extracted_sub_image_struct->padding[1]
                        << extracted_sub_image_struct->padding[2] << extracted_sub_image_struct->padding[3]; }
}

void landmarks_inference_debug_logging(std::vector<Tensor> landmarks_inference_output_tensors) {
  ABSL_LOG(INFO) << "landmarks inference first few values (the viewport landmarks unnormalized): ";
  const auto& first_tensor_vals = landmarks_inference_output_tensors[0].GetCpuReadView().buffer<float>();
  for (int i = 0; i < 21; ++i) {
    std::cout << first_tensor_vals[i] << " ";
  }
  std::cout << std::endl;
}

} // namespace hand_tracking_mp_lean

