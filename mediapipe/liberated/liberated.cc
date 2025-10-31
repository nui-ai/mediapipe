#include "mediapipe/liberated/liberated.h"

namespace mediapipe {

Liberated::Liberated(MemoryManager* memory_manager) {

  // initialize for image to tensors conversions
  auto options = ImageToTensorCalculatorOptions();
  options.set_output_tensor_width(192);
  options.set_output_tensor_height(192);
  options.set_keep_aspect_ratio(true);
  options.mutable_output_tensor_float_range()->set_min(0.0f);
  options.mutable_output_tensor_float_range()->set_max(1.0f);
  options.set_border_mode(mediapipe::ImageToTensorCalculatorOptions::BORDER_ZERO);
  auto params = GetOutputTensorParams(options);
  int tensor_width = params.output_width.value_or(0);
  int tensor_height = params.output_height.value_or(0);
  image_to_tensor_core_ = std::make_unique<api2::ImageToTensorCalculatorCore>(
      options, tensor_width, tensor_height, params,
      gpu_converter_, cpu_converter_, memory_manager);
}

  absl::Status Liberated::Process(const std::vector<mediapipe::NormalizedRect> &prev_hand_rects_from_landmarks, std::shared_ptr<const Image> image, uint32_t max_hands_to_track) const {
    auto palm_detection_image = nullptr;

    if (prev_hand_rects_from_landmarks.size() == max_hands_to_track) {
      ABSL_LOG(INFO) << "the number of hands detected from the previous frame's landmarks (" << prev_hand_rects_from_landmarks.size() << ") is equal to the globally set maximum number of hands to track " << max_hands_to_track;
      ABSL_LOG(INFO) << "skipping palm detection";
    } else if (prev_hand_rects_from_landmarks.size() > max_hands_to_track) {
      ABSL_LOG(INFO) << "the number of hands detected from the previous frame's landmarks (" << prev_hand_rects_from_landmarks.size() << ") is larger than the globally set maximum number of hands to track " << max_hands_to_track;
      ABSL_LOG(INFO) << "skipping palm detection as";
    } else if (prev_hand_rects_from_landmarks.size() < max_hands_to_track) {

      ABSL_LOG(INFO) << "palm detection will be triggered for the current frame as the number of previous frame's detections from landmarks is smaller than the set maximum number of hands to track";

      api2::ImageToTensorCoreResult core_result;
      absl::optional<NormalizedRect> norm_rect = absl::nullopt;
      return image_to_tensor_core_->Process(*image, norm_rect, &core_result);




    }
}

}