#include "mediapipe/liberated/liberated.h"
#include "mediapipe/calculators/tensor/model_inference.h"
#include "mediapipe/calculators/tensor/inference_calculator.h"
#include "mediapipe/calculators/tensor/inference_calculator_utils.h"
#include "mediapipe/calculators/tensor/inference_interpreter_delegate_runner_new.h"
#include "mediapipe/calculators/tensor/inference_runner.h"
#include "mediapipe/calculators/util/detection_letterbox_removal.h"

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

  // initialize for palm detection inference
  const std::string& model_path = "mediapipe/modules/palm_detection/palm_detection_full.tflite";
  palm_detection_inference_ = std::make_unique<api2::ModelInference>(model_path);

  // initialize for detection inference conversion to tensors
  inference_filter_stage1_ = std::make_unique<api2::ConvertDetectionTensors>();

}

  absl::StatusOr<std::unique_ptr<std::vector<Detection>>> Liberated::Process(const std::vector<mediapipe::NormalizedRect> &prev_hand_rects_from_landmarks, std::shared_ptr<const Image> image, uint32_t max_hands_to_track) const {
    // auto palm_detection_image = nullptr;
    auto count_capped_detections = absl::make_unique<std::vector<Detection>>();

    if (prev_hand_rects_from_landmarks.size() == max_hands_to_track) {
      ABSL_LOG(INFO) << "the number of hands detected from the previous frame's landmarks (" << prev_hand_rects_from_landmarks.size() << ") is equal to the globally set maximum number of hands to track " << max_hands_to_track;
      ABSL_LOG(INFO) << "skipping palm detection";
    } else if (prev_hand_rects_from_landmarks.size() > max_hands_to_track) {
      ABSL_LOG(INFO) << "the number of hands detected from the previous frame's landmarks (" << prev_hand_rects_from_landmarks.size() << ") is larger than the globally set maximum number of hands to track " << max_hands_to_track;
      ABSL_LOG(INFO) << "skipping palm detection as";
    } else if (prev_hand_rects_from_landmarks.size() < max_hands_to_track) {

      ABSL_LOG(INFO) << "palm detection will be triggered for the current frame as the number of previous frame's detections from landmarks is smaller than the set maximum number of hands to track";

      // image to tensor input format for the palm detection model
      api2::ImageToTensorCoreResult image_as_tensor;
      absl::optional<NormalizedRect> norm_rect = absl::nullopt;
      MP_RETURN_IF_ERROR(image_to_tensor_core_->Process(*image, norm_rect, &image_as_tensor));
      auto letterbox_padding_ = image_as_tensor.padding;
      TensorSpan image_as_tensor_span;
      image_as_tensor_span = MakeTensorSpan(image_as_tensor.tensors);

      // palm detection inference
      absl::StatusOr<std::vector<Tensor>> inference;
      MP_ASSIGN_OR_RETURN(inference, palm_detection_inference_->Process(image_as_tensor_span));
      ABSL_LOG(INFO) << "palm detection inference completed";

      // extract and first step filter the detection inference output
      std::unique_ptr<std::vector<Detection>> filtered_detections_letterboxed;
      MP_ASSIGN_OR_RETURN(filtered_detections_letterboxed, inference_filter_stage1_->Process(*inference));
      ABSL_LOG(INFO) << "detection inference conversion to tensors completed";

      std::unique_ptr<std::vector<Detection>> filtered_detections = UnLetterBox(*filtered_detections_letterboxed, letterbox_padding_);
      ABSL_LOG(INFO) << "detection letterbox removal completed";

      // extremely naively fit the number of detections to be no larger than the maximum hands being tracked constant;
      // this was part of the original pipeline (as a ClipVectorSizeCalculator subclass calculator).
      // this is mostly a weak stop-gap element unless they have been ordered in some semantic way by the previous
      // above stages, and otherwise would be thrown out as part of harmonizing the overall handling of the potential
      // and expected multiplicity of detection (and their landmarks) which are inherent to SSD and to our overall.
      if (filtered_detections->size() > max_hands_to_track) {
        for (int i = 0; i < max_hands_to_track; ++i) {
          count_capped_detections->push_back(filtered_detections->at(i));
        }
      } else {
        for (int i = 0; i < filtered_detections->size(); ++i) {
          count_capped_detections->push_back(filtered_detections->at(i));
        }
      }
    }

    return count_capped_detections;
}

}
