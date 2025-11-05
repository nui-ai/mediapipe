#include "mediapipe/liberated/liberated.h"

namespace mediapipe_v01013_based {

Liberated::Liberated(MemoryManager* memory_manager) {

  // initialize for image to tensors conversions (to be reduced into much less surface)
  auto image_to_tensor_options = ImageToTensorCalculatorOptions();
  image_to_tensor_options.set_output_tensor_width(192);
  image_to_tensor_options.set_output_tensor_height(192);
  image_to_tensor_options.set_keep_aspect_ratio(true);
  image_to_tensor_options.mutable_output_tensor_float_range()->set_min(0.0f);
  image_to_tensor_options.mutable_output_tensor_float_range()->set_max(1.0f);
  image_to_tensor_options.set_border_mode(mediapipe_v01013_based::ImageToTensorCalculatorOptions::BORDER_ZERO);
  auto params = GetOutputTensorParams(image_to_tensor_options);
  int input_tensor_width = params.output_width.value_or(0);
  int input_tensor_height = params.output_height.value_or(0);
  image_to_tensor_core_ = std::make_unique<api2::ImageToTensorCalculatorCore>(
      image_to_tensor_options, input_tensor_width, input_tensor_height, params,
      gpu_converter_, cpu_converter_, memory_manager);

  // initialize for palm detection inference
  const std::string& palm_detection_model_path = "mediapipe/modules/palm_detection/palm_detection_full.tflite";
  palm_detection_inference_ = std::make_unique<api2::ModelInference>(palm_detection_model_path);

  // initialize for detection inference conversion to tensors
  inference_filter_stage1_ = std::make_unique<api2::ConvertDetectionTensors>();

  // initialize for orienting the raw (axes parallel) palm rect detected by SSD, to the palm's rough shape by detection keypoints
  // included in the output of the palm detection inference itself (https://chatgpt.com/s/t_690b528ae748819181a48117cb417908).
  auto target_angle_rad = static_cast<float>(M_PI * 90.0 / 180.0);
  palm_detection_to_oriented_palm_rect_ = std::make_unique<DetectionsToOrientedRects>(target_angle_rad);

  // initialize for expanding from aligned palm rects to aligned hand (palm + fingers) rects
  auto oriented_palm_rect_to_hand_rect_expander_options = RectTransformationCalculatorOptions();
  oriented_palm_rect_to_hand_rect_expander_options.set_scale_x(2.6f);
  oriented_palm_rect_to_hand_rect_expander_options.set_scale_y(2.6f);
  oriented_palm_rect_to_hand_rect_expander_options.set_shift_y(-0.5);
  oriented_palm_rect_to_hand_rect_expander_options.set_square_long(true);
  // ABSL_LOG(INFO) << "RectTransformationCalculator options: " << options_.DebugString();
  oriented_palm_rect_to_hand_rect_expander_ = std::make_unique<PalmRectToHandRect>(oriented_palm_rect_to_hand_rect_expander_options);

  // initialize for extracting the sub-image implied by the hand rectangles (to be reduced into much less surface)
  auto sub_image_extraction_options = ImageToTensorCalculatorOptions();
  sub_image_extraction_options.set_output_tensor_width(224);
  sub_image_extraction_options.set_output_tensor_height(224);
  sub_image_extraction_options.set_keep_aspect_ratio(true);
  sub_image_extraction_options.mutable_output_tensor_float_range()->set_min(0.0f);
  sub_image_extraction_options.mutable_output_tensor_float_range()->set_max(1.0f);
  sub_image_extraction_options.set_border_mode(mediapipe_v01013_based::ImageToTensorCalculatorOptions::BORDER_UNSPECIFIED);
  auto params_ = GetOutputTensorParams(sub_image_extraction_options);
  auto sub_image_extraction_input_tensor_width = params_.output_width.value_or(0);
  auto sub_image_extraction_input_tensor_height = params_.output_height.value_or(0);
  sub_image_for_landmarks_inference_extractor_ = std::make_unique<api2::ImageToTensorCalculatorCore>(
      sub_image_extraction_options, sub_image_extraction_input_tensor_width, sub_image_extraction_input_tensor_height, params_,
      gpu_converter_, cpu_converter_, memory_manager);

  // initialize for landmarks inference
  const std::string& landmarks_infernce_model_path = "mediapipe/modules/hand_landmark/hand_landmark_full.tflite";
  landmarks_inference_ = std::make_unique<api2::ModelInference>(landmarks_infernce_model_path);

  // initialize for splitting the output tensors of the landmarks inference output by topic
  landmarks_inference_splitter_ = std::make_unique<InferenceOutputTensorSplitting<Tensor, false>>(SplitVectorCalculatorOptions());

  // initialize for extracting the handedness information from a landmarks inference output
  handedness_classification_config_ = api2::TensorsToClassificationConfig();
  handedness_classification_config_.is_binary_classification = true;
  handedness_classification_config_.label_map_loaded = true;
  handedness_classification_config_.class_index_set.is_allowlist = false;
  handedness_classification_config_.top_k = 1;
  handedness_classification_config_.sort_by_descending_score = false;

  // initialize for extracting the viewport landmarks from the landmarks inference output
  auto landmarks_extraction_options = TensorsToLandmarksCalculatorOptions();
  landmarks_extractor_ = std::make_unique<api2::TensorsToLandmarksCore>(
    224, 224,
    landmarks_extraction_options.visibility_activation(), landmarks_extraction_options.presence_activation(),
    0,21);

}

  absl::StatusOr<std::unique_ptr<std::vector<NormalizedRect>>> Liberated::Process(const std::vector<NormalizedRect> &prev_hand_rects_from_landmarks, std::shared_ptr<const Image> image, uint32_t max_hands_to_track) const {
    // auto palm_detection_image = nullptr;
    auto count_capped_detections = absl::make_unique<std::vector<Detection>>();
    auto hand_rects_from_detections = absl::make_unique<std::vector<NormalizedRect>>();
    auto merged_hand_rectangles_list = absl::make_unique<std::list<NormalizedRect>>();
    auto merged_hand_rectangles = absl::make_unique<std::vector<NormalizedRect>>();

    if (prev_hand_rects_from_landmarks.size() == max_hands_to_track) {
      ABSL_LOG(INFO) << "the number of hands detected from the previous frame's landmarks (" << prev_hand_rects_from_landmarks.size() << ") is equal to the globally set maximum number of hands to track " << max_hands_to_track;
      ABSL_LOG(INFO) << "skipping palm detection";
    } else if (prev_hand_rects_from_landmarks.size() > max_hands_to_track) { // this does happen, arising in the de-facto chains of calculation mirroring the original pipeline
      ABSL_LOG(INFO) << "the number of hands rectangles from the previous frame's landmarks (" << prev_hand_rects_from_landmarks.size() << ") is larger than the globally set maximum number of hands to track " << max_hands_to_track;
      ABSL_LOG(INFO) << "skipping palm detection";
      // return absl::InternalError("the number of hands rectangels from the previous frame's landmarks is larger than the globally set maximum number of hands to track, which is currently unexpected");
    }

    // start the palm detection -> expanded oriented hand region for landmark inference path of computation
    if (prev_hand_rects_from_landmarks.size() < max_hands_to_track) {

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
      ABSL_LOG(INFO) << "naive detections capping completed";

      // orient the palm detections all by one function call which takes all of them
      std::vector<NormalizedRect> oriented_palm_norm_rects;
      std::vector<Rect> oriented_palm_rects; // unused output argument required by the below function in its current legacy form
      auto image_size = std::make_pair(image->width(), image->height());
      MP_RETURN_IF_ERROR(palm_detection_to_oriented_palm_rect_->OrientedRectsFromDetections(*count_capped_detections, image_size, &oriented_palm_norm_rects, &oriented_palm_rects));

      // technically speaking unlike the former step, we loop each rect here not in the inside expander fn but by looping the inside expander fn
      hand_rects_from_detections = absl::make_unique<std::vector<NormalizedRect>>(oriented_palm_rects.size());
      for (int i = 0; i < oriented_palm_rects.size(); ++i) {
        // copy the rect
        hand_rects_from_detections->at(i) = oriented_palm_norm_rects[i];
        // expand the rect
        auto it = hand_rects_from_detections->begin() + i;
        oriented_palm_rect_to_hand_rect_expander_->ExpandNormalizedRect(&(*it), image->width(), image->height());
      }
    }

    // merges with IoU threshold based filtering, the set of hand rects derived directly from palm detection inference, with the set of hand rects derived from the previous frame's landmarks inference,
    // both of which input sets to this merge may be empty or not. if both are empty, an empty set should be the result.
    MP_ASSIGN_OR_RETURN(*merged_hand_rectangles_list, mediapipe_v01013_based::IouFilterMerge(*hand_rects_from_detections, prev_hand_rects_from_landmarks, 0.5));
    merged_hand_rectangles = absl::make_unique<std::vector<NormalizedRect>>(merged_hand_rectangles_list->begin(), merged_hand_rectangles_list->end());  // convert from list to vector

    // start looping or fanning out in place of the original pipeline's fanning out of the hand rects for landmarks inference,
    // which have been accomplished above.
    for (auto norm_rect : *merged_hand_rectangles) {
      // extract the sub-image implied by the established hand rectangles, for landmarks inference.
      // this sub-image will currently be 224x224 pixels.
      api2::ImageToTensorCoreResult sub_image_extraction_struct;
      MP_RETURN_IF_ERROR(sub_image_for_landmarks_inference_extractor_->Process(*image, norm_rect, &sub_image_extraction_struct));
      MP_ASSIGN_OR_RETURN(std::vector<Tensor> output_tensors, landmarks_inference_->Process(MakeTensorSpan(sub_image_extraction_struct.tensors)));

      // get a unique pointer to output_tensors that can be passed to a function expecting std::unique_ptr<std::vector<T>>*
      auto output_tensors_ptr = std::make_unique<std::vector<Tensor>>(std::move(output_tensors));

      // split the result of the landmarks inference into topics (this can be much distilled to discard the over-generalized Run method entirely and just assign the inference outputs directly)
      std::vector<Tensor> output_elements;  // output argument not consumed by our code (an over generalization feature of the original pipeline)
      std::unique_ptr<std::vector<Tensor>> combined_output;  // output argument not consumed by our code (an over generalization feature of the original pipeline)
      std::vector<std::unique_ptr<std::vector<Tensor>>> output_vectors;
      MP_RETURN_IF_ERROR(landmarks_inference_splitter_->Run(&output_tensors_ptr, &output_vectors, &output_elements, &combined_output));
      std::unique_ptr<std::vector<Tensor>> inference_output_viewport_landmarks = std::move(output_vectors[0]);
      std::unique_ptr<std::vector<Tensor>> inference_output_hand_presence = std::move(output_vectors[1]);
      std::unique_ptr<std::vector<Tensor>> inference_output_hand_handedness = std::move(output_vectors[2]);
      std::unique_ptr<std::vector<Tensor>> inference_output_object_landmarks = std::move(output_vectors[3]);

      // extract the viewport landmarks from the landmarks inference output
      NormalizedLandmarkList inferred_landmarks;
      MP_RETURN_IF_ERROR(landmarks_extractor_->TensorsToLandmarks(*inference_output_viewport_landmarks, &inferred_landmarks));

      // extract the hand presence score from the landmarks inference output
      auto result = tensors_to_floats_calculator_core::Process(*inference_output_hand_presence, TensorsToFloatsCalculatorOptions());

      // extract the hand handedness classification object (object holding handedness value and its confidence) from the landmarks inference output
      auto inferred_handedness_score = inference_output_hand_handedness->at(0).GetCpuReadView().buffer<float>();
      std::unique_ptr<ClassificationList> inferred_handedness_classification_object = ProcessTensorToClassifications(inferred_handedness_score, 2, handedness_classification_config_);

      // extract the object landmarks from the landmarks inference output

    }



  return merged_hand_rectangles;
}

}
