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
  oriented_palm_rect_to_hand_rect_expander_options.set_shift_y(-0.5f);
  oriented_palm_rect_to_hand_rect_expander_options.set_square_long(true);
  // ABSL_LOG(INFO) << "RectTransformationCalculator options: " << options_.DebugString();
  oriented_palm_rect_to_hand_rect_expander_ = std::make_unique<RectTransformation>(oriented_palm_rect_to_hand_rect_expander_options);

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
    0.4f,
    21);

  // initialize for expanding a hand rectangle derived from inferred hand landmarks, to a rectangle to be used for landmarks inference on the next frame
  auto expand_rect_for_next_frame_options = RectTransformationCalculatorOptions();
  ABSL_ASSERT(!(expand_rect_for_next_frame_options.has_rotation() && expand_rect_for_next_frame_options.has_rotation_degrees()));
  ABSL_ASSERT(!(expand_rect_for_next_frame_options.has_square_long() && expand_rect_for_next_frame_options.has_square_short()));
  expand_rect_for_next_frame_options.set_scale_x(2.0f);
  expand_rect_for_next_frame_options.set_scale_y(2.0f);
  expand_rect_for_next_frame_options.set_shift_y(-0.1f);
  expand_rect_for_next_frame_options.set_square_long(true);
  expand_rect_for_next_frame_ = std::make_unique<RectTransformation>(expand_rect_for_next_frame_options);

  // on the very first video frame there are no hand rectangles derived from the previous frame
  hand_rect_from_previous_frame_ = std::vector<NormalizedRect>();

  // ABSL_LOG(INFO) << "RectTransformationCalculator options: " << options_.DebugString();

}

  absl::StatusOr<std::unique_ptr<ImageHandTrackingAndInferenceResult>> Liberated::Process(
    std::shared_ptr<const Image> image,
    uint32_t max_hands_to_track) {

    // initiate the result structure for the current image as empty vectors for all of its fields
    auto result = std::make_unique<ImageHandTrackingAndInferenceResult>(
    ImageHandTrackingAndInferenceResult{
        std::make_unique<std::vector<NormalizedLandmarkList>>(),
        std::make_unique<std::vector<LandmarkList>>(),
        std::make_unique<std::vector<ClassificationList>>()
    });

  // auto palm_detection_image = nullptr;

    auto count_capped_detections = absl::make_unique<std::vector<Detection>>();
    auto hand_rects_from_detections = absl::make_unique<std::vector<NormalizedRect>>();
    auto merged_hand_rectangles_list = absl::make_unique<std::list<NormalizedRect>>();
    auto merged_hand_rectangles = absl::make_unique<std::vector<NormalizedRect>>();

    if (hand_rect_from_previous_frame_.size() == max_hands_to_track) {
      ABSL_LOG(INFO) << "the number of hands detected from the previous frame's landmarks (" << hand_rect_from_previous_frame_.size() << ") is equal to the globally set maximum number of hands to track " << max_hands_to_track;
      ABSL_LOG(INFO) << "skipping palm detection";
    } else if (hand_rect_from_previous_frame_.size() > max_hands_to_track) { // this does happen, arising in the de-facto chains of calculation mirroring the original pipeline
      ABSL_LOG(INFO) << "the number of hands rectangles from the previous frame's landmarks (" << hand_rect_from_previous_frame_.size() << ") is larger than the globally set maximum number of hands to track " << max_hands_to_track;
      ABSL_LOG(INFO) << "skipping palm detection";
      // return absl::InternalError("the number of hands rectangels from the previous frame's landmarks is larger than the globally set maximum number of hands to track, which is currently unexpected");
    }

    // start the palm detection -> expanded oriented hand region for landmark inference path of computation
    if (hand_rect_from_previous_frame_.size() < max_hands_to_track) {

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
    MP_ASSIGN_OR_RETURN(*merged_hand_rectangles_list, mediapipe_v01013_based::IouFilterMerge(*hand_rects_from_detections, hand_rect_from_previous_frame_, 0.5));
    merged_hand_rectangles = absl::make_unique<std::vector<NormalizedRect>>(merged_hand_rectangles_list->begin(), merged_hand_rectangles_list->end());  // convert from list to vector

    // reset the list of rectangles passed from the previous frame's pass, before we start building one from scratch for the next frame ...
    hand_rect_from_previous_frame_ = std::vector<NormalizedRect>();

    // start looping or fanning out in place of the original pipeline's fanning out of the hand rects for landmarks inference, which have been accomplished above.
    for (auto rectangle_for_landmarks_inference : *merged_hand_rectangles) {
      // extract the sub-image implied by the established hand rectangles, for landmarks inference.
      // this sub-image will currently be 224x224 pixels.
      api2::ImageToTensorCoreResult extracted_sub_image_struct;
      MP_RETURN_IF_ERROR(sub_image_for_landmarks_inference_extractor_->Process(*image, rectangle_for_landmarks_inference, &extracted_sub_image_struct));
      MP_ASSIGN_OR_RETURN(std::vector<Tensor> output_tensors, landmarks_inference_->Process(MakeTensorSpan(extracted_sub_image_struct.tensors)));

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

      // extract the hand presence score from the landmarks inference output
      auto hand_presence_raw = tensors_to_floats_calculator_core::HandPresenceExtract(*inference_output_hand_presence, TensorsToFloatsCalculatorOptions());
      ABSL_ASSERT(hand_presence_raw.num_values == 1);
      float hand_presence_in_landmarks_inference = hand_presence_raw.output_floats->at(0);

      // gate naively by the hand presence detection which is part of the landmarks inference.
      // the palm detection phase v.s. this phase:
      //
      //   • palm detection seeks to detect a palm, it's currently hard to tell how much seeing fingers helps with that ―
      //     a tentative intuition might be that a lot, yet we don't really know how it was trained other than picking inside
      //     sneaky papers of detection models leading to mediapipe's palm detection model, and we don't a-priori know for now.
      //   • presence detection as part of the landmarks inference model assumes to see an entire hand sans allowing occlusions,
      //     we haven't assessed how accurate it thus far is in any way.
      //
      // currently, the latter is by the cascade from palm detection fed by the former's input and a naive tuned expansion of it.
      // elaborated flows and feedback loops can be envisioned which stage this differently and not necessarily in a linear
      // forward-in-time way only ― which can only be addressed as overall tracking improvement research under firm desiderata ―
      // first step would be to quantify and qualify fail cases into categories in a very intelligent manner ―
      // one that gives immediate rise to an evaluation metric which tracks flows between error categories
      // and the soft nature of the presence scoring which the current model yields.
      //
      // the current flow is a good (great) baseline of notably relatively few moving parts;
      // furthermore it's expansion ratios transitioning between the bouding rectangles may have been finely optimized.
      //
      // at the same time it feels a little brittle in e.g. not being more explicitly stateful across frames, in:
      //   - not taking exploit of kinematics (which hinges on anatomy here)
      //   - not being more explicitly stateful in perhaps other ways
      //   - not explicitly harmonizing the various bounding rectangle filtering stages
      //
      // well, as we know, good baselines are hard to beat without very disciplined effort;
      // further, what isn't explicitly harmonized may still be near optimal.
      if (hand_presence_in_landmarks_inference < hand_presence_in_landmarks_inference_threshold_) {
        ABSL_LOG(INFO) << "a rectangle for hand landmarks inference failed in presence validation by a presence post-hoc score of " << hand_presence_in_landmarks_inference;
        continue;
      } else {
        ABSL_LOG(INFO) << "presence validation value is " << hand_presence_in_landmarks_inference;
      }

      // extract the hand handedness classification object (object holding handedness value and its confidence) from the landmarks inference output
      auto inferred_handedness_score = inference_output_hand_handedness->at(0).GetCpuReadView().buffer<float>();
      std::unique_ptr<ClassificationList> inferred_handedness_classification_object = HandednessClassificationExtract(inferred_handedness_score, 2, handedness_classification_config_);

      // extract the viewport landmarks from the landmarks inference output
      NormalizedLandmarkList inferred_landmarks;
      MP_RETURN_IF_ERROR(landmarks_extractor_->OutputTensorsToLandmarks(*inference_output_viewport_landmarks, &inferred_landmarks));

      // unletterbox (aspect ratio scale back) the viewport landmarks inference coordinates (from the letterboxing effected by the sub-image extraction for landmarks inference)
      // notworthy, there is typically no letterboxing taking place (all letterboxing values are zero) as we pass a square of pixels of aspect ratio 1:1 to begin with,
      // in which case there is no need to neither scale nor letterbox our sub-image being passed for landmarks inference by the square assuming inference model.
      // you only need to letterbox if you pass in something that's not square at either the pixel dimensions or the rectangle dimensions ―
      // which we don't do ― unless we pass non-square rectangles when palm detections are near the viewport edges to be tested.

      // every so often the ensuing padding values are infenitesimal but not zero, which is likely just an unnecessary computational artefact ―
      // probably due to the cutting out of the hand rectangle which is a non-axes-parallel rectangle, in the sub-image extraction phase.
      if (std::any_of(extracted_sub_image_struct.padding.begin(), extracted_sub_image_struct.padding.end(),
                      [](float v) { return v > 0.0001f; })) {
        ABSL_LOG(INFO) << "non-zero letterbox padding detected: "
                       << "left " << extracted_sub_image_struct.padding[0]
                       << ", top " << extracted_sub_image_struct.padding[1]
                       << ", right " << extracted_sub_image_struct.padding[2]
                       << ", bottom " << extracted_sub_image_struct.padding[3]; }

      // apply the (mostly dummy) unletterboxing to the coordinates of the viewport landmarks.
      // lets anyway keep this for now, to avoid accumulating infinitesimal drift and apply the infinitesimal unletterboxing till we cleaned up
      // the letterboxing not to accumulate computationl drift, or analyzed the computational effect of it. it's just another fn call while
      // the time in this module is vastly dominated by the inference steps.
      NormalizedLandmarkList inferred_landmarks_unletterboxed = AdjustLandmarkListToLetterboxRemoval(inferred_landmarks, extracted_sub_image_struct.padding);

      // translate and rotate the (possibly unletterboxed) coordinates of the viewport landmarks to their viewport coordinates.
      // rotation is applied counter to the rotation applied when passing the sub-image for landmarks inference to landmarks inference.
      NormalizedLandmarkList final_viewport_landmarks;
      ToViewportCoordinates(inferred_landmarks_unletterboxed, &rectangle_for_landmarks_inference, &final_viewport_landmarks);

      // extract the object landmarks from the landmarks inference output
      auto inferred_object_landmarks = LandmarkList();
      MP_RETURN_IF_ERROR(api2::OutputTensorsToWorldLandmarks(*inference_output_object_landmarks, &inferred_object_landmarks));

      // rotate the coordinates of the object landmarks counter to the rotation applied when passing the sub-image for landmarks inference to landmarks inference
      LandmarkList final_object_landmarks = api3::RotateWorldLandmarks(inferred_object_landmarks, &rectangle_for_landmarks_inference);

      // get a rectangle more evenly encapsulating the hand according to (derived from) the final landmarks we got for the hand contained in it.
      // the previous rectangle we had was an oriented expansion of the an SSD detection rectangle, oriented by its 7 included palm key points.
      // but now that we have landmarks inference of 21 hand landmarks, we will use that information instead, as the basis for a rectangle more
      // properly hugging around the hand. this is the same as in the original pipeline's implementation.
      // we can probably improve on this as part of a larger "better tracking" epic.
      auto hand_rect_for_next_frame = std::make_unique<NormalizedRect>();
      MP_RETURN_IF_ERROR(AdjustHandRectByInferredLanmdarks(final_viewport_landmarks, std::make_pair(image->width(), image->height()), hand_rect_for_next_frame.get()));

      // expand the latter obtained rectangle for use as a candidate rectangle to apply landmarks inference to in the next frame,
      // as if assuming that if we got the landmarks pretty much accurately inferred, then assuming the next frame is close in time to the current one,
      // (relative to hand motion speed and naively agnostic of distance from the camera and more), then by enlarging that said rectangle by an amount
      // we'd capture the same hand again within that rectangle on the next camera frame.
      // of course, the contours and amount of expansion here are only a baseline tradeoff between being too small to miss some of the hand on the next
      // frame and between being too large to allow more noise pixels and have the hand smaller than trained for by the landmarks inference model ―
      // so we can assume this expansion was finely tuned by the original mediapipe team, who knows the sensitivity of the landmarks inference model
      // to the pixel sizes of hands showing in its 224x224 input, which depends (we can't know how much) on the same sizes in its training data.
      expand_rect_for_next_frame_->ExpandNormalizedRect(hand_rect_for_next_frame.get(), image->width(), image->height());

      // accumulate into vectors of results that we return to the caller -> like the original pipeline, like so:
      // each result type is a vector and all those result vectors are implicitly indexed by the hand to which they apply.
      // you may notice that this indexing does not in any way preserve "hand identity" since the implementation makes
      // no effort to associate hands detected, across frames! this shortcoming should be part of the tracking redo!
      // the implicit indexing *is* however technically consistent for the current detections:
      // each index across the result vectors corresponds to the same detected hand,
      // or at least arises from the implicit indexing of the model's inference outputs,
      // which are assumed to have that trait.
      result->viewport_landmarkss->push_back(final_viewport_landmarks);
      result->object_landmarkss->push_back(final_object_landmarks);
      result->handedness_classifications->push_back(*inferred_handedness_classification_object);

      hand_rect_from_previous_frame_.push_back(*hand_rect_for_next_frame);
    }

  return result;

  // return merged_hand_rectangles;
}

}
