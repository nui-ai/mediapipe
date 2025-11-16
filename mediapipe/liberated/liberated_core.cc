#include "mediapipe/liberated/liberated_core.h"

namespace hand_tracking_mp_lean {

/// object for driving the entire processing of images from an image stream, for hand tracking and inference;
/// formerly this was a mediapipe pipeline HandLandmarkTrackingCpu of mediapipe commit tag v0.10.13.
HandTrackingCore::HandTrackingCore(uint32_t max_hands_to_track, MemoryManager* memory_manager) {

  max_hands_to_track_ = max_hands_to_track;

  // initialize for converting the input image to a 192x192 grid for palm detection inference.
  // this cascade of argument setting can be simplified for less surface and the converters
  // can be made encapsulated by it, by simplifying ImageToTensorCalculatorCore for that.
  auto image_to_palm_detection_input_options = ImageToTensorCalculatorOptions();
  image_to_palm_detection_input_options.set_output_tensor_width(192);
  image_to_palm_detection_input_options.set_output_tensor_height(192);
  image_to_palm_detection_input_options.set_keep_aspect_ratio(true);
  image_to_palm_detection_input_options.mutable_output_tensor_float_range()->set_min(0.0f);
  image_to_palm_detection_input_options.mutable_output_tensor_float_range()->set_max(1.0f);
  image_to_palm_detection_input_options.set_border_mode(hand_tracking_mp_lean::ImageToTensorCalculatorOptions::BORDER_ZERO);
  auto params = GetOutputTensorParams(image_to_palm_detection_input_options);
  image_to_palm_detection_input_ = std::make_unique<api2::ImageToTensorCalculatorCore>(
      image_to_palm_detection_input_options, 192, 192, params,
      palm_detection_gpu_converter_, palm_detection_cpu_converter_, memory_manager);

  // initialize for palm detection inference
  const std::string& palm_detection_model_path = "mediapipe/modules/palm_detection/palm_detection_full.tflite";
  palm_detection_inference_ = std::make_unique<api2::ModelInference>(palm_detection_model_path);

  // initialize for extracting the raw palm detections inference outputs, and also for filtering them
  palm_detection_inference_filter_ = std::make_unique<api2::DetectionsExtractionAndFiltering>(0.5);

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

  // initialize for extracting the sub-image implied by each oriented hand rectangle.
  // this cascade of argument setting can be simplified for less surface and the converters
  // can be made encapsulated by it, by simplifying ImageToTensorCalculatorCore for that.
  auto sub_image_extraction_options = ImageToTensorCalculatorOptions();
  sub_image_extraction_options.set_output_tensor_width(224);
  sub_image_extraction_options.set_output_tensor_height(224);
  sub_image_extraction_options.set_keep_aspect_ratio(true);
  sub_image_extraction_options.mutable_output_tensor_float_range()->set_min(0.0f);
  sub_image_extraction_options.mutable_output_tensor_float_range()->set_max(1.0f);
  sub_image_extraction_options.set_border_mode(hand_tracking_mp_lean::ImageToTensorCalculatorOptions::BORDER_UNSPECIFIED);
  auto params_ = GetOutputTensorParams(sub_image_extraction_options);
  sub_image_for_landmarks_inference_extractor_ = std::make_unique<api2::ImageToTensorCalculatorCore>(
      sub_image_extraction_options, 224, 224, params_,
      landmarks_inference_gpu_converter_, landmarks_inference_cpu_converter_, memory_manager);

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
  hand_rects_from_previous_frame_ = std::vector<NormalizedRect>();
}

/// helper debug logging function
void HandTrackingCore::image_debug_logging(api2::ImageToTensorCoreResult *image_struct) {
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
    // since there's a lot of letterbox padding in placing a monitor aspect ratio image into a square,
    // we debug-log a small arbitrary middle section of the image rather than its beginning pixel values,
    // as the latter are all zero when the given image is the one prepared for palm inference.
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

/// helper debug logging function
void HandTrackingCore::sub_image_for_landmarks_inference_debug_logging(api2::ImageToTensorCoreResult *extracted_sub_image_struct) {
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

/// helper debug logging function
void HandTrackingCore::sub_image_padding_debug_logging(api2::ImageToTensorCoreResult* extracted_sub_image_struct) {
if (std::any_of(extracted_sub_image_struct->padding.begin(), extracted_sub_image_struct->padding.end(),
                [](float v) { return v > 0.0001f; })) {
  ABSL_LOG(INFO) << "non-zero letterbox padding: "
                      << extracted_sub_image_struct->padding[0] << extracted_sub_image_struct->padding[1]
                      << extracted_sub_image_struct->padding[2] << extracted_sub_image_struct->padding[3]; }
}

/// helper debug logging function
void HandTrackingCore::landmarks_inference_debug_logging(std::vector<Tensor> landmarks_inference_output_tensors) {
  ABSL_LOG(INFO) << "landmarks inference first few values (the viewport landmarks unnormalized): ";
  const auto& first_tensor_vals = landmarks_inference_output_tensors[0].GetCpuReadView().buffer<float>();
  for (int i = 0; i < 21; ++i) {
    std::cout << first_tensor_vals[i] << " ";
  }
  std::cout << std::endl;
}

/// this method performs weak tracking and inference of hand landmarks for hands, up to the initialization
/// given argument for the number of hands to track.
///
/// formerly this was a mediapipe pipeline HandLandmarkTrackingCpu of mediapipe commit tag v0.10.13,
/// from which the original logic was reverse-ported by trimming and ridding the many layers of excessive
/// generalization's that the pipeline calculators were wrapped and infused with, as the original pipeline
/// is reusing calculators having may dimensions of generalization for the sake of many usage cases which
/// are used by other unrelated mediapipe provided pipelines but act purely as "dead code" for the
/// hand tracking use case and pipeline.
///
/// note that the input argument for the number of hands to track currently restricts the number of hands which
/// which it returns for each frame to the amount of hands given upon initialization by that argument, as does
/// the original mediapipe pipeline; this is a baseline behavior which the original pipeline is also tuned for
/// in ways somewhat implicit in its workflow, even now that the workflow is much more transparent
/// to follow in the current pipeline-liberated implementation that the current class is.
///
/// tracking and hand identities can't be seen as separate in this domain, yet hand identities are only
/// made (very) weakly consistent in the original mediapipe pipeline which the current class re-implements:
///
/// they tend to keep tracking the same user hands which it has started tracking (started tracking at the onset)
/// rather than arbitrarily picking which hands to keep for each frame's output ― a property which it only softly
/// maintains and will fail to maintain that desireable trait in various edgy cases:
///
///   ✤ hands moved too fast relative to (the reverse of) distance from the camera and relative to the time distance between a pair of frames
///   ✤ after a frame where a hand tracked in the previous frame fails to register as present in a valid way in the current frame.
///   ⊛ when the scene is very noisy for the machine vision parts of this pipeline, of course, but that will make the tracking
///     unusable anyway, and is not what you'd focus on in the context of a next-gen of this algorithm.. e.g. if hand detection
///     often fails to follow the hands in basic ways deep in the pipeline, or non-hand skin like face areas register as hands,
///     but these situations should be best thought of as just things that a new algorithm would be immune to while planning
///     that new algorithm to tackle the former two cases.
///
/// the above description of the tracking behavior is not mirrored by any explicit branching logic of the original
/// pipeline nor the current implementation mirroring it, but is an analytic outcome of how the current workflow works.
///
/// the current implementation may partially maintain hand identity across frames in a weak manner which caters
/// for the majority of time with simple hand motion scenarios, yet is is not by any way guaranteed to be fully consistent
/// in that. in terms of how this reflects in api use, this weak identity of hands across frames is quite accordingly
/// only expressed (as noted, only weakly) by the indexing of the hand objects returned by the current function.
///
/// the success of this form of weak tracking however cannot be dismissed or refuted,
/// and it also enables easy user recovery.
///
/// alternatively, an implementation could be evolved to track hands in entirely different regimes:
///
///   ◎ not restrict hands number at all, but try to track their identity across frames always
///
///   ◎ home in on specific hands which are registered with user cooperation by a "hello" phase:
///       ◌ the user places their hands centered within a circle shown, or waves them etc to
///         identify them give or take allowing the system to calibrate their personal
///         geometry traits ("acquire control" from the user point of view).
///       ◌ other ways of acknowledging which hands to track,
///         explicit or more implicit than above.
///
///   ◎ knowing which hands to expect (and thus how many) sure can lead to robust tracking
///     flows within the algorithm
///
///   ◎ leave it entirely to the caller to implement the selection of hands from a plurality
///     of returned hand inferences ― this is a little weak since the caller may want to have
///     most of the information that the internal workflow has if it wishes to be
///     very smart about its identity tracking at each frame.
///
/// not currently ported or implemented:
/// - GPU inference.
/// - Inference on platforms which do not have solid XNNPACK support.
absl::StatusOr<std::unique_ptr<ImageHandTrackingAndInferenceResult>> HandTrackingCore::Process(std::shared_ptr<const Image> image) {

  // initiate the result structure for the current image as empty vectors for all of its fields
  auto result = std::make_unique<ImageHandTrackingAndInferenceResult>(
  ImageHandTrackingAndInferenceResult{
      std::make_unique<std::vector<NormalizedLandmarkList>>(),
      std::make_unique<std::vector<LandmarkList>>(),
      std::make_unique<std::vector<ClassificationList>>()
  });

  auto count_capped_detections = absl::make_unique<std::vector<Detection>>();
  auto hand_rects_from_detections = absl::make_unique<std::vector<NormalizedRect>>();
  auto merged_hand_rectangles_list = absl::make_unique<std::list<NormalizedRect>>();
  auto merged_hand_rectangles = absl::make_unique<std::vector<NormalizedRect>>();

  ABSL_LOG(INFO) << "hand rectangles from the previous frame's landmarks inferences: " << hand_rects_from_previous_frame_.size();
  if (hand_rects_from_previous_frame_.size() == max_hands_to_track_) {
    ABSL_LOG(INFO) << "skipping palm detection inference";
  } else if (hand_rects_from_previous_frame_.size() > max_hands_to_track_) {
    ABSL_LOG(INFO) << "(more hand rectangles from the previous frame's landmarks inferences than the number of hands to track " << max_hands_to_track_ << ")";
    ABSL_LOG(INFO) << "skipping palm detection inference";
  }

  // start the palm detection -> expanded oriented hand region for landmark inference path of computation
  if (hand_rects_from_previous_frame_.size() < max_hands_to_track_) {

    ABSL_LOG(INFO) << "palm detection inference is being invoked as there are not enough hand rectangles derived from the previous frame's landmarks inference";

    // turn the input image to a Tensor of the right size for the palm detection model ― this typically involves letterboxing as in the input image is typically not square
    api2::ImageToTensorCoreResult image_as_tensor;
    absl::optional<NormalizedRect> norm_rect = absl::nullopt;
    MP_RETURN_IF_ERROR(image_to_palm_detection_input_->Process(*image, norm_rect, &image_as_tensor));
    // image_debug_logging(&image_as_tensor);
    auto letterbox_padding_ = image_as_tensor.padding;
    TensorSpan image_as_tensor_span;
    image_as_tensor_span = MakeTensorSpan(image_as_tensor.tensors);


    // palm detection inference
    auto palm_start_time = std::chrono::high_resolution_clock::now();
    absl::StatusOr<std::vector<Tensor>> palm_detection_inference_output;
    MP_ASSIGN_OR_RETURN(palm_detection_inference_output, palm_detection_inference_->Process(image_as_tensor_span));
    auto palm_end_time = std::chrono::high_resolution_clock::now();
    auto palm_duration_us = std::chrono::duration_cast<std::chrono::microseconds>(palm_end_time - palm_start_time).count();
    ABSL_LOG(INFO) << "palm detection inference took (ms): " << (static_cast<double>(palm_duration_us) / 1000.0);

    // extract-decode the detection inference output
    std::unique_ptr<std::vector<Detection>> letterboxed_detections;
    MP_ASSIGN_OR_RETURN(letterboxed_detections, palm_detection_inference_filter_->Extract(*palm_detection_inference_output));

    // filter the extracted detections by their detection score and by NMS.
    std::unique_ptr<std::vector<Detection>> filtered_letterboxed_detections;
    MP_ASSIGN_OR_RETURN(filtered_letterboxed_detections, palm_detection_inference_filter_->Filter(*letterboxed_detections));

    // unletterbox the surviving filtered detections to the original image coordinates.
    std::unique_ptr<std::vector<Detection>> filtered_detections = UnLetterBox(*filtered_letterboxed_detections, letterbox_padding_);

    // extremely naively clip the number of detections surviving the previous filtering, to the constant number of hands to be tracked;
    // this was part of the original pipeline (as a ClipVectorSizeCalculator subclass calculator there) as an obviously naive way
    // of enforcing the number of hands being output to the overall requested number of hands to be tracked, but does not really
    // have to be handled in isolation of the plethora of state that can be used for more advanced or softer handling:
    //   ◎ the locations, presence scores, handedness confidence etc. of the hands known from previous frames.
    //   ◎ the locations, presence scores, handedness confidence etc. of the current frame (i.e. don't clip so soon here).
    // ---------------------------------------------------------------------------------------------------------------------
    // further, the filtering does not have to be linear in time but may even look back in time in some downstream scenarios
    auto excessive_detections_count = static_cast<long>(filtered_detections->size()) - static_cast<long>(max_hands_to_track_);
    if ( excessive_detections_count > 0) {
      ABSL_LOG(INFO) << "naively discarded " << excessive_detections_count << "palm detections to keep only the same number of them as the set number of hands for tracking (" << max_hands_to_track_ << ")";
      for (int i = 0; i < max_hands_to_track_; ++i) {
        count_capped_detections->push_back(filtered_detections->at(i));
      }
    } else {
      for (int i = 0; i < filtered_detections->size(); ++i) {
        count_capped_detections->push_back(filtered_detections->at(i));
      }
    }

    // orient the palm detections all by one function call which takes all of them
    std::vector<NormalizedRect> oriented_palm_norm_rects;
    std::vector<Rect> oriented_palm_rects; // unused output argument required by the below function in its current legacy form
    auto image_size = std::make_pair(image->width(), image->height());
    MP_RETURN_IF_ERROR(palm_detection_to_oriented_palm_rect_->OrientedRectsFromDetections(*count_capped_detections, image_size, &oriented_palm_norm_rects, &oriented_palm_rects));

    // expand the now oriented palm detections such that they will likely contain the entire hand (meaning palm + fingers).
    // technically speaking unlike the former step, we loop each rect here rather than inside the expanding function.
    hand_rects_from_detections = absl::make_unique<std::vector<NormalizedRect>>(oriented_palm_rects.size());
    for (int i = 0; i < oriented_palm_rects.size(); ++i) {
      // copy the rectangle
      hand_rects_from_detections->at(i) = oriented_palm_norm_rects[i];
      // expand the rectangle
      auto it = hand_rects_from_detections->begin() + i;
      oriented_palm_rect_to_hand_rect_expander_->ExpandNormalizedRect(&(*it), image->width(), image->height());
    }
  }

  // merges with IoU threshold based filtering, the set of hand rects derived directly from palm detection inference, with the set of hand rects derived from the previous frame's landmarks inference,
  // both of which input sets to this merge may be empty or not. if both are empty, an empty set should be the result.
  MP_ASSIGN_OR_RETURN(*merged_hand_rectangles_list, hand_tracking_mp_lean::IouFilterMerge(*hand_rects_from_detections, hand_rects_from_previous_frame_, 0.5));
  merged_hand_rectangles = absl::make_unique<std::vector<NormalizedRect>>(merged_hand_rectangles_list->begin(), merged_hand_rectangles_list->end());  // convert from list to vector

  // reset the list of rectangles passed from the previous frame's pass, before we start building one from scratch for the next frame ...
  hand_rects_from_previous_frame_ = std::vector<NormalizedRect>();

  // start looping or fanning out in place of the original pipeline's fanning out of the hand rects for landmarks inference, which have been accomplished above.
  for (auto rectangle_for_landmarks_inference : *merged_hand_rectangles) {

    // extract the sub-image implied by the established hand rectangle, for landmarks inference.
    // this sub-image will currently be 224x224 pixels, regardless the orientation of the hand rectangle which is typically not axes parallel.
    // as there is more than one way to extract a 224x224 grid from a rotated rectangle, this step is a little sensitive to implementation details,
    // which may ultimately slightly change the landmarks inference from the extracted grid, but presumably the outcome will not differ by *too much*
    // by just those small differences in antialiasing and padding the oriented rectangle from the original image: indeed when mistakenly using an
    // unintended converter (the one used to extract the entire image for palm detection) for extracting these sub-images for landmarks inference,
    // only about 1/1000 of hand inferences got different landmarks than they get with the intended converter, i.e. ~0.999 of the time the final
    // landmarks inference of a hand was identical despite the different extraction method details which used a different border extraction style.
    api2::ImageToTensorCoreResult extracted_sub_image_struct;
    MP_RETURN_IF_ERROR(sub_image_for_landmarks_inference_extractor_->Process(*image, rectangle_for_landmarks_inference, &extracted_sub_image_struct));

    // sub_image_for_landmarks_inference_debug_logging(&extracted_sub_image_struct);

    // perform landmarks inference over the provided sub-image
    auto start_time = std::chrono::high_resolution_clock::now();
    auto start_time_us = std::chrono::duration_cast<std::chrono::microseconds>(start_time.time_since_epoch()).count();
    MP_ASSIGN_OR_RETURN(std::vector<Tensor> landmarks_inference_output_tensors, landmarks_inference_->Process(MakeTensorSpan(extracted_sub_image_struct.tensors)));
    auto end_time = std::chrono::high_resolution_clock::now();
    auto end_time_us = std::chrono::duration_cast<std::chrono::microseconds>(end_time.time_since_epoch()).count();
    auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
    ABSL_LOG(INFO) << "landmarks inference over the given sub-image took (ms): " << (static_cast<double>(duration_us) / 1000.0);
    // debug logging: landmarks_inference_debug_logging(landmarks_inference_output_tensors);

    // get a unique pointer to output_tensors that can be passed to a function expecting std::unique_ptr<std::vector<T>>*
    auto landmarks_inference_output_tensors_ptr = std::make_unique<std::vector<Tensor>>(std::move(landmarks_inference_output_tensors));

    // split the result of the landmarks inference into topics (this can be much distilled to discard the over-generalized Run method entirely and just assign the inference outputs directly)
    std::vector<Tensor> output_elements_ignored;  // output argument not consumed by our code (an over generalization feature of the original pipeline)
    std::unique_ptr<std::vector<Tensor>> combined_output_ignored;  // output argument not consumed by our code (an over generalization feature of the original pipeline)
    std::vector<std::unique_ptr<std::vector<Tensor>>> output_vectors;
    MP_RETURN_IF_ERROR(landmarks_inference_splitter_->Run(&landmarks_inference_output_tensors_ptr, &output_vectors, &output_elements_ignored, &combined_output_ignored));
    std::unique_ptr<std::vector<Tensor>> inference_output_viewport_landmarks = std::move(output_vectors[0]);
    std::unique_ptr<std::vector<Tensor>> inference_output_hand_presence = std::move(output_vectors[1]);
    std::unique_ptr<std::vector<Tensor>> inference_output_hand_handedness = std::move(output_vectors[2]);
    std::unique_ptr<std::vector<Tensor>> inference_output_object_landmarks = std::move(output_vectors[3]);

    // extract the hand presence score from the landmarks inference output
    auto hand_presence_raw = tensors_to_floats_calculator_core::HandPresenceExtract(*inference_output_hand_presence, TensorsToFloatsCalculatorOptions());
    ABSL_ASSERT(hand_presence_raw.status.ok() && hand_presence_raw.num_values == 1);
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
    ABSL_LOG(INFO) << "hand rectangle presence validation score from landmarks inference is " << hand_presence_in_landmarks_inference;
    if (hand_presence_in_landmarks_inference < hand_presence_in_landmarks_inference_threshold_) {
      ABSL_LOG(INFO) << "hand rectangle failed in presence validation by landmarks inference and is being ignored";
      continue;
    }

    // extract the hand handedness classification object (object holding handedness value and its confidence) from the landmarks inference output
    auto inferred_handedness_score = inference_output_hand_handedness->at(0).GetCpuReadView().buffer<float>();
    std::unique_ptr<ClassificationList> inferred_handedness_classification_object = HandednessClassificationExtract(inferred_handedness_score, handedness_classification_config_);

    // extract the viewport landmarks from the landmarks inference output
    NormalizedLandmarkList inferred_landmarks;
    MP_RETURN_IF_ERROR(landmarks_extractor_->OutputTensorsToLandmarks(*inference_output_viewport_landmarks, &inferred_landmarks));

    // unletterbox (aspect ratio scale back) the viewport landmarks inference coordinates (from the letterboxing effected by the sub-image extraction for landmarks inference)
    // notworthy, there is typically no letterboxing taking place (all letterboxing values are zero) as we pass a square of pixels of aspect ratio 1:1 to begin with,
    // in which case there is no need to neither scale nor letterbox our sub-image being passed for landmarks inference by the square assuming inference model.
    // you only need to letterbox if you pass in something that's not square at either the pixel dimensions or the rectangle dimensions ―
    // which we don't do ― unless we pass non-square rectangles when palm detections are near the viewport edges to be tested.

    sub_image_padding_debug_logging(&extracted_sub_image_struct);

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
    // we'd capture the same hand again within that rectangle, on the next camera frame.
    // of course, the contours and amount of expansion here are only a baseline tradeoff between being too small to miss some of the hand on the next
    // frame and between being too large to allow more noise pixels and have the hand smaller than trained for by the landmarks inference model ―
    // so we can assume this expansion was finely tuned by the original mediapipe team, who knows the sensitivity of the landmarks inference model
    // to the pixel sizes of hands showing in its 224x224 input, which depends (we can't know how much) on the same sizes in its training data.
    expand_rect_for_next_frame_->ExpandNormalizedRect(hand_rect_for_next_frame.get(), image->width(), image->height());

    // accumulate into vectors of results that we return to the caller -> like the original pipeline, like so:
    // each result type is a vector and all those result vectors are implicitly indexed by the hand to which they apply.
    // this indexing is not guaranteed to fully preserve "hand identity" across frames as the implementation makes no
    // explicit effort to associate hands detected ― across frames ― other than perhaps as only an artefact of SSD
    // detection boxes ordering when the hands are kept each at its distinct region of the viewport.
    //
    // this implicit indexing *is* only technically consistent within the humble scope of the current pipeline result:
    //   • each index across the contained result vectors corresponds to the same detected hand,
    //     or at least arises from the implicit indexing of the model's inference outputs as such.
    result->viewport_landmarkss->push_back(final_viewport_landmarks);
    result->object_landmarkss->push_back(final_object_landmarks);
    result->handedness_classifications->push_back(*inferred_handedness_classification_object);

    hand_rects_from_previous_frame_.push_back(*hand_rect_for_next_frame);

    ++call_counter_;
  }

  return result;
}

}
