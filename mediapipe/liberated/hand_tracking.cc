#if !defined(HAND_TRACKING_VERBOSE_LOGGING)
#define HAND_TRACKING_VERBOSE_LOGGING 0
#endif
#define HAND_TRACKING_LOG(msg) \
  do { if (HAND_TRACKING_VERBOSE_LOGGING) { ABSL_LOG(INFO) << msg; } } while(0)

#include "mediapipe/liberated/hand_tracking.h"
#include "mediapipe/liberated/hand_tracking_debug.h"

namespace hand_tracking_mp_lean {

  /// object for driving the entire processing of images from an image stream, for hand tracking and inference;
  /// formerly this was a mediapipe pipeline HandLandmarkTrackingCpu of mediapipe commit tag v0.10.13.
  HandTrackingCore::HandTrackingCore(uint32_t max_hands_to_track, const std::string* models_path) {

    // MemoryManager is a class reusing memory one-time allocated from the OS, meant for making more performant repeat tensor allocations.
    // we carried forward and persevere its use from the original pipeline implementation, as indeed its use still trickles down into cored
    // components that we employ by the current class.
    //
    // our execution time is by far dominated by the tensorflow inference steps ― which may or may not benefit as much from this optimization
    // outside of resource-constrained embedded environments which we don't do yet.
    //
    // reusing this component in place of plain mallocs probably doesn't hurt, unless instantiating one per HandTrackingCore
    // is somehow non-optimal in more parallelized setups using multiple HandTrackingCore inference engines in parallel.
    // MemoryManager is a simple pass-through facade over platoform-specific such memory allocation implementations
    // (so it's a nice to have for future platform support as such, but can be avoided if necessary).
    auto memory_manager = MemoryManager();

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
        palm_detection_gpu_converter_, palm_detection_cpu_converter_, &memory_manager);

    // initialize for palm detection inference
    const std::string& palm_detection_model_path = "mediapipe/modules/palm_detection/palm_detection_full.tflite";
    palm_detection_inference_ = std::make_unique<api2::ModelInference>(palm_detection_model_path, models_path);

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
        landmarks_inference_gpu_converter_, landmarks_inference_cpu_converter_, &memory_manager);

    // initialize for landmarks inference
    const std::string& landmarks_infernce_model_path = "mediapipe/modules/hand_landmark/hand_landmark_full.tflite";
    landmarks_inference_ = std::make_unique<api2::ModelInference>(landmarks_infernce_model_path, models_path);

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
    auto landmarks_derived_hand_rect_expander_options = RectTransformationCalculatorOptions();
    ABSL_HARDENING_ASSERT(!(landmarks_derived_hand_rect_expander_options.has_rotation() && landmarks_derived_hand_rect_expander_options.has_rotation_degrees()));
    ABSL_HARDENING_ASSERT(!(landmarks_derived_hand_rect_expander_options.has_square_long() && landmarks_derived_hand_rect_expander_options.has_square_short()));
    landmarks_derived_hand_rect_expander_options.set_scale_x(2.0f);
    landmarks_derived_hand_rect_expander_options.set_scale_y(2.0f);
    landmarks_derived_hand_rect_expander_options.set_shift_y(-0.1f);
    landmarks_derived_hand_rect_expander_options.set_square_long(true);
    landmarks_derived_hand_rect_expander_ = std::make_unique<RectTransformation>(landmarks_derived_hand_rect_expander_options);

    // on the very first video frame there are no hand rectangles derived from the previous frame
    hand_rects_from_previous_frame_ = std::vector<NormalizedRect>();
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
  absl::StatusOr<std::unique_ptr<ImageHandTrackingResult>> HandTrackingCore::Process(const std::shared_ptr<const Image>& image) {

    // initiate the result structure returned by this function.
    auto result = std::make_unique<ImageHandTrackingResult>();

    std::vector<NormalizedRect> hand_rects_from_detection;
    auto merged_hand_rectangles_list = absl::make_unique<std::list<NormalizedRect>>();
    auto merged_hand_rectangles = absl::make_unique<std::vector<NormalizedRect>>();

    // ABSL_LOG(INFO) << "hand tracking lifecycle: hand rectangles from the previous frame's landmarks inferences: " << hand_rects_from_previous_frame_.size();
    if (hand_rects_from_previous_frame_.size() == max_hands_to_track_) {
      //ABSL_LOG(INFO) << "hand tracking lifecycle: skipping palm detection inference";
    } else if (hand_rects_from_previous_frame_.size() > max_hands_to_track_) {
      //ABSL_LOG(INFO) << "hand tracking lifecycle: (more hand rectangles from the previous frame's landmarks inferences than the number of hands to track " << max_hands_to_track_ << ")";
      //ABSL_LOG(INFO) << "hand tracking lifecycle: skipping palm detection inference";
    }
    if (hand_rects_from_previous_frame_.size() < max_hands_to_track_) {

      HAND_TRACKING_LOG("hand tracking lifecycle: palm detection inference is being invoked as there are not enough hand rectangles derived from the previous frame's landmarks inference");

      auto surviving_detections = absl::make_unique<std::vector<Detection>>();

      // turn the input image to a Tensor of the right size for the palm detection model ― this typically involves letterboxing as in the input image is typically not square
      api2::ImageToTensorCoreResult image_as_tensor;
      absl::optional<NormalizedRect> norm_rect = absl::nullopt;
      MP_RETURN_IF_ERROR(image_to_palm_detection_input_->Process(*image, norm_rect, &image_as_tensor));
      // image_debug_logging(&image_as_tensor);
      auto letterbox_padding_ = image_as_tensor.padding;
      TensorSpan image_as_tensor_span;
      image_as_tensor_span = MakeTensorSpan(image_as_tensor.tensors);

      // invoke palm detection inference
      auto palm_start_time = std::chrono::high_resolution_clock::now();
      absl::StatusOr<std::vector<Tensor>> palm_detection_inference_output;
      MP_ASSIGN_OR_RETURN(palm_detection_inference_output, palm_detection_inference_->Process(image_as_tensor_span));
      auto palm_end_time = std::chrono::high_resolution_clock::now();
      auto palm_duration_us = std::chrono::duration_cast<std::chrono::microseconds>(palm_end_time - palm_start_time).count();
      HAND_TRACKING_LOG(std::string("hand tracking lifecycle: palm detection inference took (ms): ") + std::to_string(static_cast<double>(palm_duration_us) / 1000.0));

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
      // further, the filtering does not have to be linear in time but may even look back in time in some downstream scenarios.
      // this step is both subject-matter naive and not memory optimized.
      auto excessive_detections_count = static_cast<long>(filtered_detections->size()) - static_cast<long>(max_hands_to_track_);
      if ( excessive_detections_count > 0) {
        HAND_TRACKING_LOG(std::string("hand tracking lifecycle: naively discarded ") + std::to_string(excessive_detections_count) + "palm detections to keep only the same number of them as the set number of hands for tracking (" + std::to_string(max_hands_to_track_) + ")");
        for (int i = 0; i < max_hands_to_track_; ++i) { surviving_detections->push_back(filtered_detections->at(i)); }
      } else {
        for (int i = 0; i < filtered_detections->size(); ++i) { surviving_detections->push_back(filtered_detections->at(i)); }
      }

      auto surviving_detections_count = surviving_detections->size();
      for (auto &detection: *surviving_detections) { HAND_TRACKING_LOG(std::string("hand tracking lifecycle: palm detection has confidence score ") + std::to_string(detection.score()[0])); }

      // a struct to hold the information being derived from the current detection
      auto detection_information = DetectionInformation();

      // per detection, commit the detection score into the final result
      for (int i = 0; i < surviving_detections_count; ++i) {
        detection_information.palm_detection_score = surviving_detections->at(i).score()[0];
      }

      // per raw detection, commit the raw detection rectangle into the final result
      for (int i = 0; i < surviving_detections_count; ++i) {
        ABSL_HARDENING_ASSERT(surviving_detections->at(i).has_location_data() && surviving_detections->at(i).location_data().has_relative_bounding_box());
        const auto& bb = surviving_detections->at(i).location_data().relative_bounding_box();
        const float xmin = bb.xmin();
        const float ymin = bb.ymin();
        const float w = bb.width();
        const float h = bb.height();
        detection_information.detected = RectGeometry{
          xmin + w/2,
          ymin + h/2,
          w,
          h,
          0.0f};
      }

      // per detection, orient the surviving palm detection and commit the resulting oriented rectangle into the final esult.
      std::vector<NormalizedRect> oriented_bbs;
      std::vector<Rect> unused; // unused output argument still required by the below function in its current legacy form
      auto image_size = std::make_pair(image->width(), image->height());
      MP_RETURN_IF_ERROR(palm_detection_to_oriented_palm_rect_->OrientedRectsFromDetections(*surviving_detections, image_size, &oriented_bbs, &unused));
      ABSL_HARDENING_ASSERT(oriented_bbs.size() == surviving_detections_count);
      for (int i = 0; i < surviving_detections_count; ++i) {
        NormalizedRect expanded = oriented_bbs[i];
        oriented_palm_rect_to_hand_rect_expander_->ExpandNormalizedRect(&expanded, image->width(), image->height());
        detection_information.oriented = RectGeometry{
          oriented_bbs[i].x_center(),
          oriented_bbs[i].y_center(),
          oriented_bbs[i].width(),
          oriented_bbs[i].height(),
          oriented_bbs[i].rotation()
        };
      }

      // expand each oriented palm detection such that it will likely contain the entire hand (meaning palm + fingers) and commit the resulting rectangles into the result.
      for (int i = 0; i < surviving_detections_count; ++i) {
        // expand the oriented rect to produce the hand rect used for landmarks inference
        NormalizedRect expanded_bb = oriented_bbs[i];
        oriented_palm_rect_to_hand_rect_expander_->ExpandNormalizedRect(&expanded_bb, image->width(), image->height());
        detection_information.expanded = RectGeometry{
          expanded_bb.x_center(),
          expanded_bb.y_center(),
          expanded_bb.width(),
          expanded_bb.height(),
          expanded_bb.rotation()
        };

        // keep it for use by landmarks inference
        hand_rects_from_detection.push_back(expanded_bb);

        // append it to the current function's process-transparency part of its returned output,
        // i.e. this information is internal to its implementation, but judicously exposed to
        // the caller for e.g. visualization of it.
        result->detections_information->push_back(detection_information);
      }
    }

    // merges with IoU threshold based filtering, the set of hand rects derived directly from palm detection inference, with the set of hand rects derived from the previous frame's landmarks inference,
    // both of which input sets to this merge may be empty or not. if both are empty, an empty set should be the result.
    MP_ASSIGN_OR_RETURN(*merged_hand_rectangles_list, hand_tracking_mp_lean::IouFilterMerge(hand_rects_from_detection, hand_rects_from_previous_frame_, 0.5));
    merged_hand_rectangles = absl::make_unique<std::vector<NormalizedRect>>(merged_hand_rectangles_list->begin(), merged_hand_rectangles_list->end());  // convert from list to vector

    // having consumed it just above, reset the list of rectangles passed from the previous frame's call into the current function,
    // before we build it from scratch for the next frame as part of looping the merged list of hand rectangles for landmarks detection ...
    hand_rects_from_previous_frame_ = std::vector<NormalizedRect>();

    // loop merged list of hand rectangles for landmarks detection ... invoking landmarks inference and processing its output per each one.
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
      HAND_TRACKING_LOG(std::string("hand tracking lifecycle: landmarks inference over the given sub-image took (ms): ") + std::to_string(static_cast<double>(duration_us) / 1000.0));
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

      HAND_TRACKING_LOG(std::string("hand tracking lifecycle: hand rectangle presence validation score from landmarks inference is ") + std::to_string(hand_presence_in_landmarks_inference));
      if (hand_presence_in_landmarks_inference < hand_presence_in_landmarks_inference_threshold_) {
        HAND_TRACKING_LOG("hand tracking lifecycle: hand rectangle failed in presence validation by landmarks inference and is being ignored");
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
      NormalizedLandmarkList inferred_landmarks_unletterboxed = AdjustLandmarkListToLetterboxRemoval(inferred_landmarks, extracted_sub_image_struct.padding);

      // translate and rotate the (possibly unletterboxed) coordinates of the viewport landmarks to their viewport coordinates.
      // rotation is applied counter to the rotation applied when passing the sub-image for landmarks inference.
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
      landmarks_derived_hand_rect_expander_->ExpandNormalizedRect(hand_rect_for_next_frame.get(), image->width(), image->height());

      // accumulate into vectors of results that we return to the caller -> like the original pipeline, like so:
      // each result type is a vector and all those result vectors are implicitly indexed by the hand to which they apply.
      // this indexing is not guaranteed to fully preserve "hand identity" across frames as the implementation makes no
      // explicit effort to associate hands detected ― across frames ― other than perhaps as only an artefact of SSD
      // detection boxes ordering when the hands are kept each at its distinct region of the viewport.
      //
      // this implicit indexing *IS* only technically consistent within the humble scope of the current pipeline result:
      //   • each index across the contained result vectors corresponds to the same detected hand,
      //     or at least arises from the implicit indexing of the model's inference outputs as such.
      result->viewport_landmarkss->push_back(final_viewport_landmarks);
      result->object_landmarkss->push_back(final_object_landmarks);
      result->handedness_classifications->push_back(*inferred_handedness_classification_object);
      result->landmarks_derived_hand_presence_scores->push_back(hand_presence_in_landmarks_inference);

      hand_rects_from_previous_frame_.push_back(*hand_rect_for_next_frame);
    }

    ++call_counter_;
    return result;
  }
}