// Copyright 2019 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// decodes (extracts) detection results from the SSD palm detection model's raw output,
// while filtering them down based on: detection validity, score thresholding, the requested maximum number
// of detections to pass forward, and lastly by non-maximum suppression (NMS).
//
// the filtering steps can sure be reorganized for better cohesion from the original inherited current form,
// but at least they are documented each inline.

#include <unordered_map>
#include <vector>

#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "mediapipe/calculators/tensor/tensors_to_detections_calculator.pb.h"
#include "mediapipe/calculators/tensor/detections_extraction.h"
#include "mediapipe/calculators/tflite/ssd_anchors_calculator_utils.h"
#include "mediapipe/calculators/util/non_max_suppression_calculator.pb.h"
#include "mediapipe/calculators/util/non_max_suppression_calculator.h"
#include "mediapipe/framework/api2/node.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/formats/location.h"
#include "mediapipe/framework/formats/object_detection/anchor.pb.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/port.h"
#include "mediapipe/framework/port/ret_check.h"

// Note: On Apple platforms MEDIAPIPE_DISABLE_GL_COMPUTE is automatically
// defined in mediapipe/framework/port.h. Therefore,
// "#ifndef MEDIAPIPE_DISABLE_GL_COMPUTE" and "#if MEDIAPIPE_METAL_ENABLED"
// below are mutually exclusive.
#ifndef MEDIAPIPE_DISABLE_GL_COMPUTE
#include "mediapipe/gpu/gl_calculator_helper.h"
#endif  // !defined(MEDIAPIPE_DISABLE_GL_COMPUTE)

#if MEDIAPIPE_METAL_ENABLED
#import <CoreVideo/CoreVideo.h>
#import <Metal/Metal.h>
#import <MetalKit/MetalKit.h>

#include "mediapipe/framework/formats/tensor_mtl_buffer_view.h"
#import "mediapipe/gpu/MPPMetalHelper.h"
#include "mediapipe/gpu/MPPMetalUtil.h"
#endif  // MEDIAPIPE_METAL_ENABLED

namespace {
constexpr int kNumCoordsPerBox = 4;

bool CanUseGpu() {
#if !defined(MEDIAPIPE_DISABLE_GL_COMPUTE) || MEDIAPIPE_METAL_ENABLED
  // TODO: Configure GPU usage policy in individual calculators.
  constexpr bool kAllowGpuProcessing = true;
  return kAllowGpuProcessing;
#else
  return false;
#endif  // !defined(MEDIAPIPE_DISABLE_GL_COMPUTE) || MEDIAPIPE_METAL_ENABLED
}
}  // namespace

namespace mediapipe_v01013_based {
namespace api2 {
  using BoxFormat = ::mediapipe_v01013_based::TensorsToDetectionsCalculatorOptions::BoxFormat;

  namespace {

    void ConvertRawValuesToAnchors(const float* raw_anchors, int num_boxes,
                                   std::vector<Anchor>* anchors) {
      anchors->clear();
      for (int i = 0; i < num_boxes; ++i) {
        Anchor new_anchor;
        new_anchor.set_y_center(raw_anchors[i * kNumCoordsPerBox + 0]);
        new_anchor.set_x_center(raw_anchors[i * kNumCoordsPerBox + 1]);
        new_anchor.set_h(raw_anchors[i * kNumCoordsPerBox + 2]);
        new_anchor.set_w(raw_anchors[i * kNumCoordsPerBox + 3]);
        anchors->push_back(new_anchor);
      }
    }

    void ConvertAnchorsToRawValues(const std::vector<Anchor>& anchors,
                                   int num_boxes, float* raw_anchors) {
      ABSL_CHECK_EQ(anchors.size(), num_boxes);
      int box = 0;
      for (const auto& anchor : anchors) {
        raw_anchors[box * kNumCoordsPerBox + 0] = anchor.y_center();
        raw_anchors[box * kNumCoordsPerBox + 1] = anchor.x_center();
        raw_anchors[box * kNumCoordsPerBox + 2] = anchor.h();
        raw_anchors[box * kNumCoordsPerBox + 3] = anchor.w();
        ++box;
      }
    }

    absl::Status CheckCustomTensorMapping(
        const TensorsToDetectionsCalculatorOptions::TensorMapping& tensor_mapping) {
      RET_CHECK(tensor_mapping.has_detections_tensor_index() &&
                tensor_mapping.has_scores_tensor_index());
      int bitmap = 0;
      bitmap |= 1 << tensor_mapping.detections_tensor_index();
      bitmap |= 1 << tensor_mapping.scores_tensor_index();
      if (!tensor_mapping.has_num_detections_tensor_index() &&
          !tensor_mapping.has_classes_tensor_index() &&
          !tensor_mapping.has_anchors_tensor_index()) {
        // Only allows the output tensor index 0 and 1 to be occupied.
        RET_CHECK_EQ(3, bitmap) << "The custom output tensor indices should only "
                                   "cover index 0 and 1.";
          } else if (tensor_mapping.has_anchors_tensor_index()) {
            RET_CHECK(!tensor_mapping.has_classes_tensor_index() &&
                      !tensor_mapping.has_num_detections_tensor_index());
            bitmap |= 1 << tensor_mapping.anchors_tensor_index();
            // If the"anchors" tensor will be available, only allows the output tensor
            // index 0, 1, 2 to be occupied.
            RET_CHECK_EQ(7, bitmap) << "The custom output tensor indices should only "
                                       "cover index 0, 1 and 2.";
          } else {
            RET_CHECK(tensor_mapping.has_classes_tensor_index() &&
                      tensor_mapping.has_num_detections_tensor_index());
            // If the "classes" and the "number of detections" tensors will be
            // available, only allows the output tensor index 0, 1, 2, 3 to be occupied.
            bitmap |= 1 << tensor_mapping.classes_tensor_index();
            bitmap |= 1 << tensor_mapping.num_detections_tensor_index();
            RET_CHECK_EQ(15, bitmap) << "The custom output tensor indices should only "
                                        "cover index 0, 1, 2 and 3.";
          }
      return absl::OkStatus();
    }

    BoxFormat GetBoxFormat(const TensorsToDetectionsCalculatorOptions& options) {
      if (options.has_box_format()) {
        return options.box_format();
      } else if (options.reverse_output_order()) {
        return mediapipe_v01013_based::TensorsToDetectionsCalculatorOptions::XYWH;
      }
      return mediapipe_v01013_based::TensorsToDetectionsCalculatorOptions::YXHW;
    }

  }  // namespace

  // Converts output tensors from the palm detection model into MediaPipe Detections.
  //
  // Input:
  //  TENSORS - Vector of Tensors of type kFloat32. The vector of tensors can have
  //            2 or 3 tensors. First tensor is the predicted raw boxes/keypoints.
  //            The size of the values must be (num_boxes * num_predicted_values).
  //            Second tensor is the score tensor. The size of the values must be
  //            (num_boxes * num_classes). It's optional to pass in a third tensor
  //            for anchors (e.g. for SSD models) depend on the outputs of the
  //            detection model. The size of anchor tensor must be (num_boxes *
  //            4).
  //
  //  IGNORE_CLASSES (optional) - The list of class ids that should be ignored, as
  //      a vector of integers. It overrides the corresponding field in the
  //      calculator options.
  //
  // Output:
  //  DETECTIONS - Result MediaPipe detections.
  //
  ExtractValidDetections::ExtractValidDetections(float score_threshold) {
    ABSL_CHECK_OK(SetDecodingParameters(score_threshold));
    ABSL_CHECK_OK(SetNmsParameters());

    if (CanUseGpu()) {
#ifndef MEDIAPIPE_DISABLE_GL_COMPUTE
#elif MEDIAPIPE_METAL_ENABLED
      gpu_helper_ = [[MPPMetalHelper alloc] initWithCalculatorContext:cc];
      RET_CHECK(gpu_helper_);
#endif  // !defined(MEDIAPIPE_DISABLE_GL_COMPUTE)
    }
  }

  /// decodes the raw SSD neural network outputs, which is a lot of technical pedantic work, into mediapipe objects
  absl::StatusOr<std::unique_ptr<std::vector<Detection>>> ExtractValidDetections::Extract(const std::vector<Tensor>& input_tensors) {
    for (const auto& tensor : input_tensors) { RET_CHECK(tensor.element_type() == Tensor::ElementType::kFloat32); }

    // extract from the model's output all but detections which are invalid (have a zero or a nan box dimension)
    auto valid_extracted_detections = absl::make_unique<std::vector<Detection>>();
    MP_RETURN_IF_ERROR(ExtractDetections(valid_extracted_detections.get(), input_tensors));
    ABSL_LOG(INFO) << "valid SSD extracted detections:  " << valid_extracted_detections->size();

    return valid_extracted_detections;
  }

  /// filters the extracted detections
  absl::StatusOr<std::unique_ptr<std::vector<Detection>>> ExtractValidDetections::Filter(const std::vector<Detection>& detections) {

    // filter them by threshold score
    auto score_thresholded_detections = absl::make_unique<std::vector<Detection>>();
    for (const Detection& detection : detections) {
      if (detection.score()[0] >= ssd_decoding_options_.min_score_thresh()) {
        score_thresholded_detections->emplace_back(detection);
      }
    }
    ABSL_LOG(INFO) << "palm detections surviving detection score threshold filtering:  " << score_thresholded_detections->size();

    // filter them by Non-Maximum Suppression
    auto nms_surviving_detections = FilterDetectionsByNonMaximumSuppression(*score_thresholded_detections, nms_options_, false, 0, 0);
    ABSL_LOG(INFO) << "palm detections surviving Non-Maximum Suppression filtering: " << nms_surviving_detections->size();

    return nms_surviving_detections;
  }

  /// extract the scored detections from the network output
  absl::Status ExtractValidDetections::ExtractDetections(std::vector<Detection>* output_detections, const std::vector<Tensor>& input_tensors) {

    ABSL_ASSERT(num_boxes_ > 0);

    auto raw_box_tensor = &input_tensors[ssd_decoding_tensor_mapping_.detections_tensor_index()];

    ABSL_ASSERT(raw_box_tensor->shape().dims.size() == 3);
      RET_CHECK_EQ(raw_box_tensor->shape().dims[0], 1);
      RET_CHECK_EQ(raw_box_tensor->shape().dims[1], num_boxes_);
      RET_CHECK_EQ(raw_box_tensor->shape().dims[2], num_coords_);
      auto raw_score_tensor = &input_tensors[ssd_decoding_tensor_mapping_.scores_tensor_index()];

    ABSL_ASSERT(raw_score_tensor->shape().dims.size() == 3);
      RET_CHECK_EQ(raw_score_tensor->shape().dims[0], 1);
      RET_CHECK_EQ(raw_score_tensor->shape().dims[1], num_boxes_);
      RET_CHECK_EQ(raw_score_tensor->shape().dims[2], num_classes_);

    auto raw_box_view = raw_box_tensor->GetCpuReadView();
    auto raw_boxes = raw_box_view.buffer<float>();
    auto raw_scores_view = raw_score_tensor->GetCpuReadView();
    auto raw_scores = raw_scores_view.buffer<float>();

    std::vector<float> boxes(num_boxes_ * num_coords_);
    MP_RETURN_IF_ERROR(DecodeSsdBoxes(raw_boxes, ssd_anchors_, &boxes));

    std::vector<float> detection_scores(num_boxes_);
    std::vector<int> detection_classes(num_boxes_);

    // extract each detection's score from the model output.
    // the model was trained for only one class (palm), so this loop over classes essentially reduces to triviality;
    // It will only iterate once per box, and the logic for finding the maximum score and class index is unnecessary
    // since there is only one possible class. The filtering and score selection logic are only meaningful
    // for multi-class models. For a single-class model, we can probably simplify it to directly assign
    // the score and class index without searching for the maximum
    for (int i = 0; i < num_boxes_; ++i) {
      int class_id = -1;
      float max_box_score = -std::numeric_limits<float>::max();
      // get the top score for the detection box, from among all class scores of it,
      // while as above said, we only have one class so looping is not necessary.
      for (int class_score_idx = 0; class_score_idx < num_classes_; ++class_score_idx) {
        auto class_score = raw_scores[i * num_classes_ + class_score_idx];
        if (ssd_decoding_options_.sigmoid_score()) {
          // optionally clip the score
          if (ssd_decoding_options_.has_score_clipping_thresh()) {
            class_score = class_score < -ssd_decoding_options_.score_clipping_thresh() ? -ssd_decoding_options_.score_clipping_thresh() : class_score;
            class_score = class_score > ssd_decoding_options_.score_clipping_thresh() ? ssd_decoding_options_.score_clipping_thresh() : class_score;
          }
          // the SSD palm detection model was trained to output logits, so it's only implied to to apply sigmoid to get the score per box, as in the original mediapipe inherited scoring code.
          class_score = 1.0f / (1.0f + std::exp(-class_score));
        }
        if (max_box_score < class_score) {
          max_box_score = class_score;
          class_id = class_score_idx;
        }
      }
      detection_scores[i] = max_box_score;
      detection_classes[i] = class_id;
    }

    MP_RETURN_IF_ERROR(AsDetections(boxes.data(), detection_scores.data(), detection_classes.data(), output_detections));
    return absl::OkStatus();
  }

  absl::Status ExtractValidDetections::SetDecodingParameters(float score_threshold) {
    MP_RETURN_IF_ERROR(SetSsdAnchors());
    MP_RETURN_IF_ERROR(SetSsdDecodingOptions(score_threshold));
    return absl::OkStatus();
  }

  // Configure to extract the detections from the neural network output in compliance to the detection neural network's
  // shapes, strides, scales, etc. which must be known here in order to extract the neural network's output. so these
  // just replicate the anchors which the neural network was trained with/for.
  absl::Status ExtractValidDetections::SetSsdAnchors() {

    // The SSD anchors parameters of the detection neural network
    // see https://chatgpt.com/s/t_6900bef5d9788191946d78b7ac6e27c9 regarding the sizes, and overlaps, of the trained anchors,
    // which the following variable values merely reproduce for reading the detections out from the network by this calculator.
    // the patterns of overlap are also relevant to thiking about intuition into probabilities of the input that our NMS is handling.
    SsdAnchorsCalculatorOptions ssd_anchors;
    ssd_anchors.set_num_layers(4);
    ssd_anchors.set_min_scale(0.1484375);
    ssd_anchors.set_max_scale(0.75);
    ssd_anchors.set_input_size_width(192);
    ssd_anchors.set_input_size_height(192);
    ssd_anchors.set_anchor_offset_x(0.5);
    ssd_anchors.set_anchor_offset_y(0.5);
    ssd_anchors.add_strides(8);
    ssd_anchors.add_strides(16);
    ssd_anchors.add_strides(16);
    ssd_anchors.add_strides(16);
    ssd_anchors.add_aspect_ratios(1.0);
    ssd_anchors.set_fixed_anchor_size(true);

    // Generate the anchors from these options (~2K of them)
    return SsdAnchorsCalculatorUtils::GenerateAnchors(&ssd_anchors_, ssd_anchors);
  }

  absl::Status ExtractValidDetections::SetNmsParameters() {
    nms_options_ = NonMaxSuppressionCalculatorOptions();

    // Directly set the non-maximum suppression options from the values that were previously provided as pipeline node options:
    nms_options_.set_min_suppression_threshold(0.3);
    nms_options_.set_overlap_type(NonMaxSuppressionCalculatorOptions::INTERSECTION_OVER_UNION);
    nms_options_.set_algorithm(NonMaxSuppressionCalculatorOptions::WEIGHTED);

    ABSL_CHECK_GT(nms_options_.num_detection_streams(), 0)
        << "At least one detection stream need to be specified.";
    ABSL_CHECK_NE(nms_options_.max_num_detections(), 0)
        << "max_num_detections=0 is not a valid value. Please choose a "
        << "positive number if you want to limit the number of output "
        << "detections, or set -1 if you do not want any limit.";
    return absl::OkStatus();
  }


  // Configure specific post-SSD decoding parameters and options ― hardwired for coupling to the class itself.
  // (originally these values were given as mediapipe graph calculator node "options")
  absl::Status ExtractValidDetections::SetSsdDecodingOptions(const float score_threshold) {

    ABSL_ASSERT((0.0f < score_threshold) && (score_threshold < 1.0f));

    ssd_decoding_options_ = TensorsToDetectionsCalculatorOptions();

    // the palm detection model is a single class model, so all class index filtering which our code still carries forward
    // are effectively no-ops and can be cleaned away on next sweeps.
    ssd_decoding_options_.set_num_classes(1);
    ssd_decoding_options_.set_num_boxes(2016);
    ssd_decoding_options_.set_num_coords(18);
    ssd_decoding_options_.set_box_coord_offset(0);
    ssd_decoding_options_.set_keypoint_coord_offset(4);
    ssd_decoding_options_.set_num_keypoints(7);
    ssd_decoding_options_.set_num_values_per_keypoint(2);

    ssd_decoding_options_.set_reverse_output_order(true);
    ssd_decoding_options_.set_x_scale(192.0);
    ssd_decoding_options_.set_y_scale(192.0);
    ssd_decoding_options_.set_w_scale(192.0);
    ssd_decoding_options_.set_h_scale(192.0);

    // Applies sigmoid activation to raw scores. Can be toggled at inference if model outputs logits.
    ssd_decoding_options_.set_sigmoid_score(true);

    // Clips raw scores to this threshold before sigmoid scoring.
    ssd_decoding_options_.set_score_clipping_thresh(100.0);

    // Minimum confidence score required for a detection to be considered valid.
    // Can be tuned after training to balance precision and recall.
    ssd_decoding_options_.set_min_score_thresh(score_threshold);

    RET_CHECK(ssd_decoding_options_.has_num_classes());
    RET_CHECK(ssd_decoding_options_.has_num_coords());

    num_classes_ = ssd_decoding_options_.num_classes();
    num_boxes_ = ssd_decoding_options_.num_boxes();
    num_coords_ = ssd_decoding_options_.num_coords();
    box_output_format_ = GetBoxFormat(ssd_decoding_options_);

    // Currently only support 2D when num_values_per_keypoint equals to 2.
    ABSL_CHECK_EQ(ssd_decoding_options_.num_values_per_keypoint(), 2);

    // Check if the output size is equal to the requested boxes and keypoints.
    ABSL_CHECK_EQ(ssd_decoding_options_.num_keypoints() * ssd_decoding_options_.num_values_per_keypoint() + kNumCoordsPerBox, num_coords_);

    ssd_decoding_tensor_mapping_.set_detections_tensor_index(0);
    ssd_decoding_tensor_mapping_.set_classes_tensor_index(1);
    ssd_decoding_tensor_mapping_.set_anchors_tensor_index(2);
    ssd_decoding_tensor_mapping_.set_num_detections_tensor_index(3);

    ssd_decoding_tensor_mapping_.set_scores_tensor_index(1);

    return absl::OkStatus();
  }


  absl::Status ExtractValidDetections::DecodeSsdBoxes(
      const float* raw_boxes, const std::vector<Anchor>& anchors,
      std::vector<float>* boxes) {
    for (int i = 0; i < num_boxes_; ++i) {
      const int box_offset = i * num_coords_ + ssd_decoding_options_.box_coord_offset();

      float y_center = 0.0;
      float x_center = 0.0;
      float h = 0.0;
      float w = 0.0;
      // TODO
      switch (box_output_format_) {
        case mediapipe_v01013_based::TensorsToDetectionsCalculatorOptions::UNSPECIFIED:
        case mediapipe_v01013_based::TensorsToDetectionsCalculatorOptions::YXHW:
          y_center = raw_boxes[box_offset];
          x_center = raw_boxes[box_offset + 1];
          h = raw_boxes[box_offset + 2];
          w = raw_boxes[box_offset + 3];
          break;
        case mediapipe_v01013_based::TensorsToDetectionsCalculatorOptions::XYWH:
          x_center = raw_boxes[box_offset];
          y_center = raw_boxes[box_offset + 1];
          w = raw_boxes[box_offset + 2];
          h = raw_boxes[box_offset + 3];
          break;
        case mediapipe_v01013_based::TensorsToDetectionsCalculatorOptions::XYXY:
          x_center = (-raw_boxes[box_offset] + raw_boxes[box_offset + 2]) / 2;
          y_center = (-raw_boxes[box_offset + 1] + raw_boxes[box_offset + 3]) / 2;
          w = raw_boxes[box_offset + 2] + raw_boxes[box_offset];
          h = raw_boxes[box_offset + 3] + raw_boxes[box_offset + 1];
          break;
      }
      x_center =
          x_center / ssd_decoding_options_.x_scale() * anchors[i].w() + anchors[i].x_center();
      y_center =
          y_center / ssd_decoding_options_.y_scale() * anchors[i].h() + anchors[i].y_center();

      if (ssd_decoding_options_.apply_exponential_on_box_size()) {
        h = std::exp(h / ssd_decoding_options_.h_scale()) * anchors[i].h();
        w = std::exp(w / ssd_decoding_options_.w_scale()) * anchors[i].w();
      } else {
        h = h / ssd_decoding_options_.h_scale() * anchors[i].h();
        w = w / ssd_decoding_options_.w_scale() * anchors[i].w();
      }

      const float ymin = y_center - h / 2.f;
      const float xmin = x_center - w / 2.f;
      const float ymax = y_center + h / 2.f;
      const float xmax = x_center + w / 2.f;

      (*boxes)[i * num_coords_ + 0] = ymin;
      (*boxes)[i * num_coords_ + 1] = xmin;
      (*boxes)[i * num_coords_ + 2] = ymax;
      (*boxes)[i * num_coords_ + 3] = xmax;

      if (ssd_decoding_options_.num_keypoints()) {
        for (int k = 0; k < ssd_decoding_options_.num_keypoints(); ++k) {
          const int offset = i * num_coords_ + ssd_decoding_options_.keypoint_coord_offset() +
                            k * ssd_decoding_options_.num_values_per_keypoint();

          float keypoint_y = 0.0;
          float keypoint_x = 0.0;
          switch (box_output_format_) {
            case mediapipe_v01013_based::TensorsToDetectionsCalculatorOptions::UNSPECIFIED:
            case mediapipe_v01013_based::TensorsToDetectionsCalculatorOptions::YXHW:
              keypoint_y = raw_boxes[offset];
              keypoint_x = raw_boxes[offset + 1];
              break;
            case mediapipe_v01013_based::TensorsToDetectionsCalculatorOptions::XYWH:
            case mediapipe_v01013_based::TensorsToDetectionsCalculatorOptions::XYXY:
              keypoint_x = raw_boxes[offset];
              keypoint_y = raw_boxes[offset + 1];
              break;
          }

          (*boxes)[offset] = keypoint_x / ssd_decoding_options_.x_scale() * anchors[i].w() +
                            anchors[i].x_center();
          (*boxes)[offset + 1] =
              keypoint_y / ssd_decoding_options_.y_scale() * anchors[i].h() +
              anchors[i].y_center();
        }
      }
    }

    return absl::OkStatus();
  }

  // extract all but invalid detections from the inference output
  // for modularity, filtering out invalid ones should be moved
  // to take place before reaching this function.
  absl::Status ExtractValidDetections::AsDetections(
      const float* detection_boxes, const float* detection_scores,
      const int* detection_classes, std::vector<Detection>* output_detections) {

    for (int i = 0; i < num_boxes_ * classes_per_detection_; i += classes_per_detection_) {
      const int box_offset = i * num_coords_;

      // extract box geometry
      Detection detection = AsDetection(
          /*box_ymin=*/detection_boxes[box_offset + box_indices_[0]],
          /*box_xmin=*/detection_boxes[box_offset + box_indices_[1]],
          /*box_ymax=*/detection_boxes[box_offset + box_indices_[2]],
          /*box_xmax=*/detection_boxes[box_offset + box_indices_[3]],
          absl::MakeConstSpan(detection_scores + i, classes_per_detection_),
          absl::MakeConstSpan(detection_classes + i, classes_per_detection_));
      if (detection.score().empty()) { continue; }

      // filter out box predictions which are possible as neural network output but should be ignored as invalid ―
      // e.g. decoded detection boxes can have negative values for width/height from model prediction.
      // we filter out those boxes, as well as width/height assigned nan values by the model.
      // this function should be made more modular: the current function should only convert to a Detection
      // object and filtering out invalid box outputs should be somewhere outside and before reaching it.
      const auto& bbox = detection.location_data().relative_bounding_box();
      if (bbox.width() < 0 || bbox.height() < 0 || std::isnan(bbox.width()) || std::isnan(bbox.height())) { continue; }

      // extract the associated keypoints
      if (ssd_decoding_options_.num_keypoints() > 0) {
        auto* location_data = detection.mutable_location_data();
        for (int kp_id = 0; kp_id < ssd_decoding_options_.num_keypoints() *
             ssd_decoding_options_.num_values_per_keypoint();
             kp_id += ssd_decoding_options_.num_values_per_keypoint()) {
          auto keypoint = location_data->add_relative_keypoints();
          const int keypoint_index = box_offset + ssd_decoding_options_.keypoint_coord_offset() + kp_id;
          keypoint->set_x(detection_boxes[keypoint_index + 0]);
          keypoint->set_y(ssd_decoding_options_.flip_vertically() ? 1.f - detection_boxes[keypoint_index + 1] : detection_boxes[keypoint_index + 1]);
        }
      }
      output_detections->emplace_back(detection);
    }
    return absl::OkStatus();
  }

  // converts to mediapipe detection object, while also filtering out by the set score threshold.
  // (really bad coupling by the original mediapipe code, these should not optimally be in the same fn).
  // the class filtering is vacuous in our case as it is a single class SSD model we consume from.
  Detection ExtractValidDetections::AsDetection(
      float box_ymin, float box_xmin, float box_ymax, float box_xmax,
      absl::Span<const float> scores, absl::Span<const int> class_ids) {

    Detection detection;

    LocationData* location_data = detection.mutable_location_data();
    location_data->set_format(LocationData::RELATIVE_BOUNDING_BOX);

    LocationData::RelativeBoundingBox* relative_bbox =
        location_data->mutable_relative_bounding_box();

    relative_bbox->set_xmin(box_xmin);
    relative_bbox->set_ymin(box_ymin);
    relative_bbox->set_width(box_xmax - box_xmin);
    relative_bbox->set_height(box_ymax - box_ymin);

    // in our case, only one class (and thus one score) per detection, so a loop is not really necessary.
    for (int i = 0; i < scores.size(); ++i) {
      detection.add_score(scores[i]);
      detection.add_label_id(class_ids[i]);
    }

    return detection;
  }

}  // namespace api2
}  // namespace mediapipe_v01013_based
