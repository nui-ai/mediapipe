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
#include "mediapipe/calculators/tensor/tensors_to_detections_calculator_core.h"
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
constexpr int kNumInputTensorsWithAnchors = 3;
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

namespace mediapipe {
namespace api2 {
  using BoxFormat = ::mediapipe::TensorsToDetectionsCalculatorOptions::BoxFormat;

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
        return mediapipe::TensorsToDetectionsCalculatorOptions::XYWH;
      }
      return mediapipe::TensorsToDetectionsCalculatorOptions::YXHW;
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
  absl::Status ConvertDetectionTensors::Open() {
    MP_RETURN_IF_ERROR(SetDecodingParameters());
    MP_RETURN_IF_ERROR(SetNmsParameters());
    initialized_ = true;

    if (CanUseGpu()) {
#ifndef MEDIAPIPE_DISABLE_GL_COMPUTE
#elif MEDIAPIPE_METAL_ENABLED
      gpu_helper_ = [[MPPMetalHelper alloc] initWithCalculatorContext:cc];
      RET_CHECK(gpu_helper_);
#endif  // !defined(MEDIAPIPE_DISABLE_GL_COMPUTE)
    }

    return absl::OkStatus();
  }

  // 1. this method decodes the raw palm detections neural network output tensors into detections;
  //    a few of the parameters it uses may also over-filter its raw output detections, those parameters are
  //    inline documented as such where we set them in the parameter setting methods of this class!
  //    so basically this class performs the pedantic work of extracting the raw SSD neural network outputs,
  //    with only some optional leverage for filtering the raw detections.
  // 2. it also transforms them into mediapipe vector types.
  absl::StatusOr<std::unique_ptr<std::vector<Detection>>> ConvertDetectionTensors::Process(const std::vector<Tensor>& input_tensors) {
    assert (initialized_);


    auto filtered_detections = absl::make_unique<std::vector<Detection>>();
    for (const auto& tensor : input_tensors) {
      RET_CHECK(tensor.element_type() == Tensor::ElementType::kFloat32);
    }
    const int num_input_tensors = input_tensors.size();
    if (!scores_tensor_index_is_set_) {
      if (num_input_tensors == 2 ||
          num_input_tensors == kNumInputTensorsWithAnchors) {
        ssd_decoding_tensor_mapping_.set_scores_tensor_index(1);
          } else {
            ssd_decoding_tensor_mapping_.set_scores_tensor_index(2);
          }
      scores_tensor_index_is_set_ = true;
    }

    MP_RETURN_IF_ERROR(ProcessCPU(filtered_detections.get(), input_tensors));
    ABSL_LOG(INFO) << "TensorsToDetectionsCore::Process: " << " filtered detections:      " << filtered_detections->size();

    auto nms_surviving_detections = FilterDetectionsByNonMaximumSuppression(*filtered_detections, nms_options_, false, 0, 0);
    ABSL_LOG(INFO) << "TensorsToDetectionsCore::Process: " << " nms surviving detections: " << nms_surviving_detections->size();

    return nms_surviving_detections;
  }

  absl::Status ConvertDetectionTensors::ProcessCPU(std::vector<Detection>* output_detections, const std::vector<Tensor>& input_tensors) {

    if (input_tensors.size() == 2 ||
        input_tensors.size() == kNumInputTensorsWithAnchors) {
      // Postprocessing on CPU for model without postprocessing op. E.g. output
      // raw score tensor and box tensor. Anchor decoding will be handled below.
      auto raw_box_tensor =
          &input_tensors[ssd_decoding_tensor_mapping_.detections_tensor_index()];
      RET_CHECK_GT(num_boxes_, 0) << "Please set num_boxes in calculator options";
      if (raw_box_tensor->shape().dims.size() == 3) {
        // The tensors from CPU inference has dim 3.
        RET_CHECK_EQ(raw_box_tensor->shape().dims[0], 1);
        RET_CHECK_EQ(raw_box_tensor->shape().dims[1], num_boxes_);
        RET_CHECK_EQ(raw_box_tensor->shape().dims[2], num_coords_);
      } else if (raw_box_tensor->shape().dims.size() == 4) {
        // The tensors from GPU inference has dim 4. For gpu-cpu fallback support,
        // we allow tensors with 4 dims.
        RET_CHECK_EQ(raw_box_tensor->shape().dims[0], 1);
        RET_CHECK_EQ(raw_box_tensor->shape().dims[1], 1);
        RET_CHECK_EQ(raw_box_tensor->shape().dims[2], num_boxes_);
        RET_CHECK_EQ(raw_box_tensor->shape().dims[3], num_coords_);
      } else {
        return absl::InvalidArgumentError(
            "The dimensions of box Tensor must be 3 or 4.");
      }
      auto raw_score_tensor =
          &input_tensors[ssd_decoding_tensor_mapping_.scores_tensor_index()];
      if (raw_score_tensor->shape().dims.size() == 3) {
        // The tensors from CPU inference has dim 3.
        RET_CHECK_EQ(raw_score_tensor->shape().dims[0], 1);
        RET_CHECK_EQ(raw_score_tensor->shape().dims[1], num_boxes_);
        RET_CHECK_EQ(raw_score_tensor->shape().dims[2], num_classes_);
      } else if (raw_score_tensor->shape().dims.size() == 4) {
        // The tensors from GPU inference has dim 4. For gpu-cpu fallback support,
        // we allow tensors with 4 dims.
        RET_CHECK_EQ(raw_score_tensor->shape().dims[0], 1);
        RET_CHECK_EQ(raw_score_tensor->shape().dims[1], 1);
        RET_CHECK_EQ(raw_score_tensor->shape().dims[2], num_boxes_);
        RET_CHECK_EQ(raw_score_tensor->shape().dims[3], num_classes_);
      } else {
        return absl::InvalidArgumentError(
            "The dimensions of score Tensor must be 3 or 4.");
      }
      auto raw_box_view = raw_box_tensor->GetCpuReadView();
      auto raw_boxes = raw_box_view.buffer<float>();
      auto raw_scores_view = raw_score_tensor->GetCpuReadView();
      auto raw_scores = raw_scores_view.buffer<float>();

      std::vector<float> boxes(num_boxes_ * num_coords_);
      MP_RETURN_IF_ERROR(DecodeBoxes(raw_boxes, ssd_anchors_, &boxes));

      std::vector<float> detection_scores(num_boxes_);
      std::vector<int> detection_classes(num_boxes_);

      // score each box's detected class instances by scores ― note that:
      // the model was trained for only one class (palm), so this loop essentially becomes trivial.
      // It will only iterate once per box, and the logic for finding the maximum score and class index
      // is unnecessary since there is only one possible class. The filtering and score selection logic
      // are only meaningful for multi-class models. For a single-class model, we can most probably simplify
      // this code to directly assign the score and class index without searching for the maximum
      for (int i = 0; i < num_boxes_; ++i) {
        int class_id = -1;
        float max_score = -std::numeric_limits<float>::max();
        // Find the top score for box i, but as said we only have one class so will only score one instance for each box.
        for (int score_idx = 0; score_idx < num_classes_; ++score_idx) {
          auto score = raw_scores[i * num_classes_ + score_idx];
          if (ssd_decoding_options_.sigmoid_score()) {
            if (ssd_decoding_options_.has_score_clipping_thresh()) {
              score = score < -ssd_decoding_options_.score_clipping_thresh()
                          ? -ssd_decoding_options_.score_clipping_thresh()
                          : score;
              score = score > ssd_decoding_options_.score_clipping_thresh()
                          ? ssd_decoding_options_.score_clipping_thresh()
                          : score;
            }

            // the SSD palm detection model was trained to output logits,
            // so it's only implied to to apply sigmoid to get the score per box.
            // and this line is original mediapipe inherited scoring code.
            score = 1.0f / (1.0f + std::exp(-score));

          }
          if (max_score < score) {
            max_score = score;
            class_id = score_idx;
          }
        }
        detection_scores[i] = max_score;
        detection_classes[i] = class_id;
      }

      MP_RETURN_IF_ERROR(
          ConvertToDetections(boxes.data(), detection_scores.data(),
                              detection_classes.data(), output_detections));
        } else {
          // Postprocessing on CPU with postprocessing op (e.g. anchor decoding and
          // non-maximum suppression) within the model.
          RET_CHECK_EQ(input_tensors.size(), 4);
          auto num_boxes_tensor =
              &input_tensors[ssd_decoding_tensor_mapping_.num_detections_tensor_index()];
          RET_CHECK_EQ(num_boxes_tensor->shape().dims.size(), 1);
          RET_CHECK_EQ(num_boxes_tensor->shape().dims[0], 1);

          auto detection_boxes_tensor =
              &input_tensors[ssd_decoding_tensor_mapping_.detections_tensor_index()];
          RET_CHECK_EQ(detection_boxes_tensor->shape().dims.size(), 3);
          RET_CHECK_EQ(detection_boxes_tensor->shape().dims[0], 1);
          const int max_detections = detection_boxes_tensor->shape().dims[1];
          RET_CHECK_EQ(detection_boxes_tensor->shape().dims[2], num_coords_);

          auto detection_classes_tensor =
              &input_tensors[ssd_decoding_tensor_mapping_.classes_tensor_index()];
          RET_CHECK_EQ(detection_classes_tensor->shape().dims.size(), 2);
          RET_CHECK_EQ(detection_classes_tensor->shape().dims[0], 1);
          RET_CHECK_EQ(detection_classes_tensor->shape().dims[1], max_detections);

          auto detection_scores_tensor =
              &input_tensors[ssd_decoding_tensor_mapping_.scores_tensor_index()];
          RET_CHECK_EQ(detection_scores_tensor->shape().dims.size(), 2);
          RET_CHECK_EQ(detection_scores_tensor->shape().dims[0], 1);
          RET_CHECK_EQ(detection_scores_tensor->shape().dims[1], max_detections);

          auto num_boxes_view = num_boxes_tensor->GetCpuReadView();
          auto num_boxes = num_boxes_view.buffer<float>();
          num_boxes_ = num_boxes[0];
          // The detection model with Detection_PostProcess op may output duplicate
          // boxes with different classes, in the following format:
          //   num_boxes_tensor = [num_boxes]
          //   detection_classes_tensor = [box_1_class_1, box_1_class_2, ...]
          //   detection_scores_tensor = [box_1_score_1, box_1_score_2, ... ]
          //   detection_boxes_tensor = [box_1, box1, ... ]
          // Each box repeats classes_per_detection_ times.
          // Note Detection_PostProcess op is only supported in CPU.
          classes_per_detection_ = ssd_decoding_options_.max_classes_per_detection();

          auto detection_boxes_view = detection_boxes_tensor->GetCpuReadView();
          auto detection_boxes = detection_boxes_view.buffer<float>();

          auto detection_scores_view = detection_scores_tensor->GetCpuReadView();
          auto detection_scores = detection_scores_view.buffer<float>();

          auto detection_classes_view = detection_classes_tensor->GetCpuReadView();
          auto detection_classes_ptr = detection_classes_view.buffer<float>();
          std::vector<int> detection_classes(num_boxes_ * classes_per_detection_);
          for (int i = 0; i < detection_classes.size(); ++i) {
            detection_classes[i] = static_cast<int>(detection_classes_ptr[i]);
          }
          MP_RETURN_IF_ERROR(ConvertToDetections(detection_boxes, detection_scores,
                                                 detection_classes.data(),
                                                 output_detections));
        }
    return absl::OkStatus();
  }

  absl::Status ConvertDetectionTensors::SetDecodingParameters() {
    MP_RETURN_IF_ERROR(SetSsdAnchors());
    MP_RETURN_IF_ERROR(SetSsdDecodingOptions());
    return absl::OkStatus();
  }

  // Configure to extract the detections from the neural network output in compliance to the detection neural network's
  // shapes, strides, scales, etc. which must be known here in order to extract the neural network's output. so these
  // just replicate the anchors which the neural network was trained with/for.
  absl::Status ConvertDetectionTensors::SetSsdAnchors() {

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

  absl::Status ConvertDetectionTensors::SetNmsParameters() {
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
  absl::Status ConvertDetectionTensors::SetSsdDecodingOptions() {

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
    ssd_decoding_options_.set_min_score_thresh(0.5);

    RET_CHECK(ssd_decoding_options_.has_num_classes());
    RET_CHECK(ssd_decoding_options_.has_num_coords());

    num_classes_ = ssd_decoding_options_.num_classes();
    num_boxes_ = ssd_decoding_options_.num_boxes();
    num_coords_ = ssd_decoding_options_.num_coords();
    box_output_format_ = GetBoxFormat(ssd_decoding_options_);
    ABSL_CHECK_NE(ssd_decoding_options_.max_results(), 0)
        << "The maximum number of the top-scored detection results must be "
           "non-zero.";
    max_results_ = ssd_decoding_options_.max_results();

    // Currently only support 2D when num_values_per_keypoint equals to 2.
    ABSL_CHECK_EQ(ssd_decoding_options_.num_values_per_keypoint(), 2);

    // Check if the output size is equal to the requested boxes and keypoints.
    ABSL_CHECK_EQ(ssd_decoding_options_.num_keypoints() * ssd_decoding_options_.num_values_per_keypoint() + kNumCoordsPerBox, num_coords_);

    // Configure tensor mappings
    if (ssd_decoding_options_.has_tensor_mapping()) {
      RET_CHECK_OK(CheckCustomTensorMapping(ssd_decoding_options_.tensor_mapping()));
      ssd_decoding_tensor_mapping_ = ssd_decoding_options_.tensor_mapping();
      scores_tensor_index_is_set_ = true;
    } else {
      // Assigns the default tensor indices.
      ssd_decoding_tensor_mapping_.set_detections_tensor_index(0);
      ssd_decoding_tensor_mapping_.set_classes_tensor_index(1);
      ssd_decoding_tensor_mapping_.set_anchors_tensor_index(2);
      ssd_decoding_tensor_mapping_.set_num_detections_tensor_index(3);
      // The scores tensor index needs to be determined based on the number of
      // model's output tensors, which will be available in the first invocation
      // of the Process() method.
      ssd_decoding_tensor_mapping_.set_scores_tensor_index(-1);
      scores_tensor_index_is_set_ = false;
    }

    if (ssd_decoding_options_.has_box_boundaries_indices()) {
      box_indices_ = {ssd_decoding_options_.box_boundaries_indices().ymin(),
                      ssd_decoding_options_.box_boundaries_indices().xmin(),
                      ssd_decoding_options_.box_boundaries_indices().ymax(),
                      ssd_decoding_options_.box_boundaries_indices().xmax()};
      int bitmap = 0;
      for (int i : box_indices_) {
        bitmap |= 1 << i;
      }
      RET_CHECK_EQ(bitmap, 15) << "The custom box boundaries indices should only "
                                  "cover index 0, 1, 2, and 3.";
      has_custom_box_indices_ = true;
    }

    return absl::OkStatus();
  }


  absl::Status ConvertDetectionTensors::DecodeBoxes(
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
        case mediapipe::TensorsToDetectionsCalculatorOptions::UNSPECIFIED:
        case mediapipe::TensorsToDetectionsCalculatorOptions::YXHW:
          y_center = raw_boxes[box_offset];
          x_center = raw_boxes[box_offset + 1];
          h = raw_boxes[box_offset + 2];
          w = raw_boxes[box_offset + 3];
          break;
        case mediapipe::TensorsToDetectionsCalculatorOptions::XYWH:
          x_center = raw_boxes[box_offset];
          y_center = raw_boxes[box_offset + 1];
          w = raw_boxes[box_offset + 2];
          h = raw_boxes[box_offset + 3];
          break;
        case mediapipe::TensorsToDetectionsCalculatorOptions::XYXY:
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
            case mediapipe::TensorsToDetectionsCalculatorOptions::UNSPECIFIED:
            case mediapipe::TensorsToDetectionsCalculatorOptions::YXHW:
              keypoint_y = raw_boxes[offset];
              keypoint_x = raw_boxes[offset + 1];
              break;
            case mediapipe::TensorsToDetectionsCalculatorOptions::XYWH:
            case mediapipe::TensorsToDetectionsCalculatorOptions::XYXY:
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

  // converts to mediapipe detection objects, while also:
  // 1. filtering to only pass forward the maximum requested number of detections
  // 2. filtering out by the set score threshold (within the ConvertToDetection() call)
  // 2. optionally vertically flipping the detection coordinates
  absl::Status ConvertDetectionTensors::ConvertToDetections(
      const float* detection_boxes, const float* detection_scores,
      const int* detection_classes, std::vector<Detection>* output_detections) {

    for (int i = 0; i < num_boxes_ * classes_per_detection_; i += classes_per_detection_) {
      if (max_results_ > 0 && output_detections->size() == max_results_) {
        break;
      }
      const int box_offset = i * num_coords_;
      Detection detection = ConvertToDetection(
          /*box_ymin=*/detection_boxes[box_offset + box_indices_[0]],
          /*box_xmin=*/detection_boxes[box_offset + box_indices_[1]],
          /*box_ymax=*/detection_boxes[box_offset + box_indices_[2]],
          /*box_xmax=*/detection_boxes[box_offset + box_indices_[3]],
          absl::MakeConstSpan(detection_scores + i, classes_per_detection_),
          absl::MakeConstSpan(detection_classes + i, classes_per_detection_),
          ssd_decoding_options_.flip_vertically());
      // if all the scores and classes are filtered out, we skip the empty detection.
      if (detection.score().empty()) {
        continue;
      }

      // filter out box predictions which are possible as neural network output but should be ignored as invalid:
      // Decoded detection boxes could have negative values for width/height due
      // to model prediction. Filter out those boxes since some downstream
      // calculators may assume non-negative values. (b/171391719)
      const auto& bbox = detection.location_data().relative_bounding_box();
      if (bbox.width() < 0 || bbox.height() < 0 || std::isnan(bbox.width()) ||
          std::isnan(bbox.height())) {
        continue;
      }

      // Add keypoints.
      if (ssd_decoding_options_.num_keypoints() > 0) {
        auto* location_data = detection.mutable_location_data();
        for (int kp_id = 0; kp_id < ssd_decoding_options_.num_keypoints() *
             ssd_decoding_options_.num_values_per_keypoint();
             kp_id += ssd_decoding_options_.num_values_per_keypoint()) {
          auto keypoint = location_data->add_relative_keypoints();
          const int keypoint_index = box_offset + ssd_decoding_options_.keypoint_coord_offset() + kp_id;
          keypoint->set_x(detection_boxes[keypoint_index + 0]);
          keypoint->set_y(ssd_decoding_options_.flip_vertically()
                              ? 1.f - detection_boxes[keypoint_index + 1]
                              : detection_boxes[keypoint_index + 1]);
        }
      }
      output_detections->emplace_back(detection);
    }
    return absl::OkStatus();
  }

  // converts to mediapipe detection object, while also filtering out by the set score threshold.
  // (really bad coupling by the original mediapipe code, these should not optimally be in the same fn).
  // the class filtering is vacuous in our case as it is a single class SSD model we consume from.
  Detection ConvertDetectionTensors::ConvertToDetection(
      float box_ymin, float box_xmin, float box_ymax, float box_xmax,
      absl::Span<const float> scores, absl::Span<const int> class_ids,
      bool flip_vertically) {
    Detection detection;
    for (int i = 0; i < scores.size(); ++i) {
      if (ssd_decoding_options_.has_min_score_thresh() &&
          scores[i] < ssd_decoding_options_.min_score_thresh()) {
        continue;
      }
      detection.add_score(scores[i]);
      detection.add_label_id(class_ids[i]);
    }

    LocationData* location_data = detection.mutable_location_data();
    location_data->set_format(LocationData::RELATIVE_BOUNDING_BOX);

    LocationData::RelativeBoundingBox* relative_bbox =
        location_data->mutable_relative_bounding_box();

    relative_bbox->set_xmin(box_xmin);
    relative_bbox->set_ymin(flip_vertically ? 1.f - box_ymax : box_ymin);
    relative_bbox->set_width(box_xmax - box_xmin);
    relative_bbox->set_height(box_ymax - box_ymin);
    return detection;
  }

}  // namespace api2
}  // namespace mediapipe
