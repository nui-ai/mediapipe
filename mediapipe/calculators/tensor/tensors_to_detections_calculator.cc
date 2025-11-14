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
#include "mediapipe/calculators/tensor/detections_extraction.h"
using hand_tracking_mp_lean::api2::DetectionsExtractionAndFiltering;

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

namespace hand_tracking_mp_lean {
namespace api2 {
  using BoxFormat = ::hand_tracking_mp_lean::TensorsToDetectionsCalculatorOptions::BoxFormat;

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
  class ConvertDetectionTensorsCalculator : public Node {
  public:
    static constexpr Input<std::vector<Tensor>> kInTensors{"TENSORS"};
    static constexpr SideInput<std::vector<Anchor>>::Optional kInAnchors{
      "ANCHORS"};
    static constexpr SideInput<std::vector<int>>::Optional kSideInIgnoreClasses{
      "IGNORE_CLASSES"};
    static constexpr Output<std::vector<Detection>> kOutDetections{"DETECTIONS"};
    MEDIAPIPE_NODE_CONTRACT(kInTensors, kInAnchors, kSideInIgnoreClasses,
                            kOutDetections);
    static absl::Status UpdateContract(CalculatorContract* cc);

    absl::Status Open(CalculatorContext* cc) override;
    absl::Status Process(CalculatorContext* cc) override;
    absl::Status Close(CalculatorContext* cc) override;

  private:

    int num_classes_ = 0;
    int num_boxes_ = 0;
    int num_coords_ = 0;
    int max_results_ = -1;
    int classes_per_detection_ = 1;
    BoxFormat box_output_format_ = hand_tracking_mp_lean::TensorsToDetectionsCalculatorOptions::YXHW;

    bool initialized_ = false;

    // Set of allowed or ignored class indices.
    struct ClassIndexSet {
      absl::flat_hash_set<int> values;
      bool is_allowlist;
    };
    // Allowed or ignored class indices based on provided options or side packet.
    // These are used to filter out the output detection results.
    ClassIndexSet class_index_set_;

    bool has_custom_box_indices_ = false;
    bool scores_tensor_index_is_set_ = false;
    std::vector<int> box_indices_ = {0, 1, 2, 3};

    // several decoding parameters, 98% of which just mirror how the detection neural network (which is a SSD neural network)
    // which this class decodes from ― was trained ― 98% of which cannot therefore be modified.
    // we just group setting their values a little by their sub-topic, much mirroring the original mediapipe pipeline's
    // divide of setting these values up for decoding. note that although mediapipe was supposed to be modular with regard
    // to underlying neural network execution stacks, the decoding code (much like other parts of mediapipe pipelines)
    // is marred with conditional logic about the execution stack (cpu v.s. gpu in the current case) which it had
    // set out to abstract away for pipelines but ended up entangling pipeline level code to handle totally
    // explicitly much of the time.
    TensorsToDetectionsCalculatorOptions::TensorMapping ssd_decoding_tensor_mapping_;
    TensorsToDetectionsCalculatorOptions ssd_decoding_options_;
    std::vector<Anchor> ssd_anchors_;
    NonMaxSuppressionCalculatorOptions nms_options_;

#ifndef MEDIAPIPE_DISABLE_GL_COMPUTE
    hand_tracking_mp_lean::GlCalculatorHelper gpu_helper_;
    GLuint decode_program_;
    GLuint score_program_;
#elif MEDIAPIPE_METAL_ENABLED
    MPPMetalHelper* gpu_helper_ = nullptr;
    id<MTLComputePipelineState> decode_program_;
    id<MTLComputePipelineState> score_program_;
#endif  // !defined(MEDIAPIPE_DISABLE_GL_COMPUTE)
    std::unique_ptr<Tensor> raw_anchors_buffer_;
    std::unique_ptr<Tensor> decoded_boxes_buffer_;
    std::unique_ptr<Tensor> scored_boxes_buffer_;
    std::unique_ptr<DetectionsExtractionAndFiltering> core_;

    bool gpu_inited_ = false;
    bool gpu_input_ = false;
    bool gpu_has_enough_work_groups_ = true;
    bool anchors_init_ = false;
  };
  MEDIAPIPE_REGISTER_NODE(ConvertDetectionTensorsCalculator);

  absl::Status ConvertDetectionTensorsCalculator::UpdateContract(
      CalculatorContract* cc) {
    if (CanUseGpu()) {
#ifndef MEDIAPIPE_DISABLE_GL_COMPUTE
      MP_RETURN_IF_ERROR(hand_tracking_mp_lean::GlCalculatorHelper::UpdateContract(
          cc, /*request_gpu_as_optional=*/true));
#elif MEDIAPIPE_METAL_ENABLED
      MP_RETURN_IF_ERROR([MPPMetalHelper updateContract:cc]);
#endif  // !defined(MEDIAPIPE_DISABLE_GL_COMPUTE)
    }

    return absl::OkStatus();
  }

  absl::Status ConvertDetectionTensorsCalculator::Open(CalculatorContext* cc) {
    initialized_ = true;

    if (CanUseGpu()) {
#ifndef MEDIAPIPE_DISABLE_GL_COMPUTE
#elif MEDIAPIPE_METAL_ENABLED
      gpu_helper_ = [[MPPMetalHelper alloc] initWithCalculatorContext:cc];
      RET_CHECK(gpu_helper_);
#endif  // !defined(MEDIAPIPE_DISABLE_GL_COMPUTE)
    }
    // Instantiate and open core
    core_ = std::make_unique<DetectionsExtractionAndFiltering>(0.5);
    return absl::OkStatus();
  }

  // 1. this method decodes the raw palm detections neural network output tensors into detections;
  //    a few of the parameters it uses may also over-filter its raw output detections, those parameters are
  //    inline documented as such where we set them in the parameter setting methods of this class!
  //    so basically this class performs the pedantic work of extracting the raw SSD neural network outputs,
  //    with only some optional leverage for filtering the raw detections.
  // 2. it also transforms them into mediapipe vector types.
  absl::Status ConvertDetectionTensorsCalculator::Process(CalculatorContext* cc) {
    assert (initialized_);

    auto output_detections = absl::make_unique<std::vector<Detection>>();
    bool gpu_processing = false;
    const auto& input_tensors = *kInTensors(cc);
    for (const auto& tensor : input_tensors) {
      RET_CHECK(tensor.element_type() == Tensor::ElementType::kFloat32);
    }

    auto filtered_detections = core_->Extract(input_tensors);
    MP_RETURN_IF_ERROR(filtered_detections.status());
    kOutDetections(cc).Send(*filtered_detections.value());

    // kOutDetections(cc).Send(std::move(nms_surviving_detections));
    return absl::OkStatus();
  }

  absl::Status ConvertDetectionTensorsCalculator::Close(CalculatorContext* cc) {
#ifndef MEDIAPIPE_DISABLE_GL_COMPUTE
    if (gpu_inited_) {
      gpu_helper_.RunInGlContext([this] {
        decoded_boxes_buffer_ = nullptr;
        scored_boxes_buffer_ = nullptr;
        raw_anchors_buffer_ = nullptr;
        glDeleteProgram(decode_program_);
        glDeleteProgram(score_program_);
      });
    }
#elif MEDIAPIPE_METAL_ENABLED
    decoded_boxes_buffer_ = nullptr;
    scored_boxes_buffer_ = nullptr;
    raw_anchors_buffer_ = nullptr;
    decode_program_ = nil;
    score_program_ = nil;
#endif  // !defined(MEDIAPIPE_DISABLE_GL_COMPUTE)

    return absl::OkStatus();
  }

}  // namespace api2
}  // namespace hand_tracking_mp_lean
