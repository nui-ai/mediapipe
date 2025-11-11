#ifndef MEDIAPIPE_CALCULATORS_TENSOR_TENSORS_TO_DETECTIONS_CALCULATOR_CORE_H_
#define MEDIAPIPE_CALCULATORS_TENSOR_TENSORS_TO_DETECTIONS_CALCULATOR_CORE_H_

#include <memory>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "mediapipe/calculators/tensor/tensors_to_detections_calculator.pb.h"
#include "mediapipe/calculators/util/non_max_suppression_calculator.pb.h"
#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/formats/object_detection/anchor.pb.h"
#include "mediapipe/framework/formats/tensor.h"

namespace mediapipe_v01013_based {
namespace api2 {

using BoxFormat = ::mediapipe_v01013_based::TensorsToDetectionsCalculatorOptions::BoxFormat;
using Anchor = ::mediapipe_v01013_based::Anchor;

class DetectionsExtractionAndFiltering {
public:
  explicit DetectionsExtractionAndFiltering(float score_threshold);
  absl::StatusOr<std::unique_ptr<std::vector<Detection>>> Extract(const std::vector<Tensor>& input_tensors);
  absl::StatusOr<std::unique_ptr<std::vector<Detection>>> Filter(const std::vector<Detection>& detections);

private:
  absl::Status ExtractDo(std::vector<Detection>* output_detections, const std::vector<Tensor>& input_tensors);
  absl::Status SetDecodingParameters(float score_threshold);
  absl::Status SetSsdAnchors();
  absl::Status SetSsdDecodingOptions(float score_threshold);
  absl::Status SetNmsParameters();
  absl::Status DecodeSsdBoxes(const float* raw_boxes,
                           const std::vector<Anchor>& anchors,
                           std::vector<float>* boxes);
  absl::Status AsDetections(const float* detection_boxes,
                                   const float* detection_scores,
                                   const int* detection_classes,
                                   std::vector<Detection>* output_detections);
  Detection AsDetection(float box_ymin, float box_xmin, float box_ymax,
                               float box_xmax, absl::Span<const float> scores,
                               absl::Span<const int> class_ids);

  int num_classes_ = 0;
  int num_boxes_ = 0;
  int num_coords_ = 0;
  int classes_per_detection_= 1;
  BoxFormat box_output_format_ = mediapipe_v01013_based::TensorsToDetectionsCalculatorOptions::YXHW;

  struct ClassIndexSet {
    absl::flat_hash_set<int> values;
    bool is_allowlist;
  };

  bool has_custom_box_indices_;
  bool scores_tensor_index_is_set_;
  std::vector<int> box_indices_ = {0, 1, 2, 3};

  bool initialized_ = false;

  TensorsToDetectionsCalculatorOptions::TensorMapping ssd_decoding_tensor_mapping_;
  TensorsToDetectionsCalculatorOptions ssd_decoding_options_;
  std::vector<Anchor> ssd_anchors_;
  NonMaxSuppressionCalculatorOptions nms_options_;

#ifndef MEDIAPIPE_DISABLE_GL_COMPUTE
  mediapipe_v01013_based::GlCalculatorHelper gpu_helper_;
  GLuint decode_program_;
  GLuint score_program_;
#elif MEDIAPIPE_METAL_ENABLED
  MPPMetalHelper* gpu_helper_;
  id<MTLComputePipelineState> decode_program_;
  id<MTLComputePipelineState> score_program_;
#endif
  std::unique_ptr<Tensor> raw_anchors_buffer_;
  std::unique_ptr<Tensor> decoded_boxes_buffer_;
  std::unique_ptr<Tensor> scored_boxes_buffer_;

  bool gpu_inited_;
  bool gpu_input_;
  bool gpu_has_enough_work_groups_;
  bool anchors_init_;
};

}  // namespace api2
}  // namespace mediapipe_v01013_based

#endif  // MEDIAPIPE_CALCULATORS_TENSOR_TENSORS_TO_DETECTIONS_CALCULATOR_CORE_H_
