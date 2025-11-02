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

#ifndef MEDIAPIPE_CALCULATORS_TENSOR_TENSORS_TO_CLASSIFICATION_CALCULATOR_CORE_H_
#define MEDIAPIPE_CALCULATORS_TENSOR_TENSORS_TO_CLASSIFICATION_CALCULATOR_CORE_H_

#include <limits>
#include <memory>
#include <unordered_map>

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "mediapipe/framework/formats/classification.pb.h"
#include "mediapipe/util/label_map.pb.h"

namespace mediapipe_v01013_based {

// Forward declaration
class TensorsToClassificationCalculatorOptions;

namespace api2 {

// Core configuration for tensor to classification conversion.
struct TensorsToClassificationConfig {
  int top_k = 0;
  bool sort_by_descending_score = false;
  std::unordered_map<int64_t, LabelMapItem> local_label_map;
  bool label_map_loaded = false;
  bool is_binary_classification = false;
  float min_score_threshold = std::numeric_limits<float>::lowest();

  // Set of allowed or ignored class indices.
  struct ClassIndexSet {
    absl::flat_hash_set<int> values;
    bool is_allowlist;
  };
  ClassIndexSet class_index_set;
};

// Initialize the configuration from calculator options.
absl::Status InitializeTensorsToClassificationConfig(
    const TensorsToClassificationCalculatorOptions& options,
    const std::unordered_map<int64_t, LabelMapItem>& external_label_map,
    TensorsToClassificationConfig* config);

// Check if a class index is allowed based on the configuration.
bool IsClassIndexAllowed(const TensorsToClassificationConfig& config,
                         int class_index);

// Set classification label from label map item.
void SetClassificationLabel(const LabelMapItem& label_map_item,
                            Classification* classification);

// Process tensor data and convert to classification list.
std::unique_ptr<ClassificationList> ProcessTensorToClassifications(
    const float* raw_scores,
    int num_classes,
    const TensorsToClassificationConfig& config,
    const std::unordered_map<int64_t, LabelMapItem>& label_map);

}  // namespace api2
}  // namespace mediapipe_v01013_based

#endif  // MEDIAPIPE_CALCULATORS_TENSOR_TENSORS_TO_CLASSIFICATION_CALCULATOR_CORE_H_
