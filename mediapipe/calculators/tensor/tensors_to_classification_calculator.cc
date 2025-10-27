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

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "mediapipe/calculators/tensor/tensors_to_classification_calculator.pb.h"
#include "mediapipe/calculators/tensor/tensors_to_classification_calculator_core.h"
#include "mediapipe/framework/api2/node.h"
#include "mediapipe/framework/api2/port.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/classification.pb.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status_macros.h"
#include "mediapipe/framework/resources.h"
#include "mediapipe/util/label_map.pb.h"
#include "mediapipe/util/label_map_util.h"
#include "mediapipe/util/resource_util.h"
#if defined(MEDIAPIPE_MOBILE)
#include "mediapipe/util/android/file/base/file.h"
#include "mediapipe/util/android/file/base/helpers.h"
#else
#include "mediapipe/framework/port/file_helpers.h"
#endif

namespace mediapipe {
namespace api2 {

// Convert result tensors from classification models into MediaPipe
// classifications.
class ExtractHandedness : public Node {
 public:
  static constexpr Input<std::vector<Tensor>> kInTensors{"TENSORS"};
  static constexpr Output<ClassificationList> kOutClassificationList{
      "CLASSIFICATIONS"};
  MEDIAPIPE_NODE_CONTRACT(kInTensors, kOutClassificationList);

  absl::Status Open(CalculatorContext* cc) override;
  absl::Status Process(CalculatorContext* cc) override;
  absl::Status Close(CalculatorContext* cc) override;

 private:
  TensorsToClassificationConfig config_;
  std::unordered_map<int64_t, LabelMapItem> external_label_map_;

  const std::unordered_map<int64_t, LabelMapItem>& GetLabelMap(
      CalculatorContext* cc);
};
MEDIAPIPE_REGISTER_NODE(ExtractHandedness);

absl::Status ExtractHandedness::Open(CalculatorContext* cc) {
  const auto& options = cc->Options<TensorsToClassificationCalculatorOptions>();

  // Handle label map loading from file path
  if (options.has_label_map_path()) {
    std::string string_path;
    MP_ASSIGN_OR_RETURN(string_path,
                        PathToResourceAsFile(options.label_map_path()));
    MP_ASSIGN_OR_RETURN(std::unique_ptr<mediapipe::Resource> label_map,
                        cc->GetResources().Get(string_path));
    proto_ns::Map<int64_t, LabelMapItem> temp_label_map;
    MP_ASSIGN_OR_RETURN(
        temp_label_map,
        BuildLabelMapFromFiles(label_map->ToStringView(),
                               /*display_names_file_contents*/ {}));
    // Convert proto_ns::Map to std::unordered_map
    for (const auto& entry : temp_label_map) {
      external_label_map_[entry.first] = entry.second;
    }
  }

  // Initialize configuration using core function
  return InitializeTensorsToClassificationConfig(options, external_label_map_, &config_);
}

absl::Status ExtractHandedness::Process(CalculatorContext* cc) {
  const auto& input_tensors = *kInTensors(cc);
  RET_CHECK_EQ(input_tensors.size(), 1);
  RET_CHECK(input_tensors[0].element_type() == Tensor::ElementType::kFloat32);

  int num_classes = input_tensors[0].shape().num_elements();

  if (config_.is_binary_classification) {
    RET_CHECK_EQ(num_classes, 1);
    // Number of classes for binary classification.
    num_classes = 2;
  }
  if (config_.label_map_loaded) {
    RET_CHECK_EQ(num_classes, GetLabelMap(cc).size());
  }
  auto view = input_tensors[0].GetCpuReadView();
  auto raw_scores = view.buffer<float>();

  // Use core function to process tensor to classifications
  auto classification_list = ProcessTensorToClassifications(
      raw_scores, num_classes, config_, GetLabelMap(cc));

  kOutClassificationList(cc).Send(std::move(classification_list));
  return absl::OkStatus();
}

absl::Status ExtractHandedness::Close(CalculatorContext* cc) {
  return absl::OkStatus();
}

const std::unordered_map<int64_t, LabelMapItem>&
ExtractHandedness::GetLabelMap(CalculatorContext* cc) {
  static std::unordered_map<int64_t, LabelMapItem> temp_map;
  if (!external_label_map_.empty()) {
    return external_label_map_;
  } else {
    const auto& proto_map = cc->Options<TensorsToClassificationCalculatorOptions>()
                               .label_items();
    temp_map.clear();
    for (const auto& entry : proto_map) {
      temp_map[entry.first] = entry.second;
    }
    return temp_map;
  }
}

}  // namespace api2
}  // namespace mediapipe
