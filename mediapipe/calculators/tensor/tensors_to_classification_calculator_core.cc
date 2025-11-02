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

#include "mediapipe/calculators/tensor/tensors_to_classification_calculator_core.h"

#include <algorithm>
#include <memory>

#include "mediapipe/calculators/tensor/tensors_to_classification_calculator.pb.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status_macros.h"

namespace mediapipe_v01013_based {
namespace api2 {

absl::Status InitializeTensorsToClassificationConfig(
    const TensorsToClassificationCalculatorOptions& options,
    const std::unordered_map<int64_t, LabelMapItem>& external_label_map,
    TensorsToClassificationConfig* config) {
  config->top_k = options.top_k();
  config->sort_by_descending_score = options.sort_by_descending_score();

  if (options.has_label_map()) {
    for (int i = 0; i < options.label_map().entries_size(); ++i) {
      const auto& entry = options.label_map().entries(i);
      RET_CHECK(config->local_label_map.find(entry.id()) == config->local_label_map.end())
          << "Duplicate id found: " << entry.id();
      LabelMapItem item;
      item.set_name(entry.label());
      config->local_label_map[entry.id()] = std::move(item);
    }
    config->label_map_loaded = true;
  } else if (!external_label_map.empty() || !options.label_items().empty()) {
    config->label_map_loaded = true;
  }

  if (options.has_min_score_threshold()) {
    config->min_score_threshold = options.min_score_threshold();
  }
  config->is_binary_classification = options.binary_classification();

  if (config->is_binary_classification) {
    RET_CHECK(options.allow_classes().empty() &&
              options.ignore_classes().empty());
  }

  if (!options.allow_classes().empty()) {
    RET_CHECK(options.ignore_classes().empty());
    config->class_index_set.is_allowlist = true;
    for (int i = 0; i < options.allow_classes_size(); ++i) {
      config->class_index_set.values.insert(options.allow_classes(i));
    }
  } else {
    config->class_index_set.is_allowlist = false;
    for (int i = 0; i < options.ignore_classes_size(); ++i) {
      config->class_index_set.values.insert(options.ignore_classes(i));
    }
  }

  return absl::OkStatus();
}

bool IsClassIndexAllowed(const TensorsToClassificationConfig& config,
                         int class_index) {
  if (config.class_index_set.values.empty()) {
    return true;
  }
  if (config.class_index_set.is_allowlist) {
    return config.class_index_set.values.contains(class_index);
  } else {
    return !config.class_index_set.values.contains(class_index);
  }
}

void SetClassificationLabel(const LabelMapItem& label_map_item,
                            Classification* classification) {
  classification->set_label(label_map_item.name());
  if (label_map_item.has_display_name()) {
    classification->set_display_name(label_map_item.display_name());
  }
}

std::unique_ptr<ClassificationList> ProcessTensorToClassifications(
    const float* raw_scores,
    int num_classes,
    const TensorsToClassificationConfig& config,
    const std::unordered_map<int64_t, LabelMapItem>& label_map) {
  auto classification_list = std::make_unique<ClassificationList>();

  if (config.is_binary_classification) {
    Classification* class_first = classification_list->add_classification();
    Classification* class_second = classification_list->add_classification();
    class_first->set_index(0);
    class_second->set_index(1);
    class_first->set_score(raw_scores[0]);
    class_second->set_score(1. - raw_scores[0]);

    if (config.label_map_loaded) {
      SetClassificationLabel(label_map.at(0), class_first);
      SetClassificationLabel(label_map.at(1), class_second);
    }
  } else {
    for (int i = 0; i < num_classes; ++i) {
      if (!IsClassIndexAllowed(config, i)) {
        continue;
      }
      if (raw_scores[i] < config.min_score_threshold) {
        continue;
      }
      Classification* classification =
          classification_list->add_classification();
      classification->set_index(i);
      classification->set_score(raw_scores[i]);
      if (config.label_map_loaded) {
        SetClassificationLabel(label_map.at(i), classification);
      }
    }
  }

  auto raw_classification_list = classification_list->mutable_classification();
  if (config.top_k > 0) {
    int desired_size =
        std::min(classification_list->classification_size(), config.top_k);
    std::partial_sort(raw_classification_list->begin(),
                      raw_classification_list->begin() + desired_size,
                      raw_classification_list->end(),
                      [](const Classification a, const Classification b) {
                        return a.score() > b.score();
                      });

    if (desired_size >= config.top_k) {
      // Resizes the underlying list to have only top_k_ classifications.
      raw_classification_list->DeleteSubrange(
          config.top_k, raw_classification_list->size() - config.top_k);
    }
  } else if (config.sort_by_descending_score) {
    std::sort(raw_classification_list->begin(), raw_classification_list->end(),
              [](const Classification a, const Classification b) {
                return a.score() > b.score();
              });
  }

  return classification_list;
}

}  // namespace api2
}  // namespace mediapipe_v01013_based
