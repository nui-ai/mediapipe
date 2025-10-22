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

#ifndef MEDIAPIPE_CALCULATORS_UTIL_ASSOCIATION_CALCULATOR_CORE_H_
#define MEDIAPIPE_CALCULATORS_UTIL_ASSOCIATION_CALCULATOR_CORE_H_

#include <list>
#include <memory>
#include <vector>

#include "absl/log/absl_check.h"
#include "absl/memory/memory.h"
#include "mediapipe/calculators/util/association_calculator.pb.h"
#include "mediapipe/framework/calculator_context.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/collection_item_id.h"
#include "mediapipe/framework/port/canonical_errors.h"
#include "mediapipe/framework/port/rectangle.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/util/rectangle_util.h"

namespace mediapipe {

  absl::StatusOr<Rectangle_f> GetRectangle(
      const ::mediapipe::NormalizedRect& input) {
    if (!input.has_x_center() || !input.has_y_center() || !input.has_width() ||
        !input.has_height()) {
      return absl::InternalError("Missing dimensions in NormalizedRect.");
        }
    const float xmin = input.x_center() - input.width() / 2.0;
    const float ymin = input.y_center() - input.height() / 2.0;
    // TODO: Support rotation for rectangle.
    return Rectangle_f(xmin, ymin, input.width(), input.height());
  }

  template <typename T>
  absl::Status AddElementToListWhileSuppressing(const T& element, std::list<T>* current, float min_similarity_threshold) {
    MP_ASSIGN_OR_RETURN(auto new_element, GetRectangle(element));
    for (auto uit = current->begin(); uit != current->end();) {
      MP_ASSIGN_OR_RETURN(auto collection_element, GetRectangle(*uit));
      if (CalculateIou(new_element, collection_element) > min_similarity_threshold) {
        ABSL_LOG(INFO) << "filtering by association is pushing out an overlapping element.";
        uit = current->erase(uit);
      } else {
        ++uit;
      }
    }
    current->push_back(element);
    return absl::OkStatus();
  }

  template <typename T>
  absl::StatusOr<std::list<T>> FilterByAssociation(
      const std::vector<T>& explicit_palm_detections,
      const std::vector<T>& landmarks_derived_palm_detections,
      float min_similarity_threshold = 0.5) {

    std::list<T> result_set;
    if (!explicit_palm_detections.empty()) {
      result_set.push_back(explicit_palm_detections[0]);
      for (size_t j = 1; j < explicit_palm_detections.size(); ++j) {
        MP_RETURN_IF_ERROR(AddElementToListWhileSuppressing(explicit_palm_detections[j], &result_set, min_similarity_threshold));
      }
    }

    if (!landmarks_derived_palm_detections.empty()) {
      for (size_t vi = 0; vi < landmarks_derived_palm_detections.size(); ++vi) {
        MP_RETURN_IF_ERROR(AddElementToListWhileSuppressing(landmarks_derived_palm_detections[vi], &result_set, min_similarity_threshold));
      }
    }
    return result_set;
  }

}  // namespace mediapipe

#endif  // MEDIAPIPE_CALCULATORS_UTIL_ASSOCIATION_CALCULATOR_CORE_H_

