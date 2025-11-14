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
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "mediapipe/calculators/util/non_max_suppression_calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/location.h"
#include "mediapipe/framework/port/rectangle.h"
#include "mediapipe/calculators/util/non_max_suppression_calculator.h"

namespace hand_tracking_mp_lean {

typedef std::vector<Detection> Detections;
typedef std::vector<std::pair<int, float>> IndexedScores;

namespace {

constexpr char kImageTag[] = "IMAGE";

bool SortBySecond(const std::pair<int, float>& indexed_score_0,
                  const std::pair<int, float>& indexed_score_1) {
  return (indexed_score_0.second > indexed_score_1.second);
}

// Removes all but the max scoring label and its score from the detection.
// Returns true if the detection has at least one label.
bool RetainMaxScoringLabelOnly(Detection* detection) {
  if (detection->label_id_size() == 0 && detection->label_size() == 0) {
    return false;
  }
  ABSL_CHECK(detection->label_id_size() == detection->score_size() ||
             detection->label_size() == detection->score_size())
      << "Number of scores must be equal to number of detections.";

  std::vector<std::pair<int, float>> indexed_scores;
  indexed_scores.reserve(detection->score_size());
  for (int k = 0; k < detection->score_size(); ++k) {
    indexed_scores.push_back(std::make_pair(k, detection->score(k)));
  }
  std::sort(indexed_scores.begin(), indexed_scores.end(), SortBySecond);
  const int top_index = indexed_scores[0].first;
  detection->clear_score();
  detection->add_score(indexed_scores[0].second);
  if (detection->label_id_size() > top_index) {
    const int top_label_id = detection->label_id(top_index);
    detection->clear_label_id();
    detection->add_label_id(top_label_id);
  } else {
    const std::string top_label = detection->label(top_index);
    detection->clear_label();
    detection->add_label(top_label);
  }

  return true;
}

// Computes an overlap similarity between two rectangles. Similarity measure is
// defined by overlap_type parameter.
float OverlapSimilarity(
    const NonMaxSuppressionCalculatorOptions::OverlapType overlap_type,
    const Rectangle_f& rect1, const Rectangle_f& rect2) {
  if (!rect1.Intersects(rect2)) return 0.0f;
  const float intersection_area = Rectangle_f(rect1).Intersect(rect2).Area();
  float normalization;
  switch (overlap_type) {
    case NonMaxSuppressionCalculatorOptions::JACCARD:
      normalization = Rectangle_f(rect1).Union(rect2).Area();
      break;
    case NonMaxSuppressionCalculatorOptions::MODIFIED_JACCARD:
      normalization = rect2.Area();
      break;
    case NonMaxSuppressionCalculatorOptions::INTERSECTION_OVER_UNION:
      normalization = rect1.Area() + rect2.Area() - intersection_area;
      break;
    default:
      ABSL_LOG(FATAL) << "Unrecognized overlap type: " << overlap_type;
  }
  return normalization > 0.0f ? intersection_area / normalization : 0.0f;
}

// Computes an overlap similarity between two locations by first extracting the
// relative box (dimension normalized by frame width/height) from the location.
float OverlapSimilarity(
    const int frame_width, const int frame_height,
    const NonMaxSuppressionCalculatorOptions::OverlapType overlap_type,
    const Location& location1, const Location& location2) {
  const auto rect1 = location1.ConvertToRelativeBBox(frame_width, frame_height);
  const auto rect2 = location2.ConvertToRelativeBBox(frame_width, frame_height);
  return OverlapSimilarity(overlap_type, rect1, rect2);
}

// Computes an overlap similarity between two locations by first extracting the
// relative box from the location. It assumes that a relative-box representation
// is already available in the location, and therefore frame width and height
// are not needed for further normalization.
float OverlapSimilarity(
    const NonMaxSuppressionCalculatorOptions::OverlapType overlap_type,
    const Location& location1, const Location& location2) {
  const auto rect1 = location1.GetRelativeBBox();
  const auto rect2 = location2.GetRelativeBBox();
  return OverlapSimilarity(overlap_type, rect1, rect2);
}

// Copy all the scores (there is a single score in each detection after
// pruning detections) to an indexed vector for sorting. The first value is
// the index of the detection in the original vector from which the score
// stems, while the second is the actual score.
IndexedScores GetIndexedScores(const Detections& detections) {
  IndexedScores indexed_scores;
  indexed_scores.reserve(detections.size());
  for (int index = 0; index < detections.size(); ++index) {
    indexed_scores.push_back(
        std::make_pair(index, detections.at(index).score(0)));
  }
  std::sort(indexed_scores.begin(), indexed_scores.end(), SortBySecond);
  return indexed_scores;
}

// Standard non-maximum suppression algorithm.
void NonMaxSuppression(const IndexedScores& indexed_scores,
                       const Detections& detections,
                       int max_num_detections,
                       const NonMaxSuppressionCalculatorOptions& options,
                       bool has_dimensions,
                       int frame_width,
                       int frame_height,
                       Detections* output_detections) {

  std::vector<Location> retained_locations;
  retained_locations.reserve(max_num_detections);

  // iterate the detections by decreasing score, admitting them into the result collection of detections
  // while suppressing those that overlap too much with a previously retained detection.
  for (const auto& indexed_score : indexed_scores) {
    const auto& detection = detections[indexed_score.first];
    if (options.min_score_threshold() > 0 &&
        detection.score(0) < options.min_score_threshold()) {
      break;
    }
    const Location location(detection.location_data());
    bool suppressed = false;
    // The current detection is suppressed iff there exists a retained detection whose location overlaps more than the specified
    // threshold with the location of the current detection. since we iterate by decreasing score, we keep the higher scoring one
    // of all those considered as partially overlapping by the overlap criteria.
    for (const auto& retained_location : retained_locations) {
      float similarity;
      if (has_dimensions) {
        similarity = OverlapSimilarity(frame_width, frame_height, options.overlap_type(), retained_location, location);
      } else {
        similarity = OverlapSimilarity(options.overlap_type(), retained_location, location);
      }
      if (similarity > options.min_suppression_threshold()) {
        suppressed = true;
        break;
      }
    }
    if (!suppressed) {
      // admit into the surviving detections collection
      output_detections->push_back(detection);
      retained_locations.push_back(location);
    }
    if (output_detections->size() >= max_num_detections) {
      break;
    }
  }
}

// Weighted non-maximum suppression algorithm: same as standard NMS, but the location of a retained detection is not the location of the retained detection as in the default NMS implememntation,
// but the weighted average of all locations that were considered overlapping with it ― weighted by their detection scores as seen below.
void WeightedNonMaxSuppression(const IndexedScores& indexed_scores,
                               const Detections& detections,
                               int max_num_detections,
                               const NonMaxSuppressionCalculatorOptions& options,
                               Detections* output_detections) {
  IndexedScores remained_indexed_scores;
  remained_indexed_scores.assign(indexed_scores.begin(),
                                 indexed_scores.end());

  IndexedScores remained;
  IndexedScores candidates;
  output_detections->clear();
  while (!remained_indexed_scores.empty()) {
    const int original_indexed_scores_size = remained_indexed_scores.size();
    const auto& detection = detections[remained_indexed_scores[0].first];
    if (options.min_score_threshold() > 0 &&
        detection.score(0) < options.min_score_threshold()) {
      break;
    }
    remained.clear();
    candidates.clear();
    const Location location(detection.location_data());
    // This includes the first box.
    for (const auto& indexed_score : remained_indexed_scores) {
      Location rest_location(detections[indexed_score.first].location_data());
      float similarity =
          OverlapSimilarity(options.overlap_type(), rest_location, location);
      if (similarity > options.min_suppression_threshold()) {
        candidates.push_back(indexed_score);
      } else {
        remained.push_back(indexed_score);
      }
    }
    auto weighted_detection = detection;
    if (!candidates.empty()) {
      const int num_keypoints =
          detection.location_data().relative_keypoints_size();
      std::vector<float> keypoints(num_keypoints * 2);
      float w_xmin = 0.0f;
      float w_ymin = 0.0f;
      float w_xmax = 0.0f;
      float w_ymax = 0.0f;
      float total_score = 0.0f;
      for (const auto& candidate : candidates) {
        total_score += candidate.second;
        const auto& location_data =
            detections[candidate.first].location_data();
        const auto& bbox = location_data.relative_bounding_box();
        w_xmin += bbox.xmin() * candidate.second;
        w_ymin += bbox.ymin() * candidate.second;
        w_xmax += (bbox.xmin() + bbox.width()) * candidate.second;
        w_ymax += (bbox.ymin() + bbox.height()) * candidate.second;

        for (int i = 0; i < num_keypoints; ++i) {
          keypoints[i * 2] +=
              location_data.relative_keypoints(i).x() * candidate.second;
          keypoints[i * 2 + 1] +=
              location_data.relative_keypoints(i).y() * candidate.second;
        }
      }
      auto* weighted_location = weighted_detection.mutable_location_data()
                                    ->mutable_relative_bounding_box();
      weighted_location->set_xmin(w_xmin / total_score);
      weighted_location->set_ymin(w_ymin / total_score);
      weighted_location->set_width((w_xmax / total_score) -
                                   weighted_location->xmin());
      weighted_location->set_height((w_ymax / total_score) -
                                    weighted_location->ymin());
      for (int i = 0; i < num_keypoints; ++i) {
        auto* keypoint = weighted_detection.mutable_location_data()
                              ->mutable_relative_keypoints(i);
        keypoint->set_x(keypoints[i * 2] / total_score);
        keypoint->set_y(keypoints[i * 2 + 1] / total_score);
      }
    }

    output_detections->push_back(weighted_detection);
    // Breaks the loop if the size of indexed scores doesn't change after an iteration.
    if (original_indexed_scores_size == remained.size()) {
      break;
    } else {
      remained_indexed_scores = std::move(remained);
    }
  }
}

// Performs non-maximum suppression on the input detections.
void DoNonMaxSuppression(Detections& input_detections,
                         const NonMaxSuppressionCalculatorOptions& options,
                         bool has_dimensions,
                         int frame_width,
                         int frame_height,
                         Detections* output_detections) {

  // first removes all but the maximum scoring label from each input detection. This corresponds to non-maximum suppression among detections which have identical locations.
  // this cross-class step is not relevant to our palm detection SSD model, as it was trained for only once class (palm).
  Detections pruned_detections;
  pruned_detections.reserve(input_detections.size());
  for (auto& detection : input_detections) {
    if (RetainMaxScoringLabelOnly(&detection)) {
      pruned_detections.push_back(detection);
    }
  }

  IndexedScores indexed_scores = GetIndexedScores(pruned_detections);
  const int max_num_detections =
      (options.max_num_detections() > -1)  // defaults to -1 in the protobuf underlying definition.
          ? options.max_num_detections()
          : static_cast<int>(indexed_scores.size());  // unlimited. so next steps after the NMS step will have to deal with all the surviving detections.

  // A set of detections and locations, wrapping the location data from each detection, which are retained after the non-maximum suppression.
  output_detections->reserve(max_num_detections);

  if (options.algorithm() == NonMaxSuppressionCalculatorOptions::WEIGHTED) {
    WeightedNonMaxSuppression(indexed_scores, pruned_detections,
                              max_num_detections, options, output_detections);
  } else { // default algorithm if WEIGHTED is not specified
    NonMaxSuppression(indexed_scores, pruned_detections, max_num_detections,
                      options, has_dimensions, frame_width, frame_height,
                      output_detections);
  }
}

} // end anonymous namespace

// filters down a set of input detections by non-maximum suppression according to
// the provided parameters for suppression. Returns a vector of the surviving detections.
std::unique_ptr<Detections> FilterDetectionsByNonMaximumSuppression(
    const Detections& input_detections,
    const NonMaxSuppressionCalculatorOptions& options,
    bool has_dimensions,
    int frame_width,
    int frame_height) {

  // Check if there are any detections at all.
  if (input_detections.empty()) {
    return std::make_unique<Detections>();
  }

  auto retained_detections = std::make_unique<Detections>();
  if (options.multiclass_nms()) {
    absl::flat_hash_map<int, Detections> category_index_to_detections;
    for (const auto& detection : input_detections) {
      for (int index : detection.label_id()) {
        category_index_to_detections[index].push_back(detection);
      }
    }
    // For each category, do non-maximum suppression separately.
    Detections detections_nms;
    for (auto& [index, detections] : category_index_to_detections) {
      auto retained_detections_per_category = std::make_unique<Detections>();
      auto detections_copy = detections;  // Create a copy to avoid const issues
      DoNonMaxSuppression(detections_copy, options, has_dimensions,
                          frame_width, frame_height,
                          retained_detections_per_category.get());

      detections_nms.insert(detections_nms.end(),
                            retained_detections_per_category->begin(),
                            retained_detections_per_category->end());
    }

    // Descending sort and shrink the collected detections according to max
    // num detections.
    IndexedScores indexed_scores = GetIndexedScores(detections_nms);
    int max_num_detections = static_cast<int>(indexed_scores.size());
    if (options.max_num_detections() > -1) {
      max_num_detections = std::min(max_num_detections, options.max_num_detections());
    }
    retained_detections->reserve(max_num_detections);
    for (int i = 0; i < max_num_detections; i++) {
      retained_detections->push_back(
          detections_nms.at(indexed_scores.at(i).first));
    }
  } else {
    auto input_detections_copy = input_detections;  // Create a copy to avoid const issues
    DoNonMaxSuppression(input_detections_copy, options, has_dimensions,
                        frame_width, frame_height,
                        retained_detections.get());
  }

  return retained_detections;
}

}  // namespace hand_tracking_mp_lean
