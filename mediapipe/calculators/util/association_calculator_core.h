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
#include "mediapipe/framework/calculator_context.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/rectangle.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/util/rectangle_util.h"

namespace mediapipe_v01013_based {

  // helper function for the below
  static inline absl::StatusOr<Rectangle_f> RectangleFromNormalizedRect(
      const ::mediapipe_v01013_based::NormalizedRect& input) {
    if (!input.has_x_center() || !input.has_y_center() || !input.has_width() ||
        !input.has_height()) {
      return absl::InternalError("Missing dimensions in NormalizedRect.");
        }
    const float xmin = input.x_center() - input.width() / 2.0;
    const float ymin = input.y_center() - input.height() / 2.0;
    // TODO: Support rotation for rectangle.
    return Rectangle_f(xmin, ymin, input.width(), input.height());
  }

    /// adds the new NormalizedRect to the collection of NormalizedRect, while discarding any of the collection's existing
    /// instances which have overlap with it above the given IoU threshold; so the new element squeezes out any one or more
    /// existing instances which have overlap with it above the given IoU threshold.
    inline absl::Status AddWhileDiscardingByIoU(const NormalizedRect& new_normalized_rect, std::list<NormalizedRect>* normalized_rects, float iou_similarity_threshold) {

    // check IoU of the new rect with the each rect of the list, transforming them from NormalizedRect to Rect for the IoU checking
    MP_ASSIGN_OR_RETURN(Rectangle_f new_rect, ::mediapipe_v01013_based::RectangleFromNormalizedRect(new_normalized_rect));
    for (auto uit = normalized_rects->begin(); uit != normalized_rects->end();) {
      MP_ASSIGN_OR_RETURN(Rectangle_f rect, RectangleFromNormalizedRect(*uit));

      // remove existing IoU threshold overlapping rectangle if threshold overlapping with the one being added
      if (CalculateIou(new_rect, rect) > iou_similarity_threshold) {
        ABSL_LOG(INFO) << "filtering by association is pushing out an overlapping element.";
        uit = normalized_rects->erase(uit);
      } else {
        ++uit;
      }
    }
    normalized_rects->push_back(new_normalized_rect);
    return absl::OkStatus();
  }

  /// naively smashes together the given set of palm detection rectangles from the current frame's explicit palm detection inference
  /// and the given set of detection rectangles derived from the previous frame's landmarks inference, filtering out any partially
  /// overlapping ones (by its overlap threshold) by a greedy ordering where the last being added wins (over those having IoU
  /// threshold overlap with it).
  ///
  /// and when there are no detections at all, it should just pass forward no detections.
  ///
  /// notice the isomorphic effect of filtering each set and both sets by the same greedy win order,
  /// with preference given to the rectangles from the previous frame's landmarks inference set over
  /// those from the palm detection phase, is arguably only a baseline taken from the original pipeline,
  ///
  /// by letting the rectangles from the last frame's landmarks inference win over those from the current frame's palm detection step
  /// in any case of IoU threshold overlap ― it may helps with tracking stability in some cases and degrade it in other ones ―
  /// so it's not a good idea to change this outside of a complete uber-overhaul of hand tracking, as small and specific
  /// scenario wins may just come at the expense of worsening the overall.
  ///
  /// filtering is an epic in our pipeline which when reworked would consume the current baseline step as well
  /// as all other ones when being redesigned for specific sets of desiderata, for now we just keep the legacy
  /// behaviors at all the arbitrarily disparate filtering steps of the pipeline.
  ///
  /// note that for tracking *hand identity* with multiple hands, we will want to know or score which new
  /// rectangle (and hand pose from landmarks in it) corresponds to which one in the previous frame,
  /// rather than trust that circumstantially they will reach here by the same order,
  /// a guarantee that's only weekly effected by the SSD anchors
  /// and certainly not reliably if hands move liberally across the scene.
  /// this kind of identity mapping is currently abscnet,
  /// other than by that circumstantial effect.
  ///
  /// some considerations apply to one hand tracking and not in a generalized way to multi-hand tracking:
  /// with a single hand the current de-facto filtering algorithm may have an effect of avoiding noisy
  /// palm detections from hijacking the tracking of an already tracked hand. (which may not directly
  /// generalize to a helpful statement for the case of two real hands being tracked)
  template <typename T>
  absl::StatusOr<std::list<T>> IouFilterMerge(
      const std::vector<T>& rects_from_palm_detection,
      const std::vector<T>& rects_from_landmarks_inference,
      float min_similarity_threshold = 0.5) {

    std::list<T> result_set;  // the final set of hand rectangles passing forward

    // this step places the hand rectangles derived from explicit palm detections into the result set ―
    // while filtering them by IoU thershold in a naive order where the later element always "wins".
    if (!rects_from_palm_detection.empty()) {
      result_set.push_back(rects_from_palm_detection[0]);
      for (size_t j = 1; j < rects_from_palm_detection.size(); ++j) {
        MP_RETURN_IF_ERROR(AddWhileDiscardingByIoU(rects_from_palm_detection[j], &result_set, min_similarity_threshold));
      }
    }

    // this step places the hand rectangles derived from landmarks inference into the result set ―
    // while filtering them by IoU threshold by the same naive order ―
    // each rectangle from landmarks inference wins over any ones already in the result set admitted from the explicit palm detections set,
    // if they have IoU threshold overlap ... and each such element also wins over any previous one from its own set if they have IoU threshold overlap.
    if (!rects_from_landmarks_inference.empty()) {
      for (size_t vi = 0; vi < rects_from_landmarks_inference.size(); ++vi) {
        MP_RETURN_IF_ERROR(AddWhileDiscardingByIoU(rects_from_landmarks_inference[vi], &result_set, min_similarity_threshold));
      }
    }
    return result_set;
  }

}  // namespace mediapipe_v01013_based

#endif  // MEDIAPIPE_CALCULATORS_UTIL_ASSOCIATION_CALCULATOR_CORE_H_

