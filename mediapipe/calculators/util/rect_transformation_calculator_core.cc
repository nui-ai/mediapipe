// Copyright 2019 The MediaPipe Authors.
// Licensed under the Apache License, Version 2.0.
//
// the purpose of transformation applied here is typically (i.e. in our pipeline) to either expand a bounding box from its palm detection result box to one which
// is likely to capture the entire hand, or to expand it in a way that it would still capture the entire same hand if the hand moved a little on the next frame.
// as a baseline it works well and forms a good baseline in enabling the overall hand tracking.
//
// the expansion performed elongates the original 2D bounding box in one direction before applying the scaling. this is only a 2D expansion process,
// and hence it has its limitations ― but it seems that first elongating on one axis is softly anatomically informed to compensate; so it can be
// a strong baseline to beat.
//
// how much does it serve both above scnearios?
//
// - to move from a palm to a full-hand fingers-extended bounding box, we want to elongate in one direction, as much as we think that a palm is always broader than long,
//   and we think that the hand is mostly within a certain range of pose/position to the camera plane where that anatomical distinction overlaps with its 2D camera perception.
//   of course, this deteriorates a lot in terms of the 2D perception of the palm's box that we have, at various palm angles (pitch, roll, yaw) that make the 2D lengths
//   unsuited for this kind of assumption.
//
//   in particular, consider the palm pointing almost to the camera. the fingers may extend to their 90 degrees angle from the palm plane, but since the 2D perception
//   in this case is that the 2D screen-axes height of the palm is much smaller than its length, the rectangle will be extended up and down in equal amounts by the
//   said squarization expansion step ― whereas the fingers may mostly only bend to one side of the palm (down, or up, depending on palm orientation) ―
//   if the scaling factor isn't large enough and the fingers are pointing their full length out down (or up), the finger tips may not fall within
//   the expanded rectangle, which may lead the entire landmarks prediction of the resulting box to fail.
//   the same concern applies at different rolls of the palm when the palm is pointing almost to the camera.
//
//   more generally put, scaling the box by screen axes is a tradeoff ― for which an anatomically informed expansion strategy which does not optimize only for certain
//   palm pose and positions as the existing one may avoid failure cases which the current one falls into.
//
//   of course, any assumed box has its opposing error type too: too large and landmark prediction may become spurious.
//
//   a stateful tracking algorithm may both differently build those boxes or differently recover when the boxes yield non-intelligible results.
//   non-intelligible is when a hand is there at one frame and not anywhere detected given the previous position and smoothed out motion of the
//   last few frames ― assuming that the time between frames is small enough and known.
//
// - these concern applies somewhat less to the second scenario ― of reusing the landmarks for the next frame's search box (or just differently).
//
// any useful work on this must start by accumulating tracking failure cases at the frame level, which Gesture Studio enables the marking of,
// assuming judicious accumulation of diverse and relevant data at various background conditions and room lighting angles ― but eliciting
// tracking failure cases at a single easy background and ceiling based lighting should be the way to start.
//

#include "mediapipe/calculators/util/rect_transformation_calculator_core.h"
#include "mediapipe/calculators/util/rect_transformation_calculator.pb.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "absl/log/absl_log.h"
#include <algorithm>
#include <cmath>

namespace mediapipe_v01013_based {

  inline float RectTransformation::NormalizeRadians(float angle) {
    return angle - 2 * M_PI * std::floor((angle - (-M_PI)) / (2 * M_PI));
  }

  RectTransformation::RectTransformation(RectTransformationCalculatorOptions &options)
    : options_(options) {}

  float RectTransformation::ComputeNewRotation(float rotation) const {
    if (options_.has_rotation()) {
      rotation += options_.rotation();
    } else if (options_.has_rotation_degrees()) {
      rotation += M_PI * options_.rotation_degrees() / 180.f;
    }
    return NormalizeRadians(rotation);
  }

  /// expands and shifts the rectangle that contains the palm so that it's likely to cover the entire hand.
  /// the scaling is applied such that either the (in our case) the shorter axis of the rectangle
  /// is enlarged to be equal to the longer axis of the rectangle, before applying the scaling ―
  /// thus making it more square, as much as the scaling factors for width and height are equal.
  /// (the input argument is modified in place as an input-output argument,
  ///  unlike most mediapipe functions we use in this pipeline).
  void RectTransformation::Expand(Rect* rect) const {
    float width = rect->width();
    float height = rect->height();
    float rotation = rect->rotation();

    if (options_.has_rotation() || options_.has_rotation_degrees()) {
      rotation = ComputeNewRotation(rotation);
    }

    // shift its center
    if (rotation == 0.f) {
      rect->set_x_center(rect->x_center() + width * options_.shift_x());
      rect->set_y_center(rect->y_center() + height * options_.shift_y());
    } else {
      const float x_shift = width * options_.shift_x() * std::cos(rotation) -
                            height * options_.shift_y() * std::sin(rotation);
      const float y_shift = width * options_.shift_x() * std::sin(rotation) +
                            height * options_.shift_y() * std::cos(rotation);
      rect->set_x_center(rect->x_center() + x_shift);
      rect->set_y_center(rect->y_center() + y_shift);
    }

    // scale it
    if (options_.square_long()) {
      const float long_side = std::max(width, height);
      width = long_side;
      height = long_side;
    } else if (options_.square_short()) {
      const float short_side = std::min(width, height);
      width = short_side;
      height = short_side;
    }
    rect->set_width(width * options_.scale_x());
    rect->set_height(height * options_.scale_y());
  }

  /// expands and shifts the rectangle that contains the palm so that it's likely to cover the entire hand.
  /// the scaling is applied such that either the (in our case) the shorter axis of the rectangle
  /// is enlarged to be equal to the longer axis of the rectangle, before applying the scaling ―
  /// thus making it more square, as much as the scaling factors for width and height are equal.
  /// (the input argument is modified in place as an input-output argument,
  ///  unlike most mediapipe functions we use in this pipeline).
  ///
  /// steps:
  /// - moves the rectangle's center coords by the constant shift ratios (complying with the rectangle's rotation)
  /// - extend the rectangle's shorter axis to the length of its longer axis, thus becoming a square-like rectangle
  ///   which has its originally shorter axis length extend to the length of its longer axis.
  /// - scale the squared rectangle of the previous step over its both axes, according to the
  ///   constant scaling parameters.
  ///
  /// the above modifications are applied to the input rectangle in-place (it's an input-output argument)
  void RectTransformation::ExpandNormalizedRect(NormalizedRect* rect, const int image_width, const int image_height) const {
    float width = rect->width();
    float height = rect->height();
    float rotation = rect->rotation();

    // shift its center
    if (options_.has_rotation() || options_.has_rotation_degrees()) {
      rotation = ComputeNewRotation(rotation);
    }
    if (rotation == 0.f) {
      rect->set_x_center(rect->x_center() + width * options_.shift_x());
      rect->set_y_center(rect->y_center() + height * options_.shift_y());
    } else {
      const float x_shift =
          (image_width * width * options_.shift_x() * std::cos(rotation) -
           image_height * height * options_.shift_y() * std::sin(rotation)) /
          image_width;
      const float y_shift =
          (image_width * width * options_.shift_x() * std::sin(rotation) +
           image_height * height * options_.shift_y() * std::cos(rotation)) /
          image_height;
      rect->set_x_center(rect->x_center() + x_shift);
      rect->set_y_center(rect->y_center() + y_shift);
    }

    // extend the rectangle's shorter axis to match the length of its longer one
    if (options_.square_long()) {

      // figure which axis of the rotated rectangle is longer (in the screen aspect-ratio cognizent sense).
      const float long_side = std::max(width * image_width, height * image_height);

      // extend the originally shorter axis of the rectangle to the same length as its longer axis,
      // thus both expanding it and making it square in screen aspect ratio terms.
      // so the ratio width/height after this step is always constant (as long as
      // the image aspect ratio is always the same).
      width = long_side / image_width;
      height = long_side / image_height;
      //ABSL_LOG(INFO) << "width/height = " << width/height;

    // extend the other way around, but we never get here in our pipeline
    } else if (options_.square_short()) {
      const float short_side = std::min(width * image_width, height * image_height);
      width = short_side / image_width;
      height = short_side / image_height;
    }

    // finally, scales the (above extended) rectangle axes by the constant scaling factors
    rect->set_width(width * options_.scale_x());
    rect->set_height(height * options_.scale_y());
  }
}
