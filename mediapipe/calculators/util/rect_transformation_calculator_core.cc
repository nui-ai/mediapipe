// Copyright 2019 The MediaPipe Authors.
// Licensed under the Apache License, Version 2.0.
#include "mediapipe/calculators/util/rect_transformation_calculator_core.h"
#include "mediapipe/calculators/util/rect_transformation_calculator.pb.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include <algorithm>
#include <cmath>

namespace mediapipe {

  inline float RectTransformationCalculatorCore::NormalizeRadians(float angle) {
    return angle - 2 * M_PI * std::floor((angle - (-M_PI)) / (2 * M_PI));
  }

  RectTransformationCalculatorCore::RectTransformationCalculatorCore(RectTransformationCalculatorOptions &options)
    : options_(options) {}

  float RectTransformationCalculatorCore::ComputeNewRotation(float rotation) const {
    if (options_.has_rotation()) {
      rotation += options_.rotation();
    } else if (options_.has_rotation_degrees()) {
      rotation += M_PI * options_.rotation_degrees() / 180.f;
    }
    return NormalizeRadians(rotation);
  }

  void RectTransformationCalculatorCore::TransformRect(Rect* rect) const {
      float width = rect->width();
      float height = rect->height();
      float rotation = rect->rotation();

      if (options_.has_rotation() || options_.has_rotation_degrees()) {
        rotation = ComputeNewRotation(rotation);
      }
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

  void RectTransformationCalculatorCore::TransformNormalizedRect(NormalizedRect* rect, int image_width, int image_height) const {
    float width = rect->width();
    float height = rect->height();
    float rotation = rect->rotation();

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

    if (options_.square_long()) {
      const float long_side =
          std::max(width * image_width, height * image_height);
      width = long_side / image_width;
      height = long_side / image_height;
    } else if (options_.square_short()) {
      const float short_side =
          std::min(width * image_width, height * image_height);
      width = short_side / image_width;
      height = short_side / image_height;
    }
    rect->set_width(width * options_.scale_x());
    rect->set_height(height * options_.scale_y());
  }
}
