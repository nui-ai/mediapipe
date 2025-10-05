// Copyright 2019 The MediaPipe Authors.
// Licensed under the Apache License, Version 2.0.
#include "mediapipe/calculators/util/rect_transformation_calculator_core.h"
#include "mediapipe/calculators/util/rect_transformation_calculator.pb.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include <algorithm>
#include <cmath>

namespace mediapipe {

inline float NormalizeRadians(float angle) {
  return angle - 2 * M_PI * std::floor((angle - (-M_PI)) / (2 * M_PI));
}

float ComputeNewRotation(const RectTransformationCalculatorOptions& options, float rotation) {
  if (options.has_rotation()) {
    rotation += options.rotation();
  } else if (options.has_rotation_degrees()) {
    rotation += M_PI * options.rotation_degrees() / 180.f;
  }
  return NormalizeRadians(rotation);
}

void TransformRect(const RectTransformationCalculatorOptions& options, Rect* rect) {
  float width = rect->width();
  float height = rect->height();
  float rotation = rect->rotation();

  if (options.has_rotation() || options.has_rotation_degrees()) {
    rotation = ComputeNewRotation(options, rotation);
  }
  if (rotation == 0.f) {
    rect->set_x_center(rect->x_center() + width * options.shift_x());
    rect->set_y_center(rect->y_center() + height * options.shift_y());
  } else {
    const float x_shift = width * options.shift_x() * std::cos(rotation) -
                          height * options.shift_y() * std::sin(rotation);
    const float y_shift = width * options.shift_x() * std::sin(rotation) +
                          height * options.shift_y() * std::cos(rotation);
    rect->set_x_center(rect->x_center() + x_shift);
    rect->set_y_center(rect->y_center() + y_shift);
  }

  if (options.square_long()) {
    const float long_side = std::max(width, height);
    width = long_side;
    height = long_side;
  } else if (options.square_short()) {
    const float short_side = std::min(width, height);
    width = short_side;
    height = short_side;
  }
  rect->set_width(width * options.scale_x());
  rect->set_height(height * options.scale_y());
}

void TransformNormalizedRect(const RectTransformationCalculatorOptions& options, NormalizedRect* rect, int image_width, int image_height) {
  float width = rect->width();
  float height = rect->height();
  float rotation = rect->rotation();

  if (options.has_rotation() || options.has_rotation_degrees()) {
    rotation = ComputeNewRotation(options, rotation);
  }
  if (rotation == 0.f) {
    rect->set_x_center(rect->x_center() + width * options.shift_x());
    rect->set_y_center(rect->y_center() + height * options.shift_y());
  } else {
    const float x_shift =
        (image_width * width * options.shift_x() * std::cos(rotation) -
         image_height * height * options.shift_y() * std::sin(rotation)) /
        image_width;
    const float y_shift =
        (image_width * width * options.shift_x() * std::sin(rotation) +
         image_height * height * options.shift_y() * std::cos(rotation)) /
        image_height;
    rect->set_x_center(rect->x_center() + x_shift);
    rect->set_y_center(rect->y_center() + y_shift);
  }

  if (options.square_long()) {
    const float long_side =
        std::max(width * image_width, height * image_height);
    width = long_side / image_width;
    height = long_side / image_height;
  } else if (options.square_short()) {
    const float short_side =
        std::min(width * image_width, height * image_height);
    width = short_side / image_width;
    height = short_side / image_height;
  }
  rect->set_width(width * options.scale_x());
  rect->set_height(height * options.scale_y());
}

} // namespace mediapipe
// Copyright 2019 The MediaPipe Authors.
// Licensed under the Apache License, Version 2.0.
#pragma once
#include <cmath>
#include "mediapipe/calculators/util/rect_transformation_calculator.pb.h"
#include "mediapipe/framework/formats/rect.pb.h"

namespace mediapipe {

float ComputeNewRotation(const mediapipe::RectTransformationCalculatorOptions& options, float rotation);
void TransformRect(const mediapipe::RectTransformationCalculatorOptions& options, mediapipe::Rect* rect);
void TransformNormalizedRect(const mediapipe::RectTransformationCalculatorOptions& options, mediapipe::NormalizedRect* rect, int image_width, int image_height);

} // namespace mediapipe
