#pragma once
#include "mediapipe/calculators/util/rect_transformation_calculator.pb.h"
#include "mediapipe/framework/formats/rect.pb.h"

namespace hand_tracking_mp_lean {

class RectTransformation {

public:

  explicit RectTransformation(RectTransformationCalculatorOptions &options);
  void Expand(Rect *rect) const;
  void ExpandNormalizedRect(NormalizedRect *rect, int image_width, int image_height) const;

 private:
  static float NormalizeRadians(float angle);
  float ComputeNewRotation(float rotation) const;
  RectTransformationCalculatorOptions options_;
};

} // namespace hand_tracking_mp_lean