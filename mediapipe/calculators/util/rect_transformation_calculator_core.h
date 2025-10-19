#pragma once
#include "mediapipe/calculators/util/rect_transformation_calculator.pb.h"
#include "mediapipe/framework/formats/rect.pb.h"

namespace mediapipe {

class RectTransformationCalculatorCore {

public:

  explicit RectTransformationCalculatorCore(RectTransformationCalculatorOptions &options);
  void TransformRect(Rect *rect) const;
  void TransformNormalizedRect(NormalizedRect *rect, int image_width, int image_height) const;

 private:
  static float NormalizeRadians(float angle);
  float ComputeNewRotation(float rotation) const;
  RectTransformationCalculatorOptions options_;
};

} // namespace mediapipe