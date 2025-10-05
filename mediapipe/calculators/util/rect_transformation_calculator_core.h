#pragma once
#include "mediapipe/calculators/util/rect_transformation_calculator.pb.h"
#include "mediapipe/framework/formats/rect.pb.h"

namespace mediapipe {

float ComputeNewRotation(const RectTransformationCalculatorOptions& options, float rotation);
void TransformRect(const RectTransformationCalculatorOptions& options, Rect* rect);
void TransformNormalizedRect(const RectTransformationCalculatorOptions& options, NormalizedRect* rect, int image_width, int image_height);

} // namespace mediapipe
