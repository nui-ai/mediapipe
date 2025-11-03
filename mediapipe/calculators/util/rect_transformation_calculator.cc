// Copyright 2019 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
#include <cmath>
#include <memory>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "mediapipe/calculators/util/rect_transformation_calculator.pb.h"
#include "mediapipe/calculators/util/rect_transformation_calculator_core.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_options.pb.h"
#include "mediapipe/framework/formats/rect.pb.h"

namespace mediapipe_v01013_based {

namespace {

constexpr char kNormRectTag[] = "NORM_RECT";
constexpr char kNormRectsTag[] = "NORM_RECTS";
constexpr char kRectTag[] = "RECT";
constexpr char kRectsTag[] = "RECTS";
constexpr char kImageSizeTag[] = "IMAGE_SIZE";

using ::mediapipe_v01013_based::NormalizedRect;
using ::mediapipe_v01013_based::Rect;

// Wraps around an angle in radians to within -M_PI and M_PI.
inline float NormalizeRadians(float angle) {
  return angle - 2 * M_PI * std::floor((angle - (-M_PI)) / (2 * M_PI));
}

}  // namespace

// Performs geometric transformation to the input Rect or NormalizedRect,
// corresponding to input stream RECT or NORM_RECT respectively. When the input
// is NORM_RECT, an addition input stream IMAGE_SIZE is required, which is a
// std::pair<int, int> representing the image width and height.
//
// the purpose of transformation is typically (i.e. in our pipeline) to either
// expand a bounding box from its palm detection nature to one that is likely
// to capture the entire hand, or to expand it in a way that it would still
// capture the entire same hand if the hand moved a little on the next frame.
// as a baseline it works well and forms a good baseline in enabling the
// overall hand tracking.
//
// the expansion performed elongates the original bounding box in one direction before applying the scaling.

class RectTransformationCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc);

  absl::Status Open(CalculatorContext* cc) override;
  absl::Status Process(CalculatorContext* cc) override;

 private:
  RectTransformationCalculatorOptions options_;
  std::unique_ptr<PalmRectToHandRect> core_;
};

class PalmDetectionToHandRectStage2 : public RectTransformationCalculator {};  // used as part of detection handling
class DeriveAnticipatoryHandRect : public RectTransformationCalculator {}; // used outside detection handling
REGISTER_CALCULATOR(PalmDetectionToHandRectStage2);
REGISTER_CALCULATOR(DeriveAnticipatoryHandRect);

absl::Status RectTransformationCalculator::GetContract(CalculatorContract* cc) {
  RET_CHECK_EQ((cc->Inputs().HasTag(kNormRectTag) ? 1 : 0) +
                   (cc->Inputs().HasTag(kNormRectsTag) ? 1 : 0) +
                   (cc->Inputs().HasTag(kRectTag) ? 1 : 0) +
                   (cc->Inputs().HasTag(kRectsTag) ? 1 : 0),
               1);
  if (cc->Inputs().HasTag(kRectTag)) {
    cc->Inputs().Tag(kRectTag).Set<Rect>();
    cc->Outputs().Index(0).Set<Rect>();
  }
  if (cc->Inputs().HasTag(kRectsTag)) {
    cc->Inputs().Tag(kRectsTag).Set<std::vector<Rect>>();
    cc->Outputs().Index(0).Set<std::vector<Rect>>();
  }
  if (cc->Inputs().HasTag(kNormRectTag)) {
    RET_CHECK(cc->Inputs().HasTag(kImageSizeTag));
    cc->Inputs().Tag(kNormRectTag).Set<NormalizedRect>();
    cc->Inputs().Tag(kImageSizeTag).Set<std::pair<int, int>>();
    cc->Outputs().Index(0).Set<NormalizedRect>();
  }
  if (cc->Inputs().HasTag(kNormRectsTag)) {
    RET_CHECK(cc->Inputs().HasTag(kImageSizeTag));
    cc->Inputs().Tag(kNormRectsTag).Set<std::vector<NormalizedRect>>();
    cc->Inputs().Tag(kImageSizeTag).Set<std::pair<int, int>>();
    cc->Outputs().Index(0).Set<std::vector<NormalizedRect>>();
  }

  return absl::OkStatus();
}

absl::Status RectTransformationCalculator::Open(CalculatorContext* cc) {
  cc->SetOffset(TimestampDiff(0));

  options_ = cc->Options<RectTransformationCalculatorOptions>();
  RET_CHECK(!(options_.has_rotation() && options_.has_rotation_degrees()));
  RET_CHECK(!(options_.has_square_long() && options_.has_square_short()));

  ABSL_LOG(INFO) << "RectTransformationCalculator options: " << options_.DebugString();

  core_ = std::make_unique<PalmRectToHandRect>(options_);

  return absl::OkStatus();
}

absl::Status RectTransformationCalculator::Process(CalculatorContext* cc) {
  if (cc->Inputs().HasTag(kRectTag) && !cc->Inputs().Tag(kRectTag).IsEmpty()) {
    auto rect = cc->Inputs().Tag(kRectTag).Get<Rect>();
    core_->Expand(&rect);
    cc->Outputs().Index(0).AddPacket(
        MakePacket<Rect>(rect).At(cc->InputTimestamp()));
  }
  if (cc->Inputs().HasTag(kRectsTag) &&
      !cc->Inputs().Tag(kRectsTag).IsEmpty()) {
    auto rects = cc->Inputs().Tag(kRectsTag).Get<std::vector<Rect>>();
    auto output_rects = absl::make_unique<std::vector<Rect>>(rects.size());
    for (int i = 0; i < rects.size(); ++i) {
      output_rects->at(i) = rects[i];
      auto it = output_rects->begin() + i;
      core_->Expand(&(*it));
    }
    cc->Outputs().Index(0).Add(output_rects.release(), cc->InputTimestamp());
  }
  if (HasTagValue(cc->Inputs(), kNormRectTag) &&
      HasTagValue(cc->Inputs(), kImageSizeTag)) {
    auto rect = cc->Inputs().Tag(kNormRectTag).Get<NormalizedRect>();
    const auto& image_size =
        cc->Inputs().Tag(kImageSizeTag).Get<std::pair<int, int>>();
    core_->ExpandNormalizedRect(&rect, image_size.first, image_size.second);
    cc->Outputs().Index(0).AddPacket(
        MakePacket<NormalizedRect>(rect).At(cc->InputTimestamp()));
  }
  if (HasTagValue(cc->Inputs(), kNormRectsTag) &&
      HasTagValue(cc->Inputs(), kImageSizeTag)) {
    auto rects =
        cc->Inputs().Tag(kNormRectsTag).Get<std::vector<NormalizedRect>>();
    const auto& image_size =
        cc->Inputs().Tag(kImageSizeTag).Get<std::pair<int, int>>();
    auto output_rects =
        absl::make_unique<std::vector<NormalizedRect>>(rects.size());
    for (int i = 0; i < rects.size(); ++i) {
      output_rects->at(i) = rects[i];
      auto it = output_rects->begin() + i;
      core_->ExpandNormalizedRect(&(*it), image_size.first, image_size.second);
    }
    cc->Outputs().Index(0).Add(output_rects.release(), cc->InputTimestamp());
  }

  return absl::OkStatus();
}

}  // namespace mediapipe_v01013_based
