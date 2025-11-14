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
#include "mediapipe/calculators/util/detections_to_rects_calculator.h"

#include <cmath>
#include <limits>

#include "mediapipe/calculators/util/detections_to_rects_calculator.pb.h"
#include "mediapipe/calculators/util/detections_to_rects_calculator_core.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_options.pb.h"
#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/formats/location_data.pb.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"

namespace hand_tracking_mp_lean {

namespace {

constexpr char kDetectionTag[] = "DETECTION";
constexpr char kDetectionsTag[] = "DETECTIONS";
constexpr char kImageSizeTag[] = "IMAGE_SIZE";
constexpr char kRectTag[] = "RECT";
constexpr char kNormRectTag[] = "NORM_RECT";
constexpr char kRectsTag[] = "RECTS";
constexpr char kNormRectsTag[] = "NORM_RECTS";

using ::hand_tracking_mp_lean::NormalizedRect;
using ::hand_tracking_mp_lean::Rect;

constexpr float kMinFloat = std::numeric_limits<float>::lowest();
constexpr float kMaxFloat = std::numeric_limits<float>::max();

absl::Status NormRectFromKeyPoints(const LocationData& location_data,
                                   NormalizedRect* rect) {
  RET_CHECK_GT(location_data.relative_keypoints_size(), 1)
      << "2 or more key points required to calculate a rect.";
  float xmin = kMaxFloat;
  float ymin = kMaxFloat;
  float xmax = kMinFloat;
  float ymax = kMinFloat;
  for (int i = 0; i < location_data.relative_keypoints_size(); ++i) {
    const auto& kp = location_data.relative_keypoints(i);
    xmin = std::min(xmin, kp.x());
    ymin = std::min(ymin, kp.y());
    xmax = std::max(xmax, kp.x());
    ymax = std::max(ymax, kp.y());
  }
  rect->set_x_center((xmin + xmax) / 2);
  rect->set_y_center((ymin + ymax) / 2);
  rect->set_width(xmax - xmin);
  rect->set_height(ymax - ymin);
  return absl::OkStatus();
}

template <class B, class R>
void RectFromBox(B box, R* rect) {
  rect->set_x_center(box.xmin() + box.width() / 2);
  rect->set_y_center(box.ymin() + box.height() / 2);
  rect->set_width(box.width());
  rect->set_height(box.height());
}

}  // namespace

absl::Status PalmDetectionToHandRectStage1::DetectionToRect(
    const Detection& detection, const DetectionSpec& /*detection_spec*/,
    Rect* rect) {
  const LocationData location_data = detection.location_data();
  RET_CHECK(location_data.format() == LocationData::BOUNDING_BOX)
      << "Only Detection with formats of BOUNDING_BOX can be converted to Rect";
  RectFromBox(location_data.bounding_box(), rect);
  return absl::OkStatus();
}

absl::Status PalmDetectionToHandRectStage1::DetectionToNormalizedRect(
    const Detection& detection, const DetectionSpec& /*detection_spec*/,
    NormalizedRect* rect) {
  const LocationData location_data = detection.location_data();
  RET_CHECK(location_data.format() == LocationData::RELATIVE_BOUNDING_BOX)
      << "Only Detection with formats of RELATIVE_BOUNDING_BOX can be converted to NormalizedRect";
  RectFromBox(location_data.relative_bounding_box(), rect);
  return absl::OkStatus();
}

absl::Status PalmDetectionToHandRectStage1::GetContract(CalculatorContract* cc) {
  RET_CHECK(cc->Inputs().HasTag(kDetectionTag) ^
            cc->Inputs().HasTag(kDetectionsTag))
      << "Exactly one of DETECTION or DETECTIONS input stream should be "
         "provided.";
  RET_CHECK_EQ((cc->Outputs().HasTag(kNormRectTag) ? 1 : 0) +
                   (cc->Outputs().HasTag(kRectTag) ? 1 : 0) +
                   (cc->Outputs().HasTag(kNormRectsTag) ? 1 : 0) +
                   (cc->Outputs().HasTag(kRectsTag) ? 1 : 0),
               1)
      << "Exactly one of NORM_RECT, RECT, NORM_RECTS or RECTS output stream "
         "should be provided.";

  if (cc->Inputs().HasTag(kDetectionTag)) {
    cc->Inputs().Tag(kDetectionTag).Set<Detection>();
  }
  if (cc->Inputs().HasTag(kDetectionsTag)) {
    cc->Inputs().Tag(kDetectionsTag).Set<std::vector<Detection>>();
  }
  if (cc->Inputs().HasTag(kImageSizeTag)) {
    cc->Inputs().Tag(kImageSizeTag).Set<std::pair<int, int>>();
  }

  if (cc->Outputs().HasTag(kRectTag)) {
    cc->Outputs().Tag(kRectTag).Set<Rect>();
  }
  if (cc->Outputs().HasTag(kNormRectTag)) {
    cc->Outputs().Tag(kNormRectTag).Set<NormalizedRect>();
  }
  if (cc->Outputs().HasTag(kRectsTag)) {
    cc->Outputs().Tag(kRectsTag).Set<std::vector<Rect>>();
  }
  if (cc->Outputs().HasTag(kNormRectsTag)) {
    cc->Outputs().Tag(kNormRectsTag).Set<std::vector<NormalizedRect>>();
  }

  return absl::OkStatus();
}

absl::Status PalmDetectionToHandRectStage1::Open(CalculatorContext* cc) {
  cc->SetOffset(TimestampDiff(0));

  auto target_angle_rad = static_cast<float>(M_PI * 90.0 / 180.0);
  core_ = std::make_unique<DetectionsToOrientedRects>(target_angle_rad);
  return absl::OkStatus();
}

absl::Status PalmDetectionToHandRectStage1::Process(CalculatorContext* cc) {
  if (cc->Inputs().HasTag(kDetectionTag) &&
      cc->Inputs().Tag(kDetectionTag).IsEmpty()) {
    return absl::OkStatus();
  }
  if (cc->Inputs().HasTag(kDetectionsTag) &&
      cc->Inputs().Tag(kDetectionsTag).IsEmpty()) {
    return absl::OkStatus();
  }

  std::vector<Detection> detections;
  if (cc->Inputs().HasTag(kDetectionTag)) {
    detections.push_back(cc->Inputs().Tag(kDetectionTag).Get<Detection>());
  }
  if (cc->Inputs().HasTag(kDetectionsTag)) {
    detections = cc->Inputs().Tag(kDetectionsTag).Get<std::vector<Detection>>();
    if (detections.empty()) {
      return absl::OkStatus();
    }
  }
  const DetectionSpec detection_spec = GetDetectionSpec(cc);
  absl::optional<std::pair<int, int>> image_size = detection_spec.image_size;

  std::vector<Rect> rects;
  std::vector<NormalizedRect> norm_rects;
  core_->OrientedRectsFromDetections(detections, image_size, &norm_rects, &rects);

  if (cc->Outputs().HasTag(kRectTag) && !rects.empty()) {
    cc->Outputs().Tag(kRectTag).AddPacket(MakePacket<Rect>(rects[0]).At(cc->InputTimestamp()));
  }
  if (cc->Outputs().HasTag(kNormRectTag) && !norm_rects.empty()) {
    cc->Outputs().Tag(kNormRectTag).AddPacket(MakePacket<NormalizedRect>(norm_rects[0]).At(cc->InputTimestamp()));
  }
  if (cc->Outputs().HasTag(kRectsTag)) {
    auto output_rects = absl::make_unique<std::vector<Rect>>(rects);
    cc->Outputs().Tag(kRectsTag).Add(output_rects.release(), cc->InputTimestamp());
  }
  if (cc->Outputs().HasTag(kNormRectsTag)) {
    auto output_rects = absl::make_unique<std::vector<NormalizedRect>>(norm_rects);
    cc->Outputs().Tag(kNormRectsTag).Add(output_rects.release(), cc->InputTimestamp());
  }

  return absl::OkStatus();
}

DetectionSpec PalmDetectionToHandRectStage1::GetDetectionSpec(
    const CalculatorContext* cc) {
  absl::optional<std::pair<int, int>> image_size;
  if (HasTagValue(cc->Inputs(), kImageSizeTag)) {
    image_size = cc->Inputs().Tag(kImageSizeTag).Get<std::pair<int, int>>();
  }

  return {image_size};
}

REGISTER_CALCULATOR(PalmDetectionToHandRectStage1);

}  // namespace hand_tracking_mp_lean
