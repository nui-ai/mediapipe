#pragma once

#include "hand_tracking_c_types.h"
#include "mediapipe/examples/desktop/pipeline_output.pb.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/classification.pb.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/liberated/hand_tracking.h"

namespace hand_tracking_mp_lean {

// Utility to build PipelineOutputData from HandTrackingCore::Process result
inline PipelineOutputData BuildPipelineOutputDataFromProcessResult(
    const ImageHandTrackingAndInferenceResult& result,
    int frame_number) {
  PipelineOutputData out;
  out.set_frame_number(frame_number);
  // World landmarks
  for (const auto& world_landmarks : *result.object_landmarkss) {
    *out.add_multi_hand_world_landmarks() = world_landmarks;
  }
  // Viewport landmarks
  for (const auto& viewport_landmarks : *result.viewport_landmarkss) {
    *out.add_multi_hand_landmarks() = viewport_landmarks;
  }
  // Handedness classifications
  for (const auto& handedness : *result.handedness_classifications) {
    *out.add_multi_handedness() = handedness;
  }
  // Palm detection rects
  if (result.detection_details) {
    for (const auto& det : *result.detection_details) {
      NormalizedRect rect;
      rect.set_x_center(det.expanded.x_center);
      rect.set_y_center(det.expanded.y_center);
      rect.set_width(det.expanded.width);
      rect.set_height(det.expanded.height);
      rect.set_rotation(det.expanded.rotation);
      *out.add_hand_rects_from_palm_detections() = rect;
    }
  }
  return out;
}

void FillPipelineOutputDataFromCResult(const HandTrackingResultC* c_result, PipelineOutputData* proto, int frame_number);

} // namespace hand_tracking_mp_lean
