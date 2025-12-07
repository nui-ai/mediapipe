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
    const ImageHandTrackingResult& result,
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
  return out;
}

void FillPipelineOutputDataFromCResult(const HandTrackingResultC* c_result, PipelineOutputData* proto, int frame_number);

} // namespace hand_tracking_mp_lean
