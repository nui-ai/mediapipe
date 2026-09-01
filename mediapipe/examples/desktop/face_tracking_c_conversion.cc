#include "mediapipe/examples/desktop/face_tracking_c_conversion.h"

#include <cstdlib>
#include <string>

namespace {

void CopyRect(const hand_tracking_mp_lean::NormalizedRect& source,
              FaceRectC* destination) {
  destination->x_center = source.x_center();
  destination->y_center = source.y_center();
  destination->width = source.width();
  destination->height = source.height();
  destination->rotation = source.rotation();
}

}  // namespace

int ConvertFaceTrackingResultToC(
    const hand_tracking_mp_lean::ImageFaceTrackingResult& cpp_result,
    FaceTrackingResultC* c_result,
    void (*set_last_error)(const std::string& error)) {
  if (c_result == nullptr) {
    if (set_last_error != nullptr) {
      set_last_error("null FaceTrackingResultC output");
    }
    return -1;
  }

  const size_t face_count = cpp_result.face_landmarks.size();
  if (cpp_result.face_presence_scores.size() != face_count ||
      cpp_result.face_rects_from_landmarks.size() != face_count) {
    if (set_last_error != nullptr) {
      set_last_error(
          "face landmarks, presence scores, and landmark-derived rectangles "
          "do not have matching counts");
    }
    return -1;
  }

  c_result->face_detector_ran = cpp_result.face_detector_ran ? 1 : 0;
  c_result->face_count = face_count;
  if (face_count == 0) {
    c_result->faces = nullptr;
    return 0;
  }

  c_result->faces = static_cast<FaceInferenceC*>(
      std::calloc(face_count, sizeof(FaceInferenceC)));
  if (c_result->faces == nullptr) {
    if (set_last_error != nullptr) {
      set_last_error("failed to allocate C face inference array");
    }
    return -1;
  }

  for (size_t face_index = 0; face_index < face_count; ++face_index) {
    const auto& source_landmarks = cpp_result.face_landmarks[face_index];
    const int landmark_count = source_landmarks.landmark_size();
    if (landmark_count != 468 && landmark_count != 478) {
      if (set_last_error != nullptr) {
        set_last_error(
            "face landmark list must contain either 468 or 478 landmarks");
      }
      return -1;
    }

    FaceInferenceC* destination = &c_result->faces[face_index];
    destination->landmark_count = static_cast<size_t>(landmark_count);
    destination->landmarks = static_cast<FaceLandmarkC*>(
        std::calloc(destination->landmark_count, sizeof(FaceLandmarkC)));
    if (destination->landmarks == nullptr) {
      if (set_last_error != nullptr) {
        set_last_error("failed to allocate C face landmark array");
      }
      return -1;
    }

    for (int landmark_index = 0; landmark_index < landmark_count;
         ++landmark_index) {
      const auto& source = source_landmarks.landmark(landmark_index);
      FaceLandmarkC* target = &destination->landmarks[landmark_index];
      target->x = source.x();
      target->y = source.y();
      target->z = source.z();
      target->visibility = source.visibility();
      target->presence = source.presence();
    }
    destination->presence_score =
        cpp_result.face_presence_scores[face_index];
    CopyRect(cpp_result.face_rects_from_landmarks[face_index],
             &destination->rect_from_landmarks);
  }
  return 0;
}
