// Copyright 2025 The MediaPipe Authors.
// Licensed under the Apache License, Version 2.0.

#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "absl/log/absl_log.h"
#include <cmath>
#include <functional>
#include <glog/log_severity.h>

namespace mediapipe::api3 {

///
/// helper function returning a rotation function which takes a landmark and rotates it the same
/// as the rotation which the rectangle object containing it has (if rect is provided, otherwise returns nullptr).
/// this function is then applied to each landmark contained in the rectangle.
///
/// the input rectangle is just the one used for passing the sub-image of the entire frame which was passed to the landmarks
/// inference model or the reverse of it, so that its predictions can be rotated back to match the overall input image.
///
/// note that this means that the inference inside the landmarks inference model leaves no work nor any external
/// parameterization (unless it has any specific input tensor positions for it) for controlling its inference
/// of its 3D object description which it calls world landmarks ― it is all self-contained in it and the only
/// thing we post-process here is to rotate the landmark coordinates back by the rotation that was applied
/// to the rectangle passed to the inference model.
///
/// (not even TensorsToWorldLandmarksCalculator performs any transformation!).
///
/// thus, coming up with the hand's 3D shape is all self-contained in the landmarks inference model!
/// we give it a rectangle, it returns the hand 3D shape, and we then only rotate it to match the rotation
/// of the rectangle which we gave the model as input.
///
/// so there's no real projection after the inference ― which aligns with calling it "an orthographic projection"
/// at the bottom line:
///
/// the model doesn't know where the box was in space, it just inferences object coords based on what it sees,
/// and we never try to perspective project what it gives out ― not in the original pipeline (nor yet elsewhere).
///
/// we don't know how it scales the hand ― it probably does not scale it by the bounding box size at all,
/// since it always receives the same dimension of its input images as per its model card ― indeed empirically
/// we know that its scaling of the hand in the world landmarks output is practically arbitrary and should
/// be just rescaled from physical units assigned by it to unit-less proportions: it conveys shape (angles)
/// and relative lengths (as much as they are all accurate) and a rotation of the entire set of predicted
/// landmarks is just implied by the square image passed to the model.
///
std::function<void(const Landmark&, Landmark*)> CreateRotationFunction(const mediapipe::NormalizedRect* rect) {
  const float cosa = std::cos(rect->rotation());
  const float sina = std::sin(rect->rotation());
  return [cosa, sina](const Landmark& in_landmark, Landmark* out_landmark) {
    out_landmark->set_x(cosa * in_landmark.x() - sina * in_landmark.y());
    out_landmark->set_y(sina * in_landmark.x() + cosa * in_landmark.y());
  };
}

/// applies the given rectangle's rotation, to each landmark, that's all
LandmarkList Process(const LandmarkList& in_landmarks, const NormalizedRect *hand_rect) {

  auto landmark_rotate = CreateRotationFunction(hand_rect);

  mediapipe::LandmarkList out_landmarks;
  for (int i = 0; i < in_landmarks.landmark_size(); ++i) {
    const auto& in_landmark = in_landmarks.landmark(i);
    Landmark* out_landmark = out_landmarks.add_landmark();
    *out_landmark = in_landmark;
    if (landmark_rotate) {
      landmark_rotate(in_landmark, out_landmark);
    }
  }
  return out_landmarks;
}

} // namespace mediapipe::api3
