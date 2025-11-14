/* Copyright 2023 The MediaPipe Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Utility methods for populating Ghum Hand joints from joints produced by Hund
Hand model and hand landmarks.
==============================================================================*/

#ifndef MEDIAPIPE_TASKS_CC_VISION_UTILS_GHUM_GHUM_HAND_UTILS_H_
#define MEDIAPIPE_TASKS_CC_VISION_UTILS_GHUM_GHUM_HAND_UTILS_H_

#include "mediapipe/framework/api2/builder.h"
#include "mediapipe/framework/formats/body_rig.pb.h"
#include "mediapipe/framework/formats/landmark.pb.h"

namespace hand_tracking_mp_lean::tasks::vision::utils::ghum {

// Sets visibility of 16 GHUM hand joints from 21 hand world landmarks.
hand_tracking_mp_lean::api2::builder::Stream<hand_tracking_mp_lean::JointList>
SetGhumHandJointsVisibilityFromWorldLandmarks(
    hand_tracking_mp_lean::api2::builder::Stream<hand_tracking_mp_lean::JointList> ghum_hand_joints,
    hand_tracking_mp_lean::api2::builder::Stream<hand_tracking_mp_lean::LandmarkList>
        hand_world_landmarks,
    hand_tracking_mp_lean::api2::builder::Graph& graph);

}  // namespace hand_tracking_mp_lean::tasks::vision::utils::ghum

#endif  // MEDIAPIPE_TASKS_CC_VISION_UTILS_GHUM_GHUM_HAND_UTILS_H_
