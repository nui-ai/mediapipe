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

#include "mediapipe/framework/deps/ret_check.h"

namespace hand_tracking_mp_lean {

hand_tracking_mp_lean::StatusBuilder RetCheckFailSlowPath(
    hand_tracking_mp_lean::source_location location) {
  // TODO Implement LogWithStackTrace().
  return hand_tracking_mp_lean::InternalErrorBuilder(location)
         << "RET_CHECK failure (" << location.file_name() << ":"
         << location.line() << ") ";
}

hand_tracking_mp_lean::StatusBuilder RetCheckFailSlowPath(
    hand_tracking_mp_lean::source_location location, const char* condition) {
  return hand_tracking_mp_lean::RetCheckFailSlowPath(location) << condition;
}

hand_tracking_mp_lean::StatusBuilder RetCheckFailSlowPath(
    hand_tracking_mp_lean::source_location location, const char* condition,
    const absl::Status& status) {
  return hand_tracking_mp_lean::RetCheckFailSlowPath(location)
         << condition << " returned " << status << " ";
}

}  // namespace hand_tracking_mp_lean
