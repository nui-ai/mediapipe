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

namespace mediapipe_v01013_based {

mediapipe_v01013_based::StatusBuilder RetCheckFailSlowPath(
    mediapipe_v01013_based::source_location location) {
  // TODO Implement LogWithStackTrace().
  return mediapipe_v01013_based::InternalErrorBuilder(location)
         << "RET_CHECK failure (" << location.file_name() << ":"
         << location.line() << ") ";
}

mediapipe_v01013_based::StatusBuilder RetCheckFailSlowPath(
    mediapipe_v01013_based::source_location location, const char* condition) {
  return mediapipe_v01013_based::RetCheckFailSlowPath(location) << condition;
}

mediapipe_v01013_based::StatusBuilder RetCheckFailSlowPath(
    mediapipe_v01013_based::source_location location, const char* condition,
    const absl::Status& status) {
  return mediapipe_v01013_based::RetCheckFailSlowPath(location)
         << condition << " returned " << status << " ";
}

}  // namespace mediapipe_v01013_based
