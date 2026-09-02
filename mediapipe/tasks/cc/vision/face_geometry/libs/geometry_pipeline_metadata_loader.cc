// Copyright 2023 The MediaPipe Authors.
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

#include "mediapipe/tasks/cc/vision/face_geometry/libs/geometry_pipeline_metadata_loader.h"

#include <string>

#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status_macros.h"
#include "mediapipe/tasks/cc/core/external_file_handler.h"

namespace hand_tracking_mp_lean::tasks::vision::face_geometry {

absl::StatusOr<proto::GeometryPipelineMetadata> ReadGeometryPipelineMetadata(
    const core::proto::ExternalFile& metadata_file) {
  MP_ASSIGN_OR_RETURN(
      const auto file_handler,
      core::ExternalFileHandler::CreateFromExternalFile(&metadata_file));

  proto::GeometryPipelineMetadata metadata;
  RET_CHECK(metadata.ParseFromString(
      std::string(file_handler->GetFileContent())))
      << "Failed to parse canonical face and fitting data from the binary proto!";
  return metadata;
}

absl::StatusOr<proto::GeometryPipelineMetadata> ReadGeometryPipelineMetadata(
    const std::string& metadata_path) {
  core::proto::ExternalFile metadata_file;
  metadata_file.set_file_name(metadata_path);
  return ReadGeometryPipelineMetadata(metadata_file);
}

}  // namespace hand_tracking_mp_lean::tasks::vision::face_geometry
