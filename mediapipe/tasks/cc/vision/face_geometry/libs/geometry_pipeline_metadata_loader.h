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

#ifndef MEDIAPIPE_TASKS_CC_VISION_FACE_GEOMETRY_LIBS_GEOMETRY_PIPELINE_METADATA_LOADER_H_
#define MEDIAPIPE_TASKS_CC_VISION_FACE_GEOMETRY_LIBS_GEOMETRY_PIPELINE_METADATA_LOADER_H_

#include <string>

#include "mediapipe/framework/port/statusor.h"
#include "mediapipe/tasks/cc/core/proto/external_file.pb.h"
#include "mediapipe/tasks/cc/vision/face_geometry/proto/geometry_pipeline_metadata.pb.h"

namespace hand_tracking_mp_lean::tasks::vision::face_geometry {

// Reads the static inputs used to construct a face GeometryPipeline.
//
// The binary proto contains the canonical mesh positions, UVs, and topology,
// together with the landmark indices and weights used by the Procrustes fit.
absl::StatusOr<proto::GeometryPipelineMetadata> ReadGeometryPipelineMetadata(
    const core::proto::ExternalFile& metadata_file);

// Reads the same canonical geometry and fitting data from one runtime asset
// path.
absl::StatusOr<proto::GeometryPipelineMetadata> ReadGeometryPipelineMetadata(
    const std::string& metadata_path);

}  // namespace hand_tracking_mp_lean::tasks::vision::face_geometry

#endif  // MEDIAPIPE_TASKS_CC_VISION_FACE_GEOMETRY_LIBS_GEOMETRY_PIPELINE_METADATA_LOADER_H_
