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

#ifndef MEDIAPIPE_TASKS_CC_VISION_FACE_GEOMETRY_LIBS_GEOMETRY_PIPELINE_H_
#define MEDIAPIPE_TASKS_CC_VISION_FACE_GEOMETRY_LIBS_GEOMETRY_PIPELINE_H_

#include <memory>
#include <vector>

#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/port/statusor.h"
#include "mediapipe/tasks/cc/vision/face_geometry/proto/environment.pb.h"
#include "mediapipe/tasks/cc/vision/face_geometry/proto/face_geometry.pb.h"
#include "mediapipe/tasks/cc/vision/face_geometry/proto/geometry_pipeline_metadata.pb.h"

namespace hand_tracking_mp_lean::tasks::vision::face_geometry {

// Reconstructs runtime facial geometry from normalized screen landmarks.
//
// The configured canonical mesh supplies the local coordinate system and scale.
// The configured virtual camera supplies the perspective projection model. For
// each input face, the pipeline reconstructs a right-handed metric landmark
// cloud, fits a canonical-to-runtime similarity transform, and returns both the
// transform and a mesh expressed in the canonical face's local coordinates.
// The estimator retains no state between calls.
class GeometryPipeline {
 public:
  virtual ~GeometryPipeline() = default;

  // Reconstructs geometry for each numerically usable input face.
  //
  // Returns an error status if any of the passed arguments is invalid.
  //
  // Each returned `FaceGeometry` contains a canonical-local metric mesh and a
  // 4x4 similarity transform which maps that mesh into the virtual camera's
  // runtime metric coordinate system. The result may omit an input face when
  // its landmarks have too little image-space extent for a stable fit. Callers
  // must therefore not assume that result indices retain input indices.
  //
  // Each face landmark list must contain the same number of points as the
  // canonical mesh supplied when this pipeline was created.
  //
  // Both `frame_width` and `frame_height` must be positive.
  virtual absl::StatusOr<std::vector<proto::FaceGeometry>> EstimateFaceGeometry(
      const std::vector<hand_tracking_mp_lean::NormalizedLandmarkList>&
          multi_face_landmarks,
      int frame_width, int frame_height) const = 0;
};

// Creates a reusable geometry estimator for one canonical mesh and camera
// configuration.
//
// `environment` must define a valid image origin and virtual perspective
// camera. `metadata` must define the input landmark source, canonical face mesh,
// and weighted Procrustes fitting basis. The canonical mesh must contain the
// `POSITION` and `TEX_COORD` vertex components.
absl::StatusOr<std::unique_ptr<GeometryPipeline>> CreateGeometryPipeline(
    const proto::Environment& environment,
    const proto::GeometryPipelineMetadata& metadata);

}  // namespace hand_tracking_mp_lean::tasks::vision::face_geometry

#endif  // MEDIAPIPE_TASKS_CC_VISION_FACE_GEOMETRY_LIBS_GEOMETRY_PIPELINE_H_
