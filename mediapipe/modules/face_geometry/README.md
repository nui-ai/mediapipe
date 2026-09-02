# Face geometry

Face geometry reconstructs a right-handed metric landmark cloud from normalized
screen landmarks and a virtual perspective camera. It then fits the canonical
face to that cloud with a weighted similarity transform. The resulting pose
matrix maps canonical-local points into runtime metric space; the accompanying
mesh retains the face-specific shape in canonical-local coordinates.

This directory contains MediaPipe's legacy calculator and subgraph integration.
The graph-free face tracker in this fork uses the corresponding Tasks library at
`mediapipe/tasks/cc/vision/face_geometry/libs` through
`mediapipe/liberated/face_geometry_estimator.cc`. See
[`FACE_LANDMARK_GEOMETRY.md`](../../../FACE_LANDMARK_GEOMETRY.md) for the
coordinate semantics, reconstruction steps, Procrustes fit, and current C ABI.

Protos|Details
:--- | :---
[`face_geometry.Environment`](https://github.com/google-ai-edge/mediapipe/tree/master/mediapipe/modules/face_geometry/protos/environment.proto)| Describes an environment; includes the camera frame origin point location as well as virtual camera parameters.
[`face_geometry.GeometryPipelineMetadata`](https://github.com/google-ai-edge/mediapipe/tree/master/mediapipe/modules/face_geometry/protos/geometry_pipeline_metadata.proto)| Carries the canonical face mesh, input landmark source, and weighted Procrustes fitting basis.
[`face_geometry.FaceGeometry`](https://github.com/google-ai-edge/mediapipe/tree/master/mediapipe/modules/face_geometry/protos/face_geometry.proto)| Describes 3D transform data for a single face; includes a face mesh surface and a face pose in a given environment.
[`face_geometry.Mesh3d`](https://github.com/google-ai-edge/mediapipe/tree/master/mediapipe/modules/face_geometry/protos/mesh_3d.proto)| Describes a 3D mesh triangular surface.

Calculators|Details
:--- | :---
[`FaceGeometryEnvGeneratorCalculator`](https://github.com/google-ai-edge/mediapipe/tree/master/mediapipe/modules/face_geometry/env_generator_calculator.cc)| Generates an environment that describes a virtual scene.
[`FaceGeometryPipelineCalculator`](https://github.com/google-ai-edge/mediapipe/tree/master/mediapipe/modules/face_geometry/geometry_pipeline_calculator.cc)| Extracts face 3D transform for multiple faces from a vector of landmark lists.
[`FaceGeometryEffectRendererCalculator`](https://github.com/google-ai-edge/mediapipe/tree/master/mediapipe/modules/face_geometry/effect_renderer_calculator.cc)| Renders a face effect.

Subgraphs|Details
:--- | :---
[`FaceGeometryFromDetection`](https://github.com/google-ai-edge/mediapipe/tree/master/mediapipe/modules/face_geometry/face_geometry_from_detection.pbtxt)| Extracts 3D transform from face detection for multiple faces.
[`FaceGeometryFromLandmarks`](https://github.com/google-ai-edge/mediapipe/tree/master/mediapipe/modules/face_geometry/face_geometry_from_landmarks.pbtxt)| Extracts 3D transform from face landmarks for multiple faces.
[`FaceGeometry`](https://github.com/google-ai-edge/mediapipe/tree/master/mediapipe/modules/face_geometry/face_geometry.pbtxt)| Extracts 3D transform from face landmarks for multiple faces. Deprecated, please use `FaceGeometryFromLandmarks` in the new code.
