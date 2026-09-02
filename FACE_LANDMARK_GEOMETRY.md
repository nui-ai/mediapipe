# Face landmark geometry and its interpretation

The graph-free face tracker returns a dense, useful description of facial shape, but its coordinates span several different kinds of geometry with different guarantees. This document explains what the landmark model predicts, how it learns relative depth, how MediaPipe derives metric geometry and pose, and which results the liberated implementation carries through its C ABI to downstream users.

## Table of contents

- [What the face landmark model returns](#what-the-face-landmark-model-returns)
- [How the model learns relative Z](#how-the-model-learns-relative-z)
- [How the liberated implementation transforms the output](#how-the-liberated-implementation-transforms-the-output)
- [How reliably three-dimensional are the landmarks?](#how-reliably-three-dimensional-are-the-landmarks)
- [The surrounding geometry](#the-surrounding-geometry)
  - [Tracking rectangles](#tracking-rectangles)
  - [Crop transformation matrices](#crop-transformation-matrices)
  - [Fixed topology and the canonical face](#fixed-topology-and-the-canonical-face)
  - [Metric face reconstruction](#metric-face-reconstruction)
  - [The facial pose transformation matrix](#the-facial-pose-transformation-matrix)
- [Attention landmarks and the iris](#attention-landmarks-and-the-iris)
- [Iris-based camera distance](#iris-based-camera-distance)
- [Blendshape coefficients](#blendshape-coefficients)
- [What currently crosses the ABI](#what-currently-crosses-the-abi)
- [Current pose integration and possible later outputs](#current-pose-integration-and-possible-later-outputs)
- [Practical interpretation](#practical-interpretation)
- [Sources](#sources)

## What the face landmark model returns

The Base face-landmark model receives a normalized 192 by 192 crop containing one candidate face. It directly regresses 468 ordered points, with three floating-point values per point, and a separate face-wide presence value. Each landmark index has a stable semantic and topological meaning across frames and subjects.

The Attention variant returns the same 468-point mesh together with specialized tensors for the lips, both eyes, and both irises. The implementation combines these into 478 points. The specialized lip and eye tensors improve localization around those regions, while the ten additional points describe the two irises.

The public screen-landmark coordinates have these semantics:

- X is horizontal position normalized by the input image width.
- Y is vertical position normalized by the input image height.
- Z is learned relative depth, scaled approximately like normalized X.
- Smaller Z values indicate positions closer to the camera.
- Depth near the center of the head acts approximately as the Z origin.

Calling all three values “normalized coordinates” can be misleading. X and Y are directly tied to the viewport. Z is not a normalized camera-space distance and is not constrained to the same interval as X and Y. Landmarks may also lie outside the nominal image interval when part of a face extends beyond the frame.

The generic MediaPipe landmark type contains optional per-landmark `visibility` and `presence` fields. These face models emit only XYZ values per point, so those optional fields remain unset. The C ABI currently reads unset protobuf values as zero. They should not be treated as landmark confidence estimates. The separate face-wide presence score is the meaningful confidence signal in this pipeline.

## How the model learns relative Z

The landmark model learns Z as a direct regression target during training. It does not triangulate points, consult a depth sensor, or run a geometric depth formula during inference.

MediaPipe trained the face model with multiple objectives. Synthetic rendered faces supplied known three-dimensional vertex coordinates because the renderer knew the source mesh and camera pose. Annotated real photographs supplied two-dimensional semantic-contour supervision where dependable dense 3D ground truth was unavailable. A shared network learned from both, allowing synthetic 3D supervision to transfer to real facial appearance while real images improved localization and robustness outside the synthetic visual domain.

The training process was also iteratively bootstrapped with increasingly difficult cases, including oblique views, expressions, and occlusions. Accurate face cropping removes much of the translation, scale, and in-plane rotation variation before inference, leaving more model capacity for surface-coordinate prediction.

From a single crop, the network can associate visual patterns with depth-related structure:

- yaw and pitch create characteristic foreshortening;
- the nose, cheek, jaw, and eye contours move predictably with pose;
- near-side features occlude far-side features;
- facial symmetry constrains plausible shape;
- shading and perspective provide weak additional cues; and
- human facial anatomy occupies a comparatively narrow statistical family.

Nevertheless, monocular depth is fundamentally ambiguous. Different faces and depths can produce nearly the same pixels. Where the image cannot determine a unique shape, the network predicts a statistically plausible result based on its learned face prior. Relative Z is therefore a mixture of image evidence, inferred head pose, and the expected anatomy represented by the training data.

The model’s output coordinate system deliberately preserves only relative depth. It is suited to describing coarse surface shape and orientation, but contains no physical unit, calibrated camera distance, or independently measured personal anatomy.

## How the liberated implementation transforms the output

The current `FaceTrackingCore` runs either `face_landmark.tflite` or `face_landmark_with_attention.tflite` over each accepted face region. The raw mesh tensor contains coordinates in the 192 by 192 model-input coordinate system.

Decoding divides raw X by 192, raw Y by 192, and raw Z by 192. The subsequent viewport projection:

1. recenters X and Y around the middle of the crop;
2. rotates them by the face region’s in-plane rotation;
3. scales them by the face region’s width and height;
4. translates them to the face region’s center in the original image; and
5. scales Z by the face region’s width, following the weak-perspective convention that Z uses the same scale as X.

This processing relocates the predicted surface from crop coordinates into input-image coordinates. It does not add new depth information. In particular, the rectangle’s image position does not become a camera-space translation, and its apparent size does not by itself become an absolute distance.

For Attention results, the implementation first copies the Base mesh XYZ values. Specialized lip and eye predictions replace X and Y at their corresponding indices but leave the Base mesh Z values in place. Each iris obtains specialized X and Y values, while its Z is assigned from the average Z of surrounding eye landmarks. Attention therefore improves fine two-dimensional placement much more than it improves depth.

## How reliably three-dimensional are the landmarks?

The most accurate description is **screen-anchored 2.5D geometry**.

X and Y are usually the strongest outputs because they are directly constrained by visible image features and real-image annotation. They can still jitter, drift under occlusion, or follow an incorrect facial feature when the crop is poor, but they are intended to locate actual pixels.

Z is useful for:

- coarse front-to-back ordering;
- estimating head orientation;
- producing a plausible face surface for rendering;
- measuring relative motion within a stable capture; and
- helping distinguish some expression changes from rigid head motion.

Z is less suitable for:

- physical distance measurement;
- millimetre-accurate facial dimensions;
- medical or biometric surface reconstruction;
- comparing absolute facial size across unrelated cameras or subjects; and
- treating every local depth deformation as observed rather than inferred.

There is also no independent confidence value for each Z coordinate. The face-wide presence score says whether the model considers the crop to contain a reasonably aligned face; it does not quantify the accuracy of individual vertices or their depth.

The later Face Geometry pipeline makes the landmarks more convenient and internally consistent for 3D applications, but it cannot create missing image evidence. Its results inherit the monocular model’s uncertainty and add assumptions about the camera and canonical face.

## The surrounding geometry

Several geometric objects participate in the pipeline. Some describe image preprocessing, some describe tracking, and some describe a reconstructed 3D scene. Their similar matrix and rectangle vocabulary should not obscure their different meanings.

### Tracking rectangles

The face detector and landmark tracker use rotated, image-normalized rectangles. A rectangle contains a center, width, height, and counter-clockwise in-plane rotation.

A detector-derived rectangle begins with the detector’s face box and keypoints, is oriented using a keypoint pair, enlarged by a factor of 1.5, and made square for landmark inference. After successful landmark inference, a new rectangle is derived from the landmark cloud, oriented using landmarks 33 and 263 around the eyes, enlarged, and retained as the likely search region for the next frame.

The liberated C ABI exposes this latter value as `rect_from_landmarks`. It is chiefly a future tracking crop. It is neither a tight semantic face bound nor a three-dimensional head pose.

### Crop transformation matrices

Image preprocessing also calculates a 4 by 4 rotated-subrectangle transformation matrix. It maps normalized coordinates inside a sampled crop back into normalized coordinates in the source image. It composes crop centering, scale, in-plane rotation, optional horizontal reflection, translation, and a convention that scales Z like X.

The detector path uses this matrix to project detections out of its tensor and into the input image. The landmark path retains a corresponding matrix in its preprocessing result but currently projects landmarks using the face rectangle directly.

This matrix can be useful for exact crop replay, visualization, model debugging, or downstream processing that consumes the same tensor. It is not a face-pose matrix and contains no information about the physical camera or subject distance.

There is an important representation hazard if such matrices cross an ABI. The image preprocessing matrix is stored as a row-major array, while MediaPipe `MatrixData` serialization defaults to column-major order. Any public matrix field must define storage order, vector convention, transform direction, and coordinate spaces explicitly.

### Fixed topology and the canonical face

Landmark indices imply a fixed facial topology. MediaPipe provides named connection sets for the lips, eyes, eyebrows, irises, nose, face oval, and dense tessellation. These connections are static data rather than per-frame inference results.

MediaPipe also ships a canonical 468-vertex face mesh. It supplies:

- canonical XYZ positions expressed in centimetres;
- UV texture coordinates;
- triangular topology; and
- a weighted landmark subset used for pose fitting.

The canonical mesh is a reference human face, not a measurement of the current subject. It gives a stable local coordinate system, a standard scale, and a surface onto which applications can model glasses, masks, textures, and other assets.

Topology, UVs, and canonical positions should generally be published once as a versioned asset or constant table. Repeating them in every frame result would waste ABI bandwidth and create unnecessary ownership complexity.

### Metric face reconstruction

MediaPipe’s Face Geometry library converts screen landmarks into a right-handed metric coordinate system configured by a virtual perspective camera. Its default environment uses a top-left image origin, a vertical field of view of 63 degrees, a near plane of 1 centimetre, and a far plane of 100 metres.

The reconstruction proceeds as follows:

1. The vertical field of view and near distance define the virtual camera’s near-plane height. The input frame’s aspect ratio defines its width.
2. Map normalized screen X and Y onto that near-plane rectangle. Scale relative Z like X, preserving the landmark model’s weak-perspective convention.
3. Fit the canonical face directly to this projected cloud. The fit’s uniform scale provides an initial conversion for relative Z without first pretending that Z is an absolute camera-space depth.
4. Subtract the landmark cloud’s mean Z offset, add the virtual near distance, and divide that complete depth expression by the initial scale. Perspective-unproject each near-plane X/Y point along its camera ray to the resulting provisional depth.
5. Fit the canonical face to this provisional metric cloud. The resulting second scale is the multiplicative correction exposed by the first perspective reconstruction.
6. Repeat the Z conversion and perspective unprojection with the product of the first and second scales. This produces the final right-handed runtime metric landmark cloud.
7. Solve a weighted Procrustes problem from the canonical landmarks to the runtime metric landmarks. The static geometry asset identifies which canonical vertices participate in the fitting basis and assigns each selected vertex its influence weight.
8. Apply the inverse fitted transform to the runtime cloud. This expresses the returned mesh in the canonical face’s local metric frame; applying the separately returned forward transform places it back in runtime metric space.

The Procrustes fit minimizes the weighted squared distances between corresponding canonical and runtime points under one uniform scale, one proper rotation, and one translation. Internally, square-root weights turn that objective into ordinary matrix least squares. Weighted centering removes translation, a singular-value decomposition of the cross-covariance supplies the best rotation without reflection, scalar projection supplies scale, and the weighted mean residual supplies translation.

The resulting mesh contains 468 XYZ vertex positions with the canonical UV coordinates and triangle indices. Applying its pose transform and the matching virtual-camera projection reproduces the original screen-space X and Y positions.

The word “metric” describes the chosen coordinate system and its canonical centimetre scale. It does not guarantee that the reconstructed subject has the true physical size of the canonical model. Translation and scale accuracy depend on the assumed field of view, the weak-perspective depth predictions, landmark accuracy, and how closely the subject’s face resembles the canonical size and proportions.

Supplying a vertical field of view close to that of the real camera improves the interpretation. The existing environment is still simpler than a full camera calibration: it does not expose an arbitrary focal matrix, principal point, skew, or lens distortion model.

### The facial pose transformation matrix

The public Face Landmarker API calls this optional output the `facial_transformation_matrixes`; internally `FaceGeometry` calls it the `pose_transform_matrix`. “Pose” is the standard name here for the fitted global placement of the canonical face. The transform contains scale and translation as well as orientation, so it is more specifically a similarity transform rather than a rotation-only pose.

It is a 4 by 4 canonical-to-runtime similarity transform containing:

- one uniform scale;
- a proper rotation without reflection;
- a translation; and
- a final homogeneous row of `[0, 0, 0, 1]`.

The weighted Procrustes basis favors facial regions intended to distinguish rigid head-pose change from expression deformation. The transform is computed downstream from the screen landmarks and canonical face; it is not another hidden neural-network tensor.

The matrix is useful for aligning a canonical-space asset with the detected face, extracting head orientation, and following relative translation and scale. Its rotation is often practically valuable for AR even when physical scale is imperfect. Translation and absolute scale should be described as model-and-prior-derived estimates unless the entire camera and subject setup has been independently calibrated.

The full `FaceGeometry` object contains both this matrix and the metric mesh. The higher-level Face Landmarker task keeps only the matrix in its user-facing result, although the internal geometry object contains the mesh as well.

## Attention landmarks and the iris

The current graph-free implementation includes only the landmark-localization part of MediaPipe’s iris functionality.

When `with_attention` is false, which is the default, the tracker uses the Base model and returns 468 landmarks. When it is true, the tracker uses the Attention model, decodes both iris tensors, and returns 478 landmarks through the existing C ABI. The two groups at indices 468 through 477 each contain an iris center and four boundary points.

The current implementation does not calculate iris diameter, camera distance, eye orientation, or gaze direction. It does not instantiate `IrisToDepthCalculator` or a graph-free equivalent. Its iris Z values are derived from neighboring eye landmarks rather than independently inferred.

The standard Tasks face-geometry path also discards iris points before reconstruction and operates on the first 468 landmarks. A legacy Face Geometry asset variant extends the canonical mesh with interpolated iris vertices and UVs, but the current Tasks graph does not use it. For most applications it is cleaner to keep the 468-point face surface and the ten specialized iris points as related but distinct geometry.

## Iris-based camera distance

MediaPipe’s separate iris-depth calculator estimates the distance from each iris to the camera using the apparent iris diameter and camera focal length. Near the image center, its core relationship is approximately:

```text
distance_mm = 11.8 mm * focal_length_pixels / iris_diameter_pixels
```

The implementation includes an off-axis correction based on the iris center’s distance from the image center. It averages vertical and horizontal iris diameters, produces separate left- and right-eye distances, and applies exponential smoothing with 90 percent of the preceding value and 10 percent of the new estimate.

This is physically more grounded than the mesh’s relative Z because it introduces a known-size cue. Its limitations are correspondingly explicit:

- the camera focal length in pixels must be known accurately;
- the calculation assumes an average human iris diameter of 11.8 millimetres;
- the iris boundary must be visible and localized accurately;
- perspective, eyelid occlusion, blur, and glare can distort the apparent diameter; and
- the result is per-eye camera distance, not a complete measured face surface.

The liberated implementation already returns the required iris points when Attention is enabled, so the unsmoothed calculation could be added as a small graph-independent helper. Temporal smoothing needs more care for multiple faces because face indices are not persistent identities across frames.

## Blendshape coefficients

The full Face Landmarker task can optionally run a separate blendshape model. It selects 146 landmarks from the 478-point result, converts their X and Y coordinates to a tensor, and predicts 52 named coefficients such as eye blink, jaw open, brow motion, smiling, cheek puff, and lip motion.

Blendshapes are semantic expression controls rather than additional geometric measurements. They are useful for avatar animation and puppeteering, but they do not improve the mesh’s Z accuracy. Adding them to the liberated implementation would require carrying the separate blendshape model and its inference step.

## What currently crosses the ABI

For each accepted face, the current C ABI exposes:

- 468 Base or 478 Attention screen landmarks;
- the face-wide presence score;
- the landmark-derived rectangle intended for tracking; and
- an optional canonical-to-runtime face pose transform when pose estimation is configured and the fit is numerically stable.

At the top level it also reports whether the face detector ran on that frame.

`FaceTrackingOptionsC::estimate_pose` controls the extra work. When it is zero, tracker construction does not load the canonical mesh and weighted fitting basis or create a geometry estimator, and every face reports `has_pose_transform == 0`. When it is nonzero, `vertical_fov_degrees` configures the virtual perspective camera. Tracker construction requires a finite value strictly between 0 and 180 degrees. Frame width and height supply the camera aspect ratio for every inference call.

`FaceInferenceC::pose_transform` contains 16 inline `float` values. The values use column-major storage, the matrix acts on homogeneous column vectors, and it maps the canonical face into MediaPipe’s right-handed runtime metric space. `has_pose_transform` states whether those values may be read. A face can pass the landmark presence threshold yet receive no pose when its projected landmark cloud is too compact for a stable geometry fit. Other geometry errors fail the frame instead of silently removing pose.

Core’s Rust wrapper converts that validity flag and matrix into `Option<FacePoseTransform>`. The Rust name emphasizes the matrix’s caller-visible role. Its documentation and `canonical_to_runtime_matrix` accessor state the exact transform direction and representation; a longer type name such as `CanonicalToRuntimeTransform` is unnecessary.

The C++ result internally retains detector results and detector-derived rectangles, but the C conversion does not expose those diagnostic collections. It also does not expose crop matrices, metric mesh vertices, topology, UVs, blendshapes, iris distance, or the complete geometry environment.

The returned faces have no persistent identity across frames. This limitation matters for any downstream temporal filter, including pose smoothing and iris-distance smoothing.

The pose fields changed the C layouts in ABI major version 2. `face_tracking_core_version()` reports `2.0.0`, and Core verifies the major version before creating a tracker. This repository changed the structures in place because the graph-free face ABI had no external consumers requiring compatibility with its earlier experimental layout. Future incompatible layout changes should increment the major version so that existing consumers fail explicitly.

## Current pose integration and possible later outputs

`FaceTrackingCore` creates one `FaceGeometryEstimator` during tracker construction when `estimate_pose` is enabled. Creation reads the canonical mesh and the weighted Procrustes basis once, then creates MediaPipe’s reusable graph-independent `GeometryPipeline`. The pipeline still performs the screen-to-metric reconstruction and weighted pose fit for every accepted face on every frame; only virtual-camera configuration and canonical mesh and fitting-basis loading are one-time work.

Base inference supplies all 468 landmarks. Attention inference supplies the final refined first 468 landmarks and omits its ten iris points from geometry calculation because the configured canonical mesh has 468 vertices. This matches the standard Tasks geometry path: refined lip and eye coordinates participate in the fit, while iris-only points do not.

MediaPipe’s full `FaceGeometry` result contains the pose matrix and a 468-point canonical-local metric mesh. The graph-free adapter deliberately retains only the matrix: pose was the required caller-facing result, while copying the full mesh through C storage on every frame would add data and ownership surface without a current consumer. The following outputs remain possible later additions:

- the canonical-local metric mesh;
- optional left and right iris distances in millimetres;
- optional blendshape coefficients when the extra model is configured; and
- a topology or canonical-model identifier if callers need to correlate results with static geometry assets.

The topology and UV data should remain static. An ABI can expose a topology identifier and provide a separate accessor or bundled asset for the canonical mesh. This also gives callers a way to detect future landmark-layout or canonical-model revisions.

It would be helpful to describe coordinate spaces in types or field names rather than relying on comments alone. At minimum, the API needs to distinguish:

- normalized screen landmarks with weak-perspective relative Z;
- crop-to-screen preprocessing transforms;
- canonical-local metric mesh coordinates; and
- the runtime metric coordinate frame configured by the virtual camera.

Changing the size of an existing public C structure can break binary consumers. Any later incompatible additions must therefore accompany an ABI-major increment, or use an opaque accessor or feature-negotiated extension whose binary contract permits growth.

## Practical interpretation

For overlaying two-dimensional graphics or measuring motion in the input frame, use X and Y directly.

For coarse surface rendering, relative deformation, and orientation-sensitive effects, use the XYZ landmarks while preserving their weak-perspective semantics.

For attaching canonical three-dimensional assets and estimating head pose, use the Face Geometry pose matrix and its canonical mesh. Treat rotation as generally more trustworthy than absolute scale or translation.

For physically meaningful eye-to-camera distance, use the iris-diameter method with real camera focal length and explicit uncertainty. Do not substitute the mesh’s relative Z.

For animation semantics, use blendshapes rather than trying to infer every expression coefficient manually from local landmark distances.

None of these outputs should be described as a direct depth-camera measurement. Together they form a strong real-time monocular face representation: pixel-anchored landmarks, learned relative shape, a canonical prior, and camera-conditioned geometric reconstruction.

## Sources

### Repository implementation and contracts

- [`mediapipe/liberated/face_tracking.h`](mediapipe/liberated/face_tracking.h) — current graph-free options and C++ result shape.
- [`mediapipe/liberated/face_tracking.cc`](mediapipe/liberated/face_tracking.cc) — Base and Attention inference, landmark refinement, iris Z construction, tracking rectangles, and result assembly.
- [`mediapipe/liberated/face_geometry_estimator.cc`](mediapipe/liberated/face_geometry_estimator.cc) — graph-free adapter from accepted landmarks to the Tasks GeometryPipeline pose result.
- [`mediapipe/examples/desktop/face_tracking_c_types.h`](mediapipe/examples/desktop/face_tracking_c_types.h) — current public C types.
- [`mediapipe/examples/desktop/face_tracking_c_conversion.cc`](mediapipe/examples/desktop/face_tracking_c_conversion.cc) — data currently copied through the C ABI.
- [`../core/src/modalities/face.rs`](../core/src/modalities/face.rs) — Rust `FaceInference` and `FacePoseTransform` domain values.
- [`mediapipe/calculators/tensor/tensors_to_landmarks_calculator.proto`](mediapipe/calculators/tensor/tensors_to_landmarks_calculator.proto) — weak-perspective Z normalization contract.
- [`mediapipe/calculators/tensor/tensors_to_landmarks_calculator_core.cc`](mediapipe/calculators/tensor/tensors_to_landmarks_calculator_core.cc) — raw XYZ tensor decoding and normalization.
- [`mediapipe/calculators/util/landmark_projection_calculator_core.cc`](mediapipe/calculators/util/landmark_projection_calculator_core.cc) — crop-to-viewport landmark projection and Z scaling.
- [`mediapipe/calculators/tensor/image_to_tensor_utils.cc`](mediapipe/calculators/tensor/image_to_tensor_utils.cc) — rotated crop transformation matrix construction.
- [`mediapipe/tasks/cc/vision/face_landmarker/face_landmarks_connections.h`](mediapipe/tasks/cc/vision/face_landmarker/face_landmarks_connections.h) — fixed contour and tessellation connections.
- [`mediapipe/tasks/cc/vision/face_geometry/proto/geometry_pipeline_metadata.proto`](mediapipe/tasks/cc/vision/face_geometry/proto/geometry_pipeline_metadata.proto) — canonical mesh units and weighted Procrustes basis contract.
- [`mediapipe/tasks/cc/vision/face_geometry/proto/face_geometry.proto`](mediapipe/tasks/cc/vision/face_geometry/proto/face_geometry.proto) — metric mesh and pose-matrix semantics.
- [`mediapipe/tasks/cc/vision/face_geometry/proto/environment.proto`](mediapipe/tasks/cc/vision/face_geometry/proto/environment.proto) — virtual-camera environment.
- [`mediapipe/tasks/cc/vision/face_geometry/libs/geometry_pipeline.cc`](mediapipe/tasks/cc/vision/face_geometry/libs/geometry_pipeline.cc) — screen-to-metric conversion, scale estimation, and geometry packing.
- [`mediapipe/tasks/cc/vision/face_geometry/libs/geometry_pipeline_metadata_loader.cc`](mediapipe/tasks/cc/vision/face_geometry/libs/geometry_pipeline_metadata_loader.cc) — shared loading of the binary canonical mesh, topology, UV coordinates, input-source mode, and fitting basis.
- [`mediapipe/tasks/cc/vision/face_geometry/libs/procrustes_solver.cc`](mediapipe/tasks/cc/vision/face_geometry/libs/procrustes_solver.cc) — weighted similarity-transform solution.
- [`mediapipe/tasks/cc/vision/face_geometry/face_geometry_from_landmarks_graph.cc`](mediapipe/tasks/cc/vision/face_geometry/face_geometry_from_landmarks_graph.cc) — default camera parameters and removal of iris landmarks before geometry estimation.
- [`mediapipe/tasks/cc/vision/face_landmarker/face_landmarker_graph.cc`](mediapipe/tasks/cc/vision/face_landmarker/face_landmarker_graph.cc) — optional Face Geometry integration in the Tasks graph.
- [`mediapipe/tasks/cc/vision/face_landmarker/face_blendshapes_graph.cc`](mediapipe/tasks/cc/vision/face_landmarker/face_blendshapes_graph.cc) — 52-coefficient blendshape inference from landmark X/Y values.
- [`mediapipe/graphs/iris_tracking/calculators/iris_to_depth_calculator.cc`](mediapipe/graphs/iris_tracking/calculators/iris_to_depth_calculator.cc) — iris-diameter camera-distance calculation and smoothing.
- [`mediapipe/tasks/cc/vision/face_geometry/data/geometry_pipeline_metadata_landmarks.pbtxt`](mediapipe/tasks/cc/vision/face_geometry/data/geometry_pipeline_metadata_landmarks.pbtxt) — canonical mesh data and landmark weights.
- [`mediapipe/tasks/cc/vision/face_geometry/data/canonical_face_model.obj`](mediapipe/tasks/cc/vision/face_geometry/data/canonical_face_model.obj) — reference canonical face asset.

### Primary and official background material

- [Real-time Facial Surface Geometry from Monocular Video on Mobile GPUs](https://research.google/pubs/real-time-facial-surface-geometry-from-monocular-video-on-mobile-gpus/) — the original MediaPipe Face Mesh paper and model overview.
- [MediaPipe Face Mesh documentation](https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/face_mesh.md) — training objectives, relative-Z semantics, Attention Mesh, and Face Geometry concepts.
- [Attention Mesh: High-fidelity Face Mesh Prediction in Real-time](https://arxiv.org/abs/2006.10962) — specialized high-resolution attention for lips, eyes, and irises.
- [MediaPipe Iris: Real-time Iris Tracking and Depth Estimation](https://research.google/blog/mediapipe-iris-real-time-iris-tracking-depth-estimation/) — iris landmarking and monocular camera-distance estimation.
