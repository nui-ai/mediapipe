# Face inference waterfall

This document delineates the graph-free face inference workflow from an input RGB frame to the face landmarks and optional pose visible to a caller. It follows each processing stage in causal order: how face detection supplies candidate regions, how landmark inference processes them, how regions of interest carry tracking across frames, where confidence gates remove candidates, how accepted landmarks can produce face geometry, and what crosses the C ABI into Rust.

It also records why the face inference API may eventually need a clean way to expose the confidence and landmarks from every landmark-model invocation, including an invocation which does not pass the tracking-confidence threshold.

## Table of contents

- [Scope and vocabulary](#scope-and-vocabulary)
- [Configuration and model setup](#configuration-and-model-setup)
- [The per-frame waterfall](#the-per-frame-waterfall)
  - [1. Choose the source regions of interest](#1-choose-the-source-regions-of-interest)
  - [2. Detect faces when tracking regions are insufficient](#2-detect-faces-when-tracking-regions-are-insufficient)
  - [3. Merge detected and tracked regions](#3-merge-detected-and-tracked-regions)
  - [4. Run landmark inference for each region](#4-run-landmark-inference-for-each-region)
  - [5. Apply the landmark presence gate](#5-apply-the-landmark-presence-gate)
  - [6. Decode accepted landmarks](#6-decode-accepted-landmarks)
  - [7. Derive the next tracking region](#7-derive-the-next-tracking-region)
  - [8. Estimate optional face pose](#8-estimate-optional-face-pose)
- [Region-of-interest tracking across frames](#region-of-interest-tracking-across-frames)
- [What crosses the C ABI and reaches Rust](#what-crosses-the-c-abi-and-reaches-rust)
- [Observable consequences of the current gate](#observable-consequences-of-the-current-gate)
- [The near-zero-threshold approximation](#the-near-zero-threshold-approximation)
- [Current performance benefits](#current-performance-benefits)
- [Possible always-visible inference output](#possible-always-visible-inference-output)
- [Implementation map](#implementation-map)

## Scope and vocabulary

The implementation contains two distinct neural-network stages:

1. The **face detector** examines a full image and proposes locations which may contain faces.
2. The **face landmark model** examines one cropped face candidate and produces a global face-presence score together with landmark tensors.

A **region of interest**, abbreviated **ROI**, is the image-normalized, rotated rectangle used to crop a possible face for the landmark model. An ROI is not itself a confirmed face. It can originate from the detector or from landmarks accepted on the preceding frame.

A **landmark attempt** in this document means one invocation of the landmark model for one ROI. The model has produced output tensors at that point, but the implementation may still reject the attempt before constructing a public face result.

An **accepted face** is a landmark attempt whose global presence score is at least `min_tracking_confidence`. Only accepted faces enter `ImageFaceTrackingResult::faces`, cross the C ABI as `FaceInferenceC`, and become Rust `FaceInference` values. One C++ face value keeps its landmarks, score, tracking rectangle, and optional pose together so those fields cannot lose index alignment.

The global landmark-model presence score controls the gate described below. MediaPipe's generic landmark representation additionally provides a per-landmark `visibility` field and a per-landmark `presence` confidence field, but the current face models emit only coordinate values for each landmark. The decoder therefore leaves those optional fields unset, and the C ABI's protobuf getters expose their unset values as zero. They must not be treated as meaningful face-confidence signals.

## Configuration and model setup

`FaceTrackingCore` is a stateful object. Its construction options select:

- `max_faces`: the desired maximum number of detector results and the number of tracked ROIs which can make a new detector pass unnecessary;
- `use_previous_landmarks`: whether accepted landmarks may produce ROIs for the following frame;
- `with_attention`: the 468-point Base landmark topology or the refined 478-point Attention topology;
- `min_detection_confidence`: the detector's candidate threshold;
- `min_tracking_confidence`: the landmark model's global face-presence threshold;
- `xnnpack_num_threads`: the interpreter worker-thread count;
- `estimate_pose`: whether accepted faces run the virtual-camera face geometry stage; and
- `vertical_fov_degrees`: the virtual camera’s vertical field of view used by that stage.

Both confidence thresholds must be finite values strictly between zero and one. They govern different stages and are not interchangeable.

Construction creates two image preprocessors and two model interpreters. The detector consumes a 128×128 tensor derived from the full image. The landmark model consumes a 192×192 tensor sampled from an ROI. Base inference produces a mesh tensor and a presence tensor. Attention inference produces mesh, lips, left-eye, right-eye, left-iris, right-iris, and presence tensors.

When `estimate_pose` is true, construction additionally reads the canonical face mesh and weighted fitting basis, configures MediaPipe’s graph-independent `GeometryPipeline`, and retains that reusable estimator. The geometry pipeline is configured once, but its screen-to-metric reconstruction and pose fit run separately for every accepted face on every processed frame. When `estimate_pose` is false, construction skips the geometry asset and the per-face geometry work. `vertical_fov_degrees` is then unused.

## The per-frame waterfall

The central `Process` operation follows this sequence:

```text
RGB image
   |
   +--> previous landmark-derived ROIs available?
   |       |
   |       +--> too few, disabled, or first frame --> run face detector
   |       |
   |       +--> enough retained ROIs -------------> detector may be skipped
   |
   +--> merge detector ROIs with retained ROIs
           |
           +--> for each ROI: run landmark model
                    |
                    +--> decode global presence score
                            |
                            +--> below threshold --> omit attempt
                            |
                            +--> accepted --------> decode landmarks
                                                       |
                                                       +--> project into image coordinates
                                                       +--> derive next-frame ROI
                                                       +--> pose enabled? reconstruct geometry and fit pose
                                                       +--> emit one cohesive public face result
```

### 1. Choose the source regions of interest

At the beginning of a frame, `Process` copies the ROIs retained from the preceding frame. The retained collection is empty for the first frame, after `Reset`, or after preceding landmark attempts failed to pass the tracking-confidence gate.

When `use_previous_landmarks` is false, the retained collection is ignored and no newly accepted ROI is stored for the next frame.

### 2. Detect faces when tracking regions are insufficient

The detector runs when either of these conditions is true:

- previous-landmark ROI reuse is disabled; or
- the number of retained ROIs is smaller than `max_faces`.

Consequently, a tracker configured for one face normally skips detection while one accepted face keeps producing a valid next-frame ROI. If that face later fails the landmark presence gate, its ROI is not retained and detection runs on the following frame.

The detector preprocesses the complete image into a square 128×128 tensor, executes the short-range face detector, decodes and filters candidates using `min_detection_confidence`, and projects the surviving detections back into input-image coordinates. It limits detector results to `max_faces`.

Each detection becomes an oriented rectangle based on the detector's face keypoints. That rectangle is expanded by a factor of 1.5 and made square, producing a detector-derived ROI suitable for the landmark model.

No landmark-model presence score or landmark tensor exists for a detector candidate which the detector itself discarded. The later idea of returning every landmark attempt therefore begins only after an ROI reaches the landmark model.

### 3. Merge detected and tracked regions

Detector-derived ROIs and retained landmark-derived ROIs can describe the same face. `Process` merges the two collections with an intersection-over-union association threshold of 0.5 so that overlapping sources do not cause duplicate landmark inference for the same region.

The result is the ROI collection used for this frame's landmark attempts. Before processing that collection, the tracker clears its saved next-frame ROIs. Only landmark attempts accepted during the current frame can repopulate that state.

### 4. Run landmark inference for each region

For each merged ROI, the image preprocessor samples the rotated face crop into a 192×192 landmark input tensor. The selected landmark model then executes and materializes all of its output tensors.

The Base model returns two outputs:

1. a 468-point mesh tensor; and
2. a global presence tensor.

The Attention model returns seven outputs:

1. a 468-point mesh tensor;
2. an 80-point lips tensor;
3. a 71-point left-eye tensor;
4. a 71-point right-eye tensor;
5. a 5-point left-iris tensor;
6. a 5-point right-iris tensor; and
7. a global presence tensor.

At this stage the expensive model invocation has already occurred. The implementation has not yet decoded the landmark tensors into `NormalizedLandmarkList` objects.

### 5. Apply the landmark presence gate

The implementation decodes the final output tensor first and applies a sigmoid to obtain a continuous global presence score. It then performs this comparison:

```cpp
if (presence_score < options_.min_tracking_confidence) {
  return std::nullopt;
}
```

A score strictly below the configured threshold rejects the attempt. A score exactly equal to the threshold is accepted.

For a rejected attempt, `InferLandmarks` returns immediately. It does not decode any landmark tensor, construct a `LandmarkInference`, calculate a landmark-derived ROI, append a public result, or retain an ROI for the next frame. The decoded score is discarded along with the remaining model outputs.

This is why the current C ABI and Rust API cannot report the score which caused an attempt to disappear.

### 6. Decode accepted landmarks

For an accepted Base attempt, the implementation decodes the 468-point mesh tensor.

For an accepted Attention attempt, it decodes all six landmark tensors. The mesh supplies the initial 468 positions. The lips and eye outputs replace the corresponding mesh X/Y coordinates, while the iris outputs add landmarks 468 through 477. Iris Z values are derived from the surrounding eye landmarks. The result is one coherent 478-point landmark list.

The decoded coordinates initially describe the 192×192 ROI crop. `ToViewportCoordinates` projects them through the ROI transform into normalized coordinates for the original input image.

### 7. Derive the next tracking region

Accepted image-space landmarks serve two roles: they are public inference output, and they define where the tracker should look on the next frame.

The implementation converts the landmark cloud to a detection-like geometry, derives an oriented face rectangle using the outer-eye landmarks at indices 33 and 263, expands the rectangle by a factor of 1.5, and makes it square. The resulting rectangle becomes `FaceInference::rect_from_landmarks` when the cohesive face result is assembled.

When `use_previous_landmarks` is enabled, that same rectangle is stored in `face_rects_from_previous_frame_`. It becomes a candidate ROI at the beginning of the next `Process` call.

### 8. Estimate optional face pose

When pose estimation is enabled, the accepted image-space landmarks enter `FaceGeometryEstimator` after the next tracking rectangle has been derived. A Base result contributes all 468 points. An Attention result contributes its final first 468 points, including refined lip and eye X/Y coordinates; its ten iris points are omitted because the configured canonical face geometry has 468 vertices.

The estimator maps screen X/Y through the configured virtual perspective camera, converts the model’s relative Z into a right-handed runtime metric reconstruction through two scale-estimation passes, and fits the canonical face to that reconstruction with MediaPipe’s weighted Procrustes basis. The fit produces a 4×4 canonical-to-runtime similarity transform containing uniform scale, proper rotation, and translation. [`FACE_LANDMARK_GEOMETRY.md`](FACE_LANDMARK_GEOMETRY.md) explains each mathematical stage and the coordinate systems in detail.

The geometry library can omit a landmark cloud whose image-space extent is too compact for a numerically stable fit. The accepted face remains in the result with no pose transform. Any other geometry error fails the complete `Process` call instead of silently emitting a face with missing pose.

## Region-of-interest tracking across frames

The ROI feedback loop lets most stable tracking frames avoid the full-image detector:

```text
detector ROI
    --> accepted landmarks
        --> landmark-derived ROI
            --> next frame's landmark attempt
                --> accepted landmarks
                    --> another next-frame ROI
```

The loop continues only while landmark presence remains at or above `min_tracking_confidence`. A rejected attempt breaks that face's feedback path because no replacement ROI is stored.

On the following frame, fewer retained ROIs are available. If their count is below `max_faces`, the detector runs and can reacquire the missing face. With `max_faces` greater than one, accepted faces can continue contributing tracked ROIs while the detector searches for enough additional candidates.

Calling `Reset` explicitly clears all retained landmark-derived ROIs. It does not change model configuration; it ensures that the next frame cannot reuse the previous tracking regions.

This stateful behavior means `min_tracking_confidence` is more than an output filter. It decides both which inference values callers observe and which regions drive future tracking.

## What crosses the C ABI and reaches Rust

The C++ `ImageFaceTrackingResult::faces` vector contains one `FaceInference` for each accepted face. Each value owns its landmark list, global presence score, landmark-derived rectangle, and optional `FacePoseTransform`. Rejected attempts occur before any value is appended to this vector.

The C conversion allocates one `FaceInferenceC` for each accepted face, deep-copies its 468 or 478 landmarks, and copies the global presence score and landmark-derived rectangle. If pose is present, it sets `has_pose_transform` and copies the 16 matrix values into inline column-major storage. The matrix acts on homogeneous column vectors and maps the canonical face into MediaPipe’s right-handed runtime metric coordinate system. When no face is accepted, `face_count` is zero and `faces` is null. The C result additionally reports whether the detector ran, but does not expose the detector's diagnostic collections.

Core's Rust wrapper deep-copies each C face into an owned `FaceInference`, converts the validity flag and matrix into `Option<FacePoseTransform>`, and releases the native result allocation. `FaceTracker::process` returns `Vec<FaceInference>`. An empty vector does not identify why no face was returned: there may have been no ROI, detection may have produced no candidate, or one or more landmark attempts may have fallen below `min_tracking_confidence`.

`FaceTrackingOptionsC` exposes `estimate_pose` and `vertical_fov_degrees`. These option and result fields form C ABI major version 2; `face_tracking_core_version()` reports `2.0.0`, and Core checks the native major before tracker creation.

## Observable consequences of the current gate

The current API has the following caller-visible contract:

- Every returned face has a global presence score greater than or equal to `min_tracking_confidence`.
- No rejected face's global presence score crosses the C ABI.
- No rejected face's landmarks cross the C ABI.
- The API does not expose how many landmark attempts were rejected.
- A caller cannot distinguish a below-threshold landmark attempt from other reasons for receiving no face.
- Rejected landmarks cannot be visualized, logged, or analyzed by a C or Rust caller.

The live Rust face application therefore renders no landmarks when its one-face result is empty. Its current confidence is unavailable for that frame. Its trailing confidence statistic can retain earlier accepted scores until they age out, but it cannot include the rejected score because that value never reached Rust.

## The near-zero-threshold approximation

Setting `min_tracking_confidence` to a very small positive value makes almost every positive presence score pass the current gate. Because construction rejects both zero and one, this is necessarily an approximation rather than an explicit “return every attempt” mode.

The approximation also changes more than result visibility. A low-scoring attempt is treated as an accepted face:

- its landmark tensors are decoded;
- its landmarks and score enter the public result;
- a new ROI is derived from those landmarks; and
- that ROI can suppress a detector run and drive landmark inference on the following frame.

Thus a near-zero threshold couples diagnostic observation to tracker admission. It may keep an unreliable or false region alive across frames, whereas the ordinary threshold would discard that region and allow detector-based reacquisition.

## Current performance benefits

Rejecting immediately after decoding the global presence score avoids real work. The savings begin after landmark-model inference, so the implementation has already paid for ROI preprocessing, TFLite/XNNPACK execution, materialization of all model output tensors, and sigmoid decoding of the one-value presence tensor.

For a rejected Base attempt, the early return avoids:

- decoding the 468-point mesh tensor;
- projecting 468 landmarks from ROI coordinates into input-image coordinates;
- constructing the detection-like landmark bounds and next-frame ROI;
- growing the C++ result collections;
- allocating and copying 468 landmarks into C ABI storage; and
- allocating and copying those landmarks again into owned Rust values.

For a rejected Attention attempt, the early return additionally avoids decoding six landmark tensors, constructing the refined 478-point topology, overlaying the lips and eyes, and assigning iris landmark Z values from the surrounding eye landmarks.

When pose estimation is enabled, rejection also avoids two screen-to-metric scale fits, perspective reconstruction, the final weighted pose fit, and pose copying through the C and Rust layers. Pose-disabled trackers incur none of that geometry work.

Omitting rejected landmarks also reduces result memory, allocator activity, and C++-to-C-to-Rust memory traffic. Returning only the rejected confidence would have much less incremental computational cost because the score is already decoded, although representing and transporting rejected attempts would still require API storage and bookkeeping.

The cross-frame performance effect is not one-directional. Rejecting an attempt removes its next-frame ROI, which can cause the more expensive full-image detector to run on the following frame. Accepting a very weak attempt might avoid that detector pass, but would spend work decoding and copying weak landmarks and could continue inference on an unreliable region. The current gate therefore combines immediate post-inference savings with the tracking-quality decision about whether an ROI remains usable.

The relative cost of tensor decoding, Attention refinement, geometry construction, ABI copying, and a possible later detector pass has not been benchmarked here. The model invocation is already complete at the gate, so the current behavior does not save landmark-model inference itself.

## Possible always-visible inference output

The face inference API may eventually need to expose the global presence score and landmarks for every ROI on which the landmark model ran, even when the score is below `min_tracking_confidence`. Such output would make threshold behavior directly observable and would permit visualization, confidence analysis, and diagnostics around rejected attempts.

In that behavior, “every” applies to landmark attempts rather than every input frame or every possible face. If no ROI reaches the landmark model, there is no landmark-model score or landmark tensor to return. Likewise, a detector candidate removed before landmark inference has no landmark output.

Supporting this behavior cleanly would require the API contract to distinguish at least two facts which the current gate combines:

1. what the landmark model produced for an ROI; and
2. whether that result passed the threshold and supplied a tracking ROI for the next frame.

The result representation would also need to state whether rejected landmarks are always decoded or controlled by an output mode, how accepted and rejected attempts are identified without confusing them with tracked faces, and how the additional values are represented and owned across the C ABI. These questions are separate from the existing tracking threshold itself.

## Implementation map

- [`mediapipe/liberated/face_tracking.h`](mediapipe/liberated/face_tracking.h) declares `FaceTrackingOptions`, `ImageFaceTrackingResult`, tracker state, and the high-level operations.
- [`mediapipe/liberated/face_tracking.cc`](mediapipe/liberated/face_tracking.cc) implements detection, ROI association, landmark inference, the presence gate, Attention refinement, optional pose invocation, and next-frame ROI retention.
- [`mediapipe/liberated/face_geometry_estimator.cc`](mediapipe/liberated/face_geometry_estimator.cc) adapts accepted Base or Attention landmarks to the 468-point GeometryPipeline and retains its pose matrix.
- [`mediapipe/tasks/cc/vision/face_geometry/libs/geometry_pipeline.cc`](mediapipe/tasks/cc/vision/face_geometry/libs/geometry_pipeline.cc) implements virtual-camera reconstruction and canonical-face fitting.
- [`mediapipe/tasks/cc/vision/face_geometry/libs/procrustes_solver.cc`](mediapipe/tasks/cc/vision/face_geometry/libs/procrustes_solver.cc) solves the weighted similarity transform.
- [`mediapipe/examples/desktop/face_tracking_c_types.h`](mediapipe/examples/desktop/face_tracking_c_types.h) declares the C configuration and result layouts.
- [`mediapipe/examples/desktop/face_tracking_c_api.cc`](mediapipe/examples/desktop/face_tracking_c_api.cc) owns the opaque native tracker and processes borrowed RGB images through the C ABI.
- [`mediapipe/examples/desktop/face_tracking_c_conversion.cc`](mediapipe/examples/desktop/face_tracking_c_conversion.cc) deep-copies accepted C++ results into C-owned allocations.
- [`../core/src/face_tracking/api.rs`](../core/src/face_tracking/api.rs) is Core's safe Rust tracker owner.
- [`../core/src/face_tracking/c_to_rust_owned.rs`](../core/src/face_tracking/c_to_rust_owned.rs) deep-copies accepted C faces and landmarks into Rust-owned values.
- [`../core/src/modalities/face.rs`](../core/src/modalities/face.rs) defines Rust's `FaceInference` and `FacePoseTransform` domain values.
