# Repeatable face-landmark parity test

Run the original `buildable-reference` graph and the graph-free
`FaceTrackingCore` against the same video with:

```bash
mediapipe/examples/desktop/face_mesh/run_face_tracking_parity_test.sh \
  --input-video /absolute/path/to/video.mp4
```

The default test uses the 478-landmark attention/iris model, stateful ROI
tracking, one face, optimized builds, and exact coordinate equality. Use
`--base` for the 468-landmark model or `--help` for all options.

The script creates a detached temporary worktree at
`origin/buildable-reference`, applies the checked-in reference-runner patch,
builds and runs both implementations, invokes the face-specific numerical
comparator, and removes the worktree and generated protobufs. Pass
`--keep-temp` to preserve them for debugging.

The supplied video is never copied into the repository. Bazel's normal caches
remain available across runs, so subsequent reference builds reuse compiled
actions.
