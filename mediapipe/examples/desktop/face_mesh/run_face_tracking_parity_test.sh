#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  run_face_tracking_parity_test.sh --input-video PATH [options]

Builds the original FaceLandmarkFrontCpu graph in an isolated temporary
buildable-reference worktree, runs it and FaceTrackingCore on the same video,
then compares every X/Y/Z landmark coordinate.

Options:
  --input-video PATH          Required input video.
  --reference-ref REF         Git ref for the legacy graph
                              (default: origin/buildable-reference).
  --base                      Compare the base 468-landmark model.
  --attention                 Compare the 478-landmark attention model
                              (default).
  --static-image-mode         Run detection independently on every frame.
  --max-num-faces N           Maximum faces (default: 1).
  --absolute-tolerance VALUE  Comparator tolerance (default: 0).
  --compilation-mode MODE     Bazel compilation mode (default: opt).
  --keep-temp                 Keep the patched reference worktree and outputs.
  -h, --help                  Show this help.
USAGE
}

input_video=""
reference_ref="origin/buildable-reference"
refine_landmarks=true
static_image_mode=false
max_num_faces=1
absolute_tolerance=0
compilation_mode="opt"
keep_temp=false

while (( $# > 0 )); do
  case "$1" in
    --input-video)
      [[ $# -ge 2 ]] || { echo "--input-video requires a value" >&2; exit 2; }
      input_video="$2"
      shift 2
      ;;
    --reference-ref)
      [[ $# -ge 2 ]] || { echo "--reference-ref requires a value" >&2; exit 2; }
      reference_ref="$2"
      shift 2
      ;;
    --base)
      refine_landmarks=false
      shift
      ;;
    --attention)
      refine_landmarks=true
      shift
      ;;
    --static-image-mode)
      static_image_mode=true
      shift
      ;;
    --max-num-faces)
      [[ $# -ge 2 ]] || { echo "--max-num-faces requires a value" >&2; exit 2; }
      max_num_faces="$2"
      shift 2
      ;;
    --absolute-tolerance)
      [[ $# -ge 2 ]] || { echo "--absolute-tolerance requires a value" >&2; exit 2; }
      absolute_tolerance="$2"
      shift 2
      ;;
    --compilation-mode)
      [[ $# -ge 2 ]] || { echo "--compilation-mode requires a value" >&2; exit 2; }
      compilation_mode="$2"
      shift 2
      ;;
    --keep-temp)
      keep_temp=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

[[ -n "$input_video" ]] || { echo "--input-video is required" >&2; exit 2; }
[[ "$max_num_faces" =~ ^[1-9][0-9]*$ ]] ||
  { echo "--max-num-faces must be a positive integer" >&2; exit 2; }
[[ "$compilation_mode" == "fastbuild" ||
   "$compilation_mode" == "dbg" ||
   "$compilation_mode" == "opt" ]] ||
  { echo "--compilation-mode must be fastbuild, dbg, or opt" >&2; exit 2; }

input_video="$(realpath "$input_video")"
[[ -f "$input_video" ]] || { echo "Input video does not exist: $input_video" >&2; exit 2; }

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(git -C "$script_dir" rev-parse --show-toplevel)"
reference_patch="$script_dir/buildable_reference_face_runner.patch"
[[ -f "$reference_patch" ]] ||
  { echo "Missing reference runner patch: $reference_patch" >&2; exit 2; }
git -C "$repo_root" rev-parse --verify "${reference_ref}^{commit}" >/dev/null

temp_root="$(mktemp -d /tmp/mediapipe-face-parity.XXXXXX)"
reference_worktree="$temp_root/buildable-reference"
reference_output="$temp_root/reference.pb"
candidate_output="$temp_root/candidate.pb"
worktree_added=false

cleanup() {
  if [[ "$keep_temp" == true ]]; then
    echo "Kept parity workspace: $temp_root"
    return
  fi
  if [[ "$worktree_added" == true ]]; then
    git -C "$repo_root" worktree remove --force "$reference_worktree" >/dev/null 2>&1 || true
  fi
  if [[ "$temp_root" == /tmp/mediapipe-face-parity.* ]]; then
    rm -rf -- "$temp_root"
  fi
}
trap cleanup EXIT

git -C "$repo_root" worktree add --detach "$reference_worktree" "$reference_ref"
worktree_added=true
git -C "$reference_worktree" apply "$reference_patch"

bazel_args=(
  -c "$compilation_mode"
  --define MEDIAPIPE_DISABLE_GPU=1
)
reference_bazel_args=("${bazel_args[@]}")
if [[ -d /usr/include/opencv4 ]]; then
  reference_bazel_args+=(--cxxopt=-I/usr/include/opencv4)
fi

reference_target="//mediapipe/examples/desktop/face_parity:reference_face_tracking_run"
candidate_target="//mediapipe/examples/desktop/face_mesh:face_tracking_no_pipeline_run"
comparator_target="//mediapipe/examples/desktop/face_mesh:compare_face_tracking_output_data"

(
  cd "$reference_worktree"
  bazel build "${reference_bazel_args[@]}" "$reference_target"
  bazel run "${reference_bazel_args[@]}" "$reference_target" -- \
    --input_video_path="$input_video" \
    --output_data_path="$reference_output" \
    --max_num_faces="$max_num_faces" \
    --refine_landmarks="$refine_landmarks" \
    --static_image_mode="$static_image_mode"
)

(
  cd "$repo_root"
  bazel build "${bazel_args[@]}" "$candidate_target" "$comparator_target"
  bazel run "${bazel_args[@]}" "$candidate_target" -- \
    --input_video_path="$input_video" \
    --output_data_path="$candidate_output" \
    --max_num_faces="$max_num_faces" \
    --refine_landmarks="$refine_landmarks" \
    --static_image_mode="$static_image_mode"
  bazel run "${bazel_args[@]}" "$comparator_target" -- \
    --reference_data_path="$reference_output" \
    --candidate_data_path="$candidate_output" \
    --absolute_tolerance="$absolute_tolerance"
)
