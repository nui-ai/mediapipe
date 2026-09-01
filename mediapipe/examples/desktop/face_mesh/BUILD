# Copyright 2019 The MediaPipe Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

licenses(["notice"])

# CalculatorGraph-free CPU face landmarks/mesh runner. This is a face-only
# artifact: it links neither the hand core nor hand model assets.
cc_binary(
    name = "face_tracking_no_pipeline_run",
    data = [
        "//mediapipe/modules/face_detection:face_detection_short_range.tflite",
        "//mediapipe/modules/face_landmark:face_landmark.tflite",
        "//mediapipe/modules/face_landmark:face_landmark_with_attention.tflite",
    ],
    deps = ["//mediapipe/examples/desktop:face_tracking_no_pipeline_run_main"],
)

# Face-only shared library consumed through the strict-C facade. Model files
# remain external runtime inputs selected by the facade's models_path.
cc_binary(
    name = "libface_tracking_c_api.so",
    data = [
        "//mediapipe/modules/face_detection:face_detection_short_range.tflite",
        "//mediapipe/modules/face_landmark:face_landmark.tflite",
        "//mediapipe/modules/face_landmark:face_landmark_with_attention.tflite",
    ],
    linkopts = ["-Wl,--no-as-needed"],
    linkshared = 1,
    linkstatic = False,
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/examples/desktop:face_tracking_c_api",
        "//third_party:opencv_cmake",
    ],
)

cc_binary(
    name = "compare_face_tracking_output_data",
    deps = [
        "//mediapipe/examples/desktop:compare_face_tracking_output_data_main",
    ],
)

# BUILD:
# bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 \
#   //mediapipe/examples/desktop/face_mesh:face_mesh_tflite
# RUN:
# bazel-bin/mediapipe/examples/desktop/face_mesh/face_mesh_tflite \
#   --calculator_graph_config_file=<.../face_mesh_desktop.pbtxt> \
#   --input_side_packets=input_video_path=<input video path>,output_video_path=<output video path>
cc_binary(
    name = "face_mesh_tflite",
    data = ["//mediapipe/modules/face_landmark:face_landmark_with_attention.tflite"],
    deps = [
        "//mediapipe/examples/desktop:simple_run_graph_main",
        "//mediapipe/graphs/face_mesh:desktop_calculators",
    ],
)

# BUILD:
# bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 \
#   //mediapipe/examples/desktop/face_mesh:face_mesh_cpu
# RUN:
# bazel-bin/mediapipe/examples/desktop/face_mesh/face_mesh_cpu \
#   --calculator_graph_config_file=<.../face_mesh_desktop_live.pbtxt> \
#   --input_video_path=<input video path> \
#   --output_video_path=<output video path>
cc_binary(
    name = "face_mesh_cpu",
    data = ["//mediapipe/modules/face_landmark:face_landmark_with_attention.tflite"],
    deps = [
        "//mediapipe/examples/desktop:demo_run_graph_main",
        "//mediapipe/graphs/face_mesh:desktop_live_calculators",
    ],
)

# Linux only
cc_binary(
    name = "face_mesh_gpu",
    data = ["//mediapipe/modules/face_landmark:face_landmark_with_attention.tflite"],
    deps = [
        "//mediapipe/examples/desktop:demo_run_graph_main_gpu",
        "//mediapipe/graphs/face_mesh:desktop_live_gpu_calculators",
    ],
)
