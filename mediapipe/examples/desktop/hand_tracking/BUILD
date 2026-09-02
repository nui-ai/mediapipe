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

# Keep this parity runner graph-free. Its main target declares the liberated
# inference dependencies directly; depending on desktop_tflite_calculators here
# would rebuild the original graph calculators and their unrelated OpenCV APIs.
cc_binary(
    name = "hand_tracking_no_pipeline_run",
    data = [
        "//mediapipe/modules/hand_landmark:hand_landmark_full.tflite",
        "//mediapipe/modules/hand_landmark:handedness.txt",
        "//mediapipe/modules/palm_detection:palm_detection_full.tflite",
    ],
    deps = [
        "//mediapipe/examples/desktop:hand_tracking_no_pipeline_run_main",
    ],
)

cc_binary(
    name = "hand_tracking_pipeline_run",
    data = [
        "//mediapipe/modules/hand_landmark:hand_landmark_full.tflite",
        "//mediapipe/modules/hand_landmark:handedness.txt",
        "//mediapipe/modules/palm_detection:palm_detection_full.tflite",
    ],
    deps = [
        "//mediapipe/examples/desktop:hand_tracking_pipeline_run_main",
        "//mediapipe/graphs/hand_tracking:desktop_tflite_calculators",
    ],
)

cc_binary(
    name = "hand_tracking_pipeline_run_via_c_api",
    data = [
        "//mediapipe/modules/hand_landmark:hand_landmark_full.tflite",
        "//mediapipe/modules/hand_landmark:handedness.txt",
        "//mediapipe/modules/palm_detection:palm_detection_full.tflite",
    ],
    deps = [
        "//mediapipe/examples/desktop:hand_tracking_pipeline_run_via_c_api",
        "//mediapipe/graphs/hand_tracking:desktop_tflite_calculators",
    ],
)

cc_binary(
    name = "hand_tracking_run_via_c_api",
    data = [
        "//mediapipe/modules/hand_landmark:hand_landmark_full.tflite",
        "//mediapipe/modules/hand_landmark:handedness.txt",
        "//mediapipe/modules/palm_detection:palm_detection_full.tflite",
    ],
    deps = [
        "//mediapipe/examples/desktop:hand_tracking_run_via_c_api",
        "//mediapipe/graphs/hand_tracking:desktop_tflite_calculators",
    ],
)

# from the same c api wrapping around the C++ HandsPipelineOperatorfor pipeline driving class,
# which the above binary target also uses ―
# also build a shared library that can be loaded by rust.
# the name of the library that this rule builds will start with the standard "lib" prefix which common scenarios expect, and ends with the standard ".so" suffix.
cc_binary(
    name = "libhands_pipeline_operator_c_api.so",
    linkopts = ["-Wl,--no-as-needed"],
    linkshared = 1,
    linkstatic = False,
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/examples/desktop:hands_pipeline_operator_c_api",
        "//third_party:opencv_cmake",  # build OpenCV to provide is directly rather than entangle with the arbitrary system OpenCV version, as per https://github.com/nui-ai/mediapipe/issues/22.
    ],
)

# builds our hand tracking C API into a library that can be loaded by e.g. our rust code.
# this build rule also builds the underlying OpenCV libs still used by it internally a little.
# the name of the library file that this rule builds (on linux) will start with the standard "lib" prefix which common scenarios expect, and end with the standard ".so" linux suffix.
cc_binary(
    name = "libhand_tracking_c_api.so",
    linkopts = ["-Wl,--no-as-needed"],
    linkshared = 1,
    linkstatic = False,
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/examples/desktop:no_pipeline_c_api",
        "//third_party:opencv_cmake",  # build OpenCV to provide is directly rather than entangle with the arbitrary system OpenCV version, as per https://github.com/nui-ai/mediapipe/issues/22.
    ],
)

cc_binary(
    name = "compare_pipelines_output_data",
    deps = [
        "//mediapipe/examples/desktop:compare_pipelines_output_data_main",
        "@com_google_protobuf//:protobuf",
    ],
)

##################################################################################################
# the legacy mediapipe example mains and how to run them (no longer interesting for this branch) #
##################################################################################################

# BUILD:
# bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 \
#   //mediapipe/examples/desktop/hand_tracking:hand_tracking_tflite
# RUN:
# bazel-bin/mediapipe/examples/desktop/hand_tracking/hand_tracking_tflite \
#   --calculator_graph_config_file=<.../hand_tracking_desktop.pbtxt> \
#   --input_side_packets=input_video_path=<input video path>,output_video_path=<output video path>
cc_binary(
    name = "hand_tracking_tflite",
    data = [
        "//mediapipe/modules/hand_landmark:hand_landmark_full.tflite",
        "//mediapipe/modules/palm_detection:palm_detection_full.tflite",
    ],
    deps = [
        "//mediapipe/examples/desktop:simple_run_graph_main",
        "//mediapipe/graphs/hand_tracking:desktop_tflite_calculators",
    ],
)

# BUILD:
# bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 \
#   //mediapipe/examples/desktop/hand_tracking:hand_tracking_cpu
# RUN:
# bazel-bin/mediapipe/examples/desktop/hand_tracking/hand_tracking_cpu \
#   --calculator_graph_config_file=<.../hand_tracking_desktop_live.pbtxt> \
#   --input_video_path=<input video path> \
#   --output_video_path=<output video path>
cc_binary(
    name = "hand_tracking_cpu",
    data = [
        "//mediapipe/modules/hand_landmark:hand_landmark_full.tflite",
        "//mediapipe/modules/palm_detection:palm_detection_full.tflite",
    ],
    deps = [
        "//mediapipe/examples/desktop:demo_run_graph_main",
        "//mediapipe/graphs/hand_tracking:desktop_tflite_calculators",
    ],
)

# Linux only
cc_binary(
    name = "hand_tracking_gpu",
    data = [
        "//mediapipe/modules/hand_landmark:hand_landmark_full.tflite",
        "//mediapipe/modules/palm_detection:palm_detection_full.tflite",
    ],
    deps = [
        "//mediapipe/examples/desktop:demo_run_graph_main_gpu",
        "//mediapipe/graphs/hand_tracking:mobile_calculators",
    ],
)
