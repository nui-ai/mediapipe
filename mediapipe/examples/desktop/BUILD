load("//mediapipe/framework/port:build_config.bzl", "mediapipe_cc_proto_library", "mediapipe_proto_library")

licenses(["notice"])

package(default_visibility = [
    "//visibility:public",
])

mediapipe_proto_library(
    name = "pipeline_output_proto",
    srcs = ["pipeline_output.proto"],
    deps = [
        "//mediapipe/framework/formats:classification_proto",
        "//mediapipe/framework/formats:landmark_proto",
        "//mediapipe/framework/formats:rect_proto",
    ],
)

# stuffs many calculators as direct dependencies, in order to make them "alwayslink",
# for the sake of the shared library that we build which depends on this target.
# we could have added them in that lib target directly, but feels cleaner here.
# not sure why we needed this for those ones defined by mediapipe_simple_subgraph
# in particular, but since we removed many source trees for unnecessary platforms,
# simple use of bazel query "somepath(//foo:bar, //dep:lib)" for detailing dependency
# graphs for making an elegant short tethering path isn't working on this branch,
# so a bit brute-force here rather than elegantly tethering the most minimal
# alwayslink path fo those extra direct depenedencies here.
cc_library(
    name = "hands_pipeline_operator",
    srcs = ["hands_pipeline_operator.cc"],
    hdrs = ["hands_pipeline_operator.h"],
    deps = [
        ":pipeline_output_cc_proto",
        "//mediapipe/calculators/core:begin_loop_calculator",
        "//mediapipe/calculators/core:clip_vector_size_calculator",
        "//mediapipe/calculators/core:constant_side_packet_calculator",
        "//mediapipe/calculators/core:end_loop_calculator",
        "//mediapipe/calculators/core:flow_limiter_calculator",
        "//mediapipe/calculators/core:gate_calculator",
        "//mediapipe/calculators/core:previous_loopback_calculator",
        "//mediapipe/calculators/image:image_properties_calculator",
        "//mediapipe/calculators/tensor:detection_inference_calculator",
        "//mediapipe/calculators/util:association_norm_rect_calculator",
        "//mediapipe/calculators/util:collection_has_min_size_calculator",
        "//mediapipe/calculators/util:detection_model_provider",
        "//mediapipe/calculators/util:dummy_calculator",
        "//mediapipe/calculators/util:filter_collection_calculator",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/formats:detection_cc_proto",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/formats:image_frame_opencv",
        "//mediapipe/framework/port:file_helpers",
        "//mediapipe/framework/port:opencv_highgui",
        "//mediapipe/framework/port:opencv_imgproc",
        "//mediapipe/framework/port:opencv_video",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/framework/port:status",
        "//mediapipe/modules/hand_landmark:hand_landmark_cpu",
        "//mediapipe/modules/hand_landmark:hand_landmark_landmarks_to_roi",
        "//mediapipe/modules/hand_landmark:palm_detection_detection_to_roi",
        "//mediapipe/modules/palm_detection:palm_detection_cpu",
        "//mediapipe/util:resource_util",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/log:absl_log",
    ],
    alwayslink = 1,
)

cc_library(
    name = "hand_tracking_pipeline_run_main",
    srcs = ["hand_tracking_pipeline_run_main.cc"],
    data = ["//mediapipe/modules/hand_landmark:handedness.txt"],  # needed when running the binary through the bazel command
    deps = [
        ":hands_pipeline_operator",
        ":pipeline_output_cc_proto",
        "//mediapipe/calculators/util:detection_model_provider",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/formats:detection_cc_proto",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/formats:image_frame_opencv",
        "//mediapipe/framework/port:file_helpers",
        "//mediapipe/framework/port:opencv_highgui",
        "//mediapipe/framework/port:opencv_imgproc",
        "//mediapipe/framework/port:opencv_video",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/framework/port:status",
        "//mediapipe/util:resource_util",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/log:absl_log",
    ],
)

cc_library(
    name = "hand_tracking_pipeline_run_via_c_api",
    srcs = ["hand_tracking_pipeline_run_main_via_c_api.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":hands_pipeline_operator_c_api",
        ":pipeline_output_cc_proto",
        "//mediapipe/calculators/util:detection_model_provider",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/formats:detection_cc_proto",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/formats:image_frame_opencv",
        "//mediapipe/framework/port:file_helpers",
        "//mediapipe/framework/port:opencv_highgui",
        "//mediapipe/framework/port:opencv_imgproc",
        "//mediapipe/framework/port:opencv_video",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/framework/port:status",
        "//mediapipe/util:resource_util",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/log:absl_log",
    ],
)

cc_library(
    name = "hands_pipeline_operator_c_api",
    srcs = ["hands_pipeline_operator_c_api.cc"],
    hdrs = ["hands_pipeline_operator_c_api.h"],
    linkstatic = True,
    visibility = ["//visibility:public"],
    deps = [
        ":hands_pipeline_operator",
        ":pipeline_output_cc_proto",
    ],
    alwayslink = 1,
)

cc_library(
    name = "compare_pipelines_output_data_main",
    srcs = ["compare_pipelines_output_data_main.cc"],
    deps = [
        ":pipeline_output_cc_proto",
        "@com_google_protobuf//:protobuf",
    ],
)

####################################
# original mediapipe example mains #
####################################

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

cc_library(
    name = "simple_run_graph_main",
    srcs = ["simple_run_graph_main.cc"],
    deps = [
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/port:file_helpers",
        "//mediapipe/framework/port:map_util",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
        "//mediapipe/framework/port:statusor",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "demo_run_graph_main",
    srcs = ["demo_run_graph_main.cc"],
    deps = [
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/formats:image_frame_opencv",
        "//mediapipe/framework/port:file_helpers",
        "//mediapipe/framework/port:opencv_highgui",
        "//mediapipe/framework/port:opencv_imgproc",
        "//mediapipe/framework/port:opencv_video",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/framework/port:status",
        "//mediapipe/util:resource_util",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/log:absl_log",
    ],
)

# Linux only.
# Must have a GPU with EGL support:
# ex: sudo apt-get install mesa-common-dev libegl1-mesa-dev libgles2-mesa-dev
# (or similar nvidia/amd equivalent)
cc_library(
    name = "demo_run_graph_main_gpu",
    srcs = ["demo_run_graph_main_gpu.cc"],
    deps = [
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/formats:image_frame_opencv",
        "//mediapipe/framework/port:file_helpers",
        "//mediapipe/framework/port:opencv_highgui",
        "//mediapipe/framework/port:opencv_imgproc",
        "//mediapipe/framework/port:opencv_video",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/framework/port:status",
        "//mediapipe/gpu:gl_calculator_helper",
        "//mediapipe/gpu:gpu_buffer",
        "//mediapipe/gpu:gpu_shared_data_internal",
        "//mediapipe/util:resource_util",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/log:absl_log",
    ],
)
