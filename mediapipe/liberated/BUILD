load("//mediapipe/framework/port:build_config.bzl", "mediapipe_proto_library")

licenses(["notice"])

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "HeadCalculator",
    srcs = ["HeadCalculator.cc"],
    hdrs = ["HeadCalculator.h"],
    deps = [
        ":liberated",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:memory_manager",
        "//mediapipe/framework:memory_manager_service",
        "//mediapipe/framework/api2:port",
        "//mediapipe/framework/formats:image",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/formats:rect_cc_proto",
        "//mediapipe/framework/formats:tensor",
        "//mediapipe/framework/port:status",
        "@com_google_absl//absl/log:absl_log",
    ],
    alwayslink = 1,
)

cc_library(
    name = "liberated",
    srcs = ["liberated.cc"],
    hdrs = ["liberated.h"],
    deps = [
        "//mediapipe/calculators/core:split_vector_calculator_core",
        "//mediapipe/calculators/tensor:detection_inference_calculator",
        "//mediapipe/calculators/tensor:image_to_tensor_calculator_core",
        "//mediapipe/calculators/tensor:image_to_tensor_utils",
        "//mediapipe/calculators/tensor:inference_calculator",  # clean to be its own dependencies not the calculator itself.
        "//mediapipe/calculators/tensor:inference_runner_new",
        "//mediapipe/calculators/tensor:model_inference",
        "//mediapipe/calculators/tensor:tensors_to_detections_calculator_core",
        "//mediapipe/calculators/tensor:tensors_to_floats_calculator_core",
        "//mediapipe/calculators/util:association_calculator_core",
        "//mediapipe/calculators/util:detection_letterbox_removal",
        "//mediapipe/calculators/util:detections_to_rects_calculator_core",
        "//mediapipe/calculators/util:rect_transformation_calculator_core",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:memory_manager",
        "//mediapipe/framework:memory_manager_service",
        "//mediapipe/framework/api2:port",
        "//mediapipe/framework/formats:image",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/formats:rect_cc_proto",
        "//mediapipe/framework/formats:tensor",
        "//mediapipe/framework/port:status",
        "@com_google_absl//absl/log:absl_log",
    ],
    alwayslink = 1,
)
