load("//mediapipe/framework/port:build_config.bzl", "mediapipe_proto_library")

licenses(["notice"])

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "HeadCalculator",
    srcs = ["HeadCalculator.cc"],
    hdrs = ["HeadCalculator.h"],
    deps = [
        "//mediapipe/calculators/tensor:image_to_tensor_calculator_core",
        "//mediapipe/calculators/tensor:image_to_tensor_utils",
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
