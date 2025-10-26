load("//mediapipe/framework/port:build_config.bzl", "mediapipe_proto_library")

licenses(["notice"])

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "HeadCalculator",
    srcs = ["HeadCalculator.cc"],
    hdrs = ["HeadCalculator.h"],
    deps = [
        "//mediapipe/calculators/util:collection_has_min_size_calculator",
        "//mediapipe/framework:calculator_framework",
        "@com_google_absl//absl/status",
        "//mediapipe/calculators/tensor:image_to_tensor_calculator_core"
    ],
    alwayslink = 1,
)
