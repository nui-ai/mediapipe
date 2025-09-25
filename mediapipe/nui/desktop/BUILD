cc_library(
    name = "shared_calculator_state",
    srcs = ["SharedCalculatorState.cc"],
    hdrs = ["SharedCalculatorState.h"],
    includes = ["mediapipe/nui/desktop"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "tick_count_calculator",
    srcs = ["TickCountCalculator.cc"],
    hdrs = ["SharedCalculatorState.h"],
    includes = ["mediapipe/nui/desktop"],
    deps = [
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:calculator_base",
        "//mediapipe/framework/formats:image",
        "@com_google_absl//absl/log:absl_log",
        ":shared_calculator_state",
    ],
    alwayslink = 1,  # Important: Ensures the calculator registration code is always linked
    visibility = ["//visibility:public"],
)

cc_library(
    name = "simple_run_graph_main",
    srcs = [
        "simple_run_graph_main.cc",
    ],
    hdrs = [],
    visibility = [
        "//mediapipe/nui/desktop/hand_tracking:__pkg__",
    ],
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
