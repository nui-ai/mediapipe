# Export the main source file for hand tracking.
exports_files(["simple_run_graph_main.cc"])

cc_library(
    name = "simple_run_graph_main",
    srcs = ["simple_run_graph_main.cc"],
    hdrs = [],
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
        "@com_google_absl//absl/strings",    ],
    visibility = ["//mediapipe/nui/desktop/hand_tracking:__pkg__"],
)
