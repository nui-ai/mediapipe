load("//mediapipe/framework/port:build_config.bzl", "mediapipe_proto_library")

licenses(["notice"])

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "HandTrackingCalculator",
    srcs = ["HandTrackingCalculator.cc"],
    hdrs = ["HandTrackingCalculator.h"],
    deps = [
        ":liberated_core",
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

# as seen in deps below, the pipeline-liberated dependencies are still scattered around
# by the original disparate locations of the original pipeline calculators they have been cored from.
cc_library(
    name = "liberated_core",
    srcs = [
        "hand_tracking.cc",
        "hand_tracking_debug.cc",
    ],
    hdrs = [
        "hand_tracking.h",
        "hand_tracking_debug.h",
    ],
    deps = [
        "//mediapipe/calculators/core:split_vector_calculator_core",
        "//mediapipe/calculators/tensor:image_to_tensor_calculator_core",
        "//mediapipe/calculators/tensor:image_to_tensor_utils",
        "//mediapipe/calculators/tensor:inference_runner_new",
        "//mediapipe/calculators/tensor:model_inference",
        "//mediapipe/calculators/tensor:tensors_to_classification_calculator_cc_proto",
        "//mediapipe/calculators/tensor:tensors_to_classification_calculator_core",
        "//mediapipe/calculators/tensor:tensors_to_detections_calculator_core",
        "//mediapipe/calculators/tensor:tensors_to_floats_calculator_core",
        "//mediapipe/calculators/tensor:tensors_to_landmarks_calculator_cc_proto",
        "//mediapipe/calculators/tensor:tensors_to_landmarks_calculator_core",
        "//mediapipe/calculators/tensor:tensors_to_world_landmarks_calculator_core",
        "//mediapipe/calculators/util:association_calculator_core",
        "//mediapipe/calculators/util:detection_letterbox_removal",
        "//mediapipe/calculators/util:detections_to_rects_calculator_core",
        "//mediapipe/calculators/util:landmark_letterbox_removal_calculator_core",
        "//mediapipe/calculators/util:landmark_projection_calculator_cc_proto",
        "//mediapipe/calculators/util:landmark_projection_calculator_core",
        "//mediapipe/calculators/util:rect_transformation_calculator_core",
        "//mediapipe/calculators/util:world_landmark_projection_calculator_core",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:memory_manager",
        "//mediapipe/framework:memory_manager_service",
        "//mediapipe/framework/api2:port",
        "//mediapipe/framework/formats:classification_cc_proto",
        "//mediapipe/framework/formats:image",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/formats:rect_cc_proto",
        "//mediapipe/framework/formats:tensor",
        "//mediapipe/framework/port:status",
        "//mediapipe/modules/hand_landmark/calculators:hand_landmarks_to_rect_calculator_core",
        "@com_google_absl//absl/log:absl_log",
    ],
    alwayslink = 1,
)

# Graph-free face tracking is a separate artifact from hand tracking. Both use
# the generalized calculator cores above, but consumers of this target do not
# link the hand API or hand model assets.
cc_library(
    name = "face_liberated_core",
    srcs = ["face_tracking.cc"],
    hdrs = ["face_tracking.h"],
    deps = [
        "//mediapipe/calculators/tensor:image_to_tensor_calculator_core",
        "//mediapipe/calculators/tensor:image_to_tensor_utils",
        "//mediapipe/calculators/tensor:model_inference",
        "//mediapipe/calculators/tensor:tensors_to_detections_calculator_core",
        "//mediapipe/calculators/tensor:tensors_to_floats_calculator_core",
        "//mediapipe/calculators/tensor:tensors_to_landmarks_calculator_cc_proto",
        "//mediapipe/calculators/tensor:tensors_to_landmarks_calculator_core",
        "//mediapipe/calculators/util:association_calculator_core",
        "//mediapipe/calculators/util:detections_to_rects_calculator_core",
        "//mediapipe/calculators/util:landmark_projection_calculator_core",
        "//mediapipe/calculators/util:rect_transformation_calculator_core",
        "//mediapipe/framework:memory_manager",
        "//mediapipe/framework/formats:detection_cc_proto",
        "//mediapipe/framework/formats:image",
        "//mediapipe/framework/formats:landmark_cc_proto",
        "//mediapipe/framework/formats:rect_cc_proto",
        "//mediapipe/framework/formats:tensor",
        "//mediapipe/framework/port:status",
        "@com_google_absl//absl/status:statusor",
    ],
    alwayslink = 1,
)
