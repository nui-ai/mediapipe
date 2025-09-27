load("//mediapipe/framework/tool:mediapipe_graph.bzl", "mediapipe_simple_subgraph")


mediapipe_simple_subgraph(
    name = "hand_landmark_tracking_cpu_new",
    graph = "hand_landmark_tracking_cpu_new.pbtxt",
    visibility = ["//visibility:public"],
    register_as = "HandLandmarkTrackingCpuNew",
    deps = [
        "//mediapipe/calculators/core:gate_calculator",  # For GateCalculatorOptions extension
        "//mediapipe/calculators/util:association_calculator",  # For AssociationCalculatorOptions extension
        "//mediapipe/nui/desktop/calculators:palm_detection_cpu_new",  # Ensure PalmDetectionCpuNew is registered
        "//mediapipe/nui/desktop/calculators:palm_detection_detection_to_roi_new",  # Ensure PalmDetectionDetectionToRoiNew is registered
        "//mediapipe/nui/desktop/calculators:hand_landmark_cpu_new",  # Ensure HandLandmarkCpuNew is registered
        "//mediapipe/nui/desktop/calculators:hand_landmark_landmarks_to_roi_new",  # Ensure HandLandmarkLandmarksToRoiNew is registered
        "//mediapipe/nui/desktop/calculators:palm_detection_model_loader_new",
        "//mediapipe/nui/desktop/calculators:hand_landmark_model_loader_new",
    ],
)

mediapipe_simple_subgraph(
    name = "hand_landmark_model_loader_new",
    graph = "hand_landmark_model_loader_new.pbtxt",
    register_as = "HandLandmarkModelLoaderNew",
    deps = [
        "//mediapipe/calculators/core:constant_side_packet_calculator",
        "//mediapipe/calculators/tflite:tflite_model_calculator",
        "//mediapipe/calculators/util:resource_provider_calculator",
        "//mediapipe/framework/tool:switch_container",
    ],
)

mediapipe_simple_subgraph(
    name = "palm_detection_model_loader_new",
    graph = "palm_detection_model_loader_new.pbtxt",
    register_as = "PalmDetectionModelLoaderNew",
    deps = [
        "//mediapipe/calculators/core:constant_side_packet_calculator",
        "//mediapipe/calculators/tflite:tflite_model_calculator",
        "//mediapipe/calculators/util:resource_provider_calculator",
        "//mediapipe/framework/tool:switch_container",
    ],
)

mediapipe_simple_subgraph(
    name = "hand_landmark_cpu_new",
    graph = "hand_landmark_cpu_new.pbtxt",
    register_as = "HandLandmarkCpuNew",
    deps = [
        "//mediapipe/modules/hand_landmark:hand_landmark_model_loader",
        "//mediapipe/calculators/core:gate_calculator",
        "//mediapipe/calculators/core:split_vector_calculator",
        "//mediapipe/calculators/tensor:image_to_tensor_calculator",
        "//mediapipe/calculators/tensor:inference_calculator",
        "//mediapipe/calculators/tensor:tensors_to_classification_calculator",
        "//mediapipe/calculators/tensor:tensors_to_floats_calculator",
        "//mediapipe/calculators/tensor:tensors_to_landmarks_calculator",
        "//mediapipe/calculators/util:landmark_letterbox_removal_calculator",
        "//mediapipe/calculators/util:landmark_projection_calculator",
        "//mediapipe/calculators/util:thresholding_calculator",
        "//mediapipe/calculators/util:world_landmark_projection_calculator",
    ],
)


mediapipe_simple_subgraph(
    name = "palm_detection_cpu_new",
    graph = "palm_detection_cpu_new.pbtxt",
    visibility = ["//visibility:public"],
    register_as = "PalmDetectionCpuNew",
    deps = [
        "//mediapipe/modules/palm_detection:palm_detection_model_loader",
        "//mediapipe/calculators/tensor:image_to_tensor_calculator",
        "//mediapipe/calculators/tensor:inference_calculator",
        "//mediapipe/calculators/tensor:tensors_to_detections_calculator",
        "//mediapipe/calculators/tflite:ssd_anchors_calculator",
        "//mediapipe/calculators/tflite:tflite_custom_op_resolver_calculator",
        "//mediapipe/calculators/util:detection_letterbox_removal_calculator",
        "//mediapipe/calculators/util:non_max_suppression_calculator"
    ])


mediapipe_simple_subgraph(
    name = "palm_detection_detection_to_roi_new",
    graph = "palm_detection_detection_to_roi_new.pbtxt",
    register_as = "PalmDetectionDetectionToRoiNew",
    deps = [
        "//mediapipe/calculators/util:detections_to_rects_calculator",
        "//mediapipe/calculators/util:rect_transformation_calculator",
    ],
)

mediapipe_simple_subgraph(
    name = "hand_landmark_landmarks_to_roi_new",
    graph = "hand_landmark_landmarks_to_roi_new.pbtxt",
    register_as = "HandLandmarkLandmarksToRoiNew",
    deps = [
        "//mediapipe/calculators/util:rect_transformation_calculator",
        "//mediapipe/modules/hand_landmark/calculators:hand_landmarks_to_rect_calculator",
    ],
)



cc_library(
    name = "new",
    srcs = [
        "association_norm_rect_calculator_new.cc",
        "collection_has_min_size_calculator_new.cc",
        "detection_letterbox_removal_calculator_new.cc",
        "detections_to_rects_calculator_new.cc",
        "gate_calculator_new.cc",
        "hand_landmarks_detector_graph_new.cc",
        "hand_landmarks_to_rect_calculator_new.cc",
        "image_properties_calculator_new.cc",
        "image_to_tensor_calculator_new.cc",
        "landmark_letterbox_removal_calculator_new.cc",
        "non_max_suppression_calculator_new.cc",
        "previous_loopback_calculator_new.cc",
        "rect_transformation_calculator_new.cc",
        "resource_provider_calculator_new.cc",
        "split_vector_calculator_new.cc",
        "ssd_anchors_calculator_new.cc",
        "tensors_to_classification_calculator_new.cc",
        "tensors_to_detections_calculator_new.cc",
        "tensors_to_floats_calculator_new.cc",
        "tensors_to_landmarks_calculator_new.cc",
        "tflite_custom_op_resolver_calculator_new.cc",
        "tflite_model_calculator_new.cc",
        "thresholding_calculator_new.cc",
    ],
    hdrs = [
        "collection_has_min_size_calculator_new.h",
        "detections_to_rects_calculator_new.h",
        "resource_provider_calculator_new.h",
        "split_vector_calculator_new.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:calculator_base",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "//mediapipe/framework/api2:builder",  # Added to fix missing header
        "//mediapipe/modules/hand_landmark:hand_landmark_cpu",
        "//mediapipe/modules/hand_landmark:hand_landmark_landmarks_to_roi",
        "//mediapipe/modules/hand_landmark:hand_landmark_model_loader",
        "//mediapipe/modules/hand_landmark:hand_landmark_tracking_cpu",
        "//mediapipe/modules/hand_landmark:palm_detection_detection_to_roi",
        "//mediapipe/modules/palm_detection:palm_detection_cpu",
        "//mediapipe/modules/palm_detection:palm_detection_model_loader",
        "//mediapipe/tasks/cc:common",  # Added to fix missing common.h header
        "//mediapipe/tasks/cc/components/processors:image_preprocessing_graph",  # Added to fix missing image_preprocessing_graph.h header
        "//mediapipe/tasks/cc/components/utils:gate",  # Added to fix missing gate.h header
        "//mediapipe/tasks/cc/core:model_task_graph",  # Added to fix missing model_task_graph.h header
        "//mediapipe/tasks/cc/vision/hand_landmarker/proto:hand_landmarks_detector_graph_options_cc_proto",  # Added for generated proto header
    ],
    alwayslink = 1,
)
