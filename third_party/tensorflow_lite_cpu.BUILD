# Minimal CPU-only TensorFlow Lite build for hand inference
# Excludes GPU, delegate, and hardware accelerator dependencies

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])

cc_library(
    name = "tensorflow_lite",
    hdrs = [
        "c_api.h",
        "c_api_experimental.h", 
        "c_api_types.h",
        "builtin_ops.h",
        "interpreter.h",
        "model.h",
        "optional_debug_tools.h",
        "kernels/register.h",
        "core/api/error_reporter.h",
        "core/api/flatbuffer_conversions.h",
        "core/api/op_resolver.h",
        "core/api/tensor_utils.h",
        "core/macros.h",
        "schema/schema_generated.h",
    ],
    srcs = [
        "c_api.cc",
        "interpreter.cc",
        "model.cc",
        "optional_debug_tools.cc",
        "kernels/builtin_op_kernels.h",
        "kernels/register.cc",
        "kernels/internal/tensor_utils.cc",
        "core/api/error_reporter.cc",
        "core/api/flatbuffer_conversions.cc",
        "core/api/op_resolver.cc",
        "core/api/tensor_utils.cc",
    ] + glob([
        "kernels/*.cc",
        "kernels/internal/*.cc",
        "core/*.cc",
        "core/api/*.cc",
    ], exclude = [
        "**/*_test.cc",
        "**/*_gpu*",
        "**/*_delegate*",
        "kernels/gpu/**",
        "delegates/**",
    ]),
    deps = [
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span", 
        "@com_github_google_flatbuffers//:flatbuffers",
        "@eigen_archive//:eigen3",
        "@XNNPACK//:XNNPACK",
    ],
    includes = ["."],
    copts = [
        "-DTFLITE_WITH_RUY",
        "-DEIGEN_USE_THREADS",
        "-DTF_LITE_STATIC_MEMORY",
    ],
)

# CPU-only kernels without GPU/delegate support
cc_library(
    name = "builtin_ops",
    hdrs = [
        "builtin_ops.h",
        "kernels/register.h",
    ],
    srcs = [
        "kernels/register.cc",
    ] + glob([
        "kernels/*.cc",
        "kernels/internal/*.cc",
    ], exclude = [
        "**/*_test.cc",
        "**/*_gpu*",
        "**/*_delegate*",
    ]),
    deps = [
        ":tensorflow_lite",
    ],
)