#include "mediapipe/examples/desktop/hand_tracking_c_api.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include <opencv2/opencv.hpp>

/// C-style error state management (thread-local)
static thread_local std::string g_last_error;
static void set_last_error(const std::string& err) { g_last_error = err; }
extern "C" const char* hand_tracking_get_last_error() { return g_last_error.c_str(); }

/// this struct is the heart of enabling the API.
/// it is a C++ struct type definition encapsulating only a pointer to a HandsPipelineOperator instance,
/// which is only used by the C++ part of the C api, and on the C facade of the api is seen as merely
/// an opaque pointer (void *) of type HandsPipelineOperatorHandle.
///
/// how this plays out in more detail:
///
/// C facade:
///   the C api consumer only sees a HandTrackingCoreOpaqueHandle (a.k.a in this context an opaque handler)
///   which it gets back from calling the C api's instantiation proxy function (hand_tracking_core_create).
///   the C api consumer then passes that opaque handle back on each subsequent api call. for the c code
///   it's just a pointer.
///
/// API implementation functions:
///   within the function implementations of the C api, that opaque handler is cast back
///   to this C++ struct type thus enabling the C api functions to operate the C++ object.
///   this pattern enables C code to operate a C++ object when linked together with this C api.
///
/// by virtue of enabling strict-C code to do that, we enable use from other languages
/// which are able to call into C api reliant on its standard ABI ― such as in our case
/// from Rust.
struct CppInstanceWrapper {
  std::unique_ptr<hand_tracking_mp_lean::HandTrackingCore> cpp_impl;
};

extern "C" HandTrackingCoreOpaqueHandle hand_tracking_core_create(const uint max_hands_to_track) {
  // safe_init_protobuf();
  set_last_error("");
  auto* cpp_instance_wrapper = new CppInstanceWrapper;
  cpp_instance_wrapper->cpp_impl = std::make_unique<hand_tracking_mp_lean::HandTrackingCore>(max_hands_to_track);
  return cpp_instance_wrapper;
}

extern "C" int hand_tracking_core_process(
  HandTrackingCoreOpaqueHandle opaque_handle,
  const uint8_t* data, size_t width, size_t height, size_t stride_row, size_t stride_col) {  // the image as a pointer to its numpy array bytes, plus the shape and strides of this array

  auto* cpp_instance_wrapper = static_cast<CppInstanceWrapper*>(opaque_handle);  // casts the opaque pointer back to C++

  // no-copy wrap the image data as a CV::Mat object and provide it as a reference to downstream api
  assert(stride_row == stride_col && "stride_row and stride_col must be equal for directly wrapping image data as a cv::Mat object"); // there's no OpenCV direct constructor that takes different stride size for cols and rows
  cv::Mat image(height, width, CV_8UC3, const_cast<uint8_t*>(data), stride_row);
  auto image_ptr = std::make_shared<cv::Mat>(std::move(image));  // wrap the cv::Mat image by a shared pointer, as the builder requires a reference type
  auto image_frame_ptr = std::make_shared<hand_tracking_mp_lean::ImageFrame>(
      hand_tracking_mp_lean::ImageFormat::SRGB,
      image_ptr->cols, image_ptr->rows,
      image_ptr->step[0],
      image_ptr->data,
      [image_ptr](uint8_t*) {
        // underlying image data lifetime management:
        // the role of the current explicit deleter lambda is only to hold a reference to the shared pointer to the cv::Mat object,
        // thus preventing the deletion of the original cv2::Mat image object up until the current ImageFrame dervied from it, is destroyed itself,
        // thus ensuring that the image data has the expected lifespan;
        // it would otherwise get freed before the ImageFrame object is actually used.
        // this is the intended scenario for the ImageFrame constructor being used here.
      });
  auto const hand_tracking_input_image_object = std::make_shared<const hand_tracking_mp_lean::Image>(image_frame_ptr);

  // pass the image to the hand tracking instance
  absl::StatusOr<std::unique_ptr<hand_tracking_mp_lean::ImageHandTrackingAndInferenceResult>> result = cpp_instance_wrapper->cpp_impl->Process(hand_tracking_input_image_object);
  if (!result.ok()) {
    set_last_error(std::string(result.status().message()));
    return -1;
  }
  return 0;
}

/// destructs a created instance, this function is internal and not exposed as API
static void hand_tracking_core_destroy(HandTrackingCoreOpaqueHandle opaque_handle) {
  // delete runs the destructor of the struct which opaque_handle points at,
  // after calling the destructor of its contained C++ instance.
  if (opaque_handle) delete static_cast<CppInstanceWrapper*>(opaque_handle);
}

/// finalizes the HandTrackingCore instance,
/// which currently, requires no special finalization other than calling its destructor
extern "C" int hand_tracking_core_finalize(HandTrackingCoreOpaqueHandle opaque_handle) {
  set_last_error("");
  if (!opaque_handle) {
    set_last_error("nullptr passed to finalize");
    return -1;
  }
  hand_tracking_core_destroy(opaque_handle);
  return 0;
}

/// returns a version string for this API, or more broadly, this API and moreso the underlying C++ implementation driven by it,
/// which is implied because this api is consumed in other repositories (via a lib) generated by the current code repository
/// and hence version compatibility becomes a thing.
const char* hand_tracking_core_version() {
  return "1.0.0";
}

