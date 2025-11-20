#include "mediapipe/examples/desktop/hand_tracking_c_api.h"

// C-style error state management (thread-local)
static thread_local std::string g_last_error;
static void set_last_error(const std::string& err) { g_last_error = err; }
extern "C" const char* hand_tracking_get_last_error() { return g_last_error.c_str(); }

// this struct is the heart of enabling the API.
// it is a C++ struct type definition encapsulating only a pointer to a HandsPipelineOperator instance,
// which is only used by the C++ part of the C api, and on the C facade of the api is seen as merely
// an opaque pointer (void *) of type HandsPipelineOperatorHandle.
//
// how this plays out in more detail:
//
// C facade:
//   the C api consumer only sees a HandTrackingCoreOpaqueHandle (a.k.a in this context an opaque handler)
//   which it gets back from calling the C api's instantiation proxy function hands_pipeline_operator_create.
//   the C api consumer then passes that opaque handle back on each subsequent api call. for the c code
//   it's just a pointer.
//
// API implementation functions:
//   within the function implementations of the C api, that opaque handler is cast back
//   to this C++ struct type thus enabling the C api functions to operate the C++ object.
//   this pattern enables C code to operate a C++ object.
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
  const std::shared_ptr<const hand_tracking_mp_lean::Image>& image) {

  auto* cpp_instance_wrapper = static_cast<CppInstanceWrapper*>(opaque_handle);  // casts the opaque pointer back to C++
  absl::StatusOr<std::unique_ptr<hand_tracking_mp_lean::ImageHandTrackingAndInferenceResult>> result = cpp_instance_wrapper->cpp_impl->Process(image);
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
