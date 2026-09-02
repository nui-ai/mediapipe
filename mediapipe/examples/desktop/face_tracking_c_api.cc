#include "mediapipe/examples/desktop/face_tracking_c_api.h"

#include <cstdlib>
#include <exception>
#include <limits>
#include <memory>
#include <string>

#include "mediapipe/examples/desktop/face_tracking_c_conversion.h"
#include "mediapipe/examples/desktop/nui_mediapipe_source_provenance.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/liberated/face_tracking.h"

namespace {

thread_local std::string last_error;

void SetLastError(const std::string& error) { last_error = error; }

struct FaceTrackingCoreWrapper {
  std::unique_ptr<hand_tracking_mp_lean::FaceTrackingCore> implementation;
};

hand_tracking_mp_lean::FaceTrackingOptions ConvertOptions(
    const FaceTrackingOptionsC* options) {
  hand_tracking_mp_lean::FaceTrackingOptions converted;
  if (options == nullptr) {
    return converted;
  }
  converted.max_faces = options->max_faces;
  converted.use_previous_landmarks = options->use_previous_landmarks != 0;
  converted.with_attention = options->with_attention != 0;
  converted.min_detection_confidence = options->min_detection_confidence;
  converted.min_tracking_confidence = options->min_tracking_confidence;
  converted.xnnpack_num_threads = options->xnnpack_num_threads;
  converted.estimate_pose = options->estimate_pose != 0;
  converted.vertical_fov_degrees = options->vertical_fov_degrees;
  return converted;
}

}  // namespace

extern "C" const char* face_tracking_get_last_error() {
  return last_error.c_str();
}

extern "C" FaceTrackingCoreOpaqueHandle face_tracking_core_create(
    const FaceTrackingOptionsC* options, const char* assets_path) {
  SetLastError("");
  try {
    const std::string assets_path_storage =
        assets_path == nullptr ? std::string() : std::string(assets_path);
    const std::string* assets_path_pointer =
        assets_path == nullptr ? nullptr : &assets_path_storage;
    auto wrapper = std::make_unique<FaceTrackingCoreWrapper>();
    wrapper->implementation =
        std::make_unique<hand_tracking_mp_lean::FaceTrackingCore>(
            ConvertOptions(options), assets_path_pointer);
    return wrapper.release();
  } catch (const std::exception& error) {
    SetLastError(error.what());
    return nullptr;
  } catch (...) {
    SetLastError("unknown exception while creating face tracker");
    return nullptr;
  }
}

extern "C" int face_tracking_core_process(
    FaceTrackingCoreOpaqueHandle opaque_handle, const uint8_t* data,
    size_t width, size_t height, size_t row_stride,
    FaceTrackingResultC** result_out) {
  SetLastError("");
  if (result_out == nullptr) {
    SetLastError("null result output pointer passed to face tracking");
    return -1;
  }
  *result_out = nullptr;
  if (opaque_handle == nullptr) {
    SetLastError("null face tracker handle passed to process");
    return -1;
  }
  if (data == nullptr) {
    SetLastError("null RGB data passed to face tracking");
    return -1;
  }
  if (width == 0 || height == 0) {
    SetLastError("face tracking input image dimensions must be non-zero");
    return -1;
  }
  if (width > static_cast<size_t>(std::numeric_limits<int>::max()) ||
      height > static_cast<size_t>(std::numeric_limits<int>::max()) ||
      row_stride > static_cast<size_t>(std::numeric_limits<int>::max())) {
    SetLastError("face tracking input image dimensions exceed ImageFrame limits");
    return -1;
  }
  if (row_stride < width * 3) {
    SetLastError("RGB row stride is smaller than three bytes per pixel");
    return -1;
  }

  try {
    auto image_frame = std::make_shared<hand_tracking_mp_lean::ImageFrame>(
        hand_tracking_mp_lean::ImageFormat::SRGB, static_cast<int>(width),
        static_cast<int>(height), static_cast<int>(row_stride),
        const_cast<uint8_t*>(data), [](uint8_t*) {});
    auto image = std::make_shared<const hand_tracking_mp_lean::Image>(
        std::move(image_frame));
    auto* wrapper = static_cast<FaceTrackingCoreWrapper*>(opaque_handle);
    auto cpp_result = wrapper->implementation->Process(image);
    if (!cpp_result.ok()) {
      SetLastError(std::string(cpp_result.status().message()));
      return -1;
    }

    auto* c_result = static_cast<FaceTrackingResultC*>(
        std::calloc(1, sizeof(FaceTrackingResultC)));
    if (c_result == nullptr) {
      SetLastError("failed to allocate face tracking result");
      return -1;
    }
    if (ConvertFaceTrackingResultToC(*cpp_result.value(), c_result,
                                     SetLastError) != 0) {
      face_tracking_result_destroy(c_result);
      return -1;
    }
    *result_out = c_result;
    return 0;
  } catch (const std::exception& error) {
    SetLastError(error.what());
    return -1;
  } catch (...) {
    SetLastError("unknown exception while processing face tracking input");
    return -1;
  }
}

extern "C" int face_tracking_core_reset(
    FaceTrackingCoreOpaqueHandle opaque_handle) {
  SetLastError("");
  if (opaque_handle == nullptr) {
    SetLastError("null face tracker handle passed to reset");
    return -1;
  }
  static_cast<FaceTrackingCoreWrapper*>(opaque_handle)
      ->implementation->Reset();
  return 0;
}

extern "C" int face_tracking_core_finalize(
    FaceTrackingCoreOpaqueHandle opaque_handle) {
  SetLastError("");
  if (opaque_handle == nullptr) {
    SetLastError("null face tracker handle passed to finalize");
    return -1;
  }
  delete static_cast<FaceTrackingCoreWrapper*>(opaque_handle);
  return 0;
}

extern "C" const char* face_tracking_core_version() { return "2.0.0"; }

extern "C" const char* face_tracking_nui_ai_mediapipe_source_commit() {
  return nui_ai_mediapipe_source_provenance::SourceCommit();
}

extern "C" int face_tracking_nui_ai_mediapipe_source_dirty() {
  return nui_ai_mediapipe_source_provenance::SourceDirty();
}

extern "C" int face_tracking_nui_ai_mediapipe_library_built_by_core_build_rs() {
  return nui_ai_mediapipe_source_provenance::LibraryBuiltByCoreBuildRs();
}
