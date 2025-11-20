// C API for the CPP HandsPipelineOperator to be used in other languages (e.g. rust).
// this is necessary because C++ ABI is not stable across compiler versions hence C++ cannot be typically be linked from other languages.
// and also because C++ exceptions cannot propagate across language boundaries so we need to catch them and convert them to plain
// error codes and messages for the other language to consume.
// this is a C api which operates an underlying HandsPipelineOperator C++ instance ― it defines C functions which
// enable C code to create/destroy and fully make use of the underlying C++ object.

#ifndef MEDIAPIPE_EXAMPLES_DESKTOP_HANDS_PIPELINE_OPERATOR_C_API_H_
#define MEDIAPIPE_EXAMPLES_DESKTOP_HANDS_PIPELINE_OPERATOR_C_API_H_
#include "mediapipe/framework/deps/safe_int.h"

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>

// Opaque handle for the C++ HandsPipelineOperator instance,
// being returned by the create function and taken as input
// by the other below API functions
typedef void* HandsPipelineOperatorOpaqueHandle;

// Create a pipeline operator C++ object, returning it as an opaque C pointer
HandsPipelineOperatorOpaqueHandle hands_pipeline_operator_create(
    uint max_hands_to_track,
    const char* graph_file_path,
    const char* output_streams_csv);  // output stream names list as comma-separated string

// Push image into the pipeline (by copy) for processing
int hands_pipeline_operator_push_image(
    HandsPipelineOperatorOpaqueHandle opaque_handle,
    const uint8_t* image_data, int width, int height, int channels,
    int64_t timestamp_us);

// Wait for pipeline output: returns a serialized PipelineOutputData
int hands_pipeline_operator_wait_for_output(
    HandsPipelineOperatorOpaqueHandle opaque_handle,
    int frame_number,
    char** output_data, size_t* output_size);

// destroys a created pipeline operator object, this function is internally called by `finalize` below,
// the code consuming this C API does not have to call it.
static void hands_pipeline_operator_destroy(HandsPipelineOperatorOpaqueHandle opaque_handle);

// Finalize the pipeline
int hands_pipeline_operator_finalize(HandsPipelineOperatorOpaqueHandle opaque_handle);

// function implementations for the above signatures return non-zero and set a "last_error"
// string value if they fail. This function should be used by the caller, to retrieve that
// last message for the caller, in case the caller gets a non-zero return value
// from any of the above API-exposed functions of this C API.
const char* hands_pipeline_operator_get_last_error();

#ifdef __cplusplus
}
#endif

#endif  // MEDIAPIPE_EXAMPLES_DESKTOP_HANDS_PIPELINE_OPERATOR_C_API_H_
