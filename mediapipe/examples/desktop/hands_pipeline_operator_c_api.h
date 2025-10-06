// C API for HandsPipelineOperator to be used in other languages (e.g. rust).
// this is necessary because C++ ABI is not stable across compiler versions hence C++ cannot be typically be linked from other languages.
// and also because C++ exceptions cannot propagate across language boundaries so we need to catch them and convert them to plain
// error codes and messages for the other language to consume.

#ifndef MEDIAPIPE_EXAMPLES_DESKTOP_HANDS_PIPELINE_OPERATOR_C_API_H_
#define MEDIAPIPE_EXAMPLES_DESKTOP_HANDS_PIPELINE_OPERATOR_C_API_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>

// Opaque handle for the C++ HandsPipelineOperator instance
typedef void* HandsPipelineOperatorHandle;

// Create operator from serialized CalculatorGraphConfig and output stream names list (comma-separated string)
HandsPipelineOperatorHandle hands_pipeline_operator_create(
    const char* graph_definition, size_t config_size,
    const char* output_streams_csv);

// Destroy operator
void hands_pipeline_operator_destroy(HandsPipelineOperatorHandle handle);

// Push image (copy): input as uint8_t* (BGR), width, height, channels, timestamp
int hands_pipeline_operator_push_image(
    HandsPipelineOperatorHandle handle,
    const uint8_t* image_data, int width, int height, int channels,
    int64_t timestamp_us);

// Wait for output: frame_number, returns serialized PipelineOutputData
int hands_pipeline_operator_wait_for_output(
    HandsPipelineOperatorHandle handle,
    int frame_number,
    char** output_data, size_t* output_size);

// Finalize operator
int hands_pipeline_operator_finalize(HandsPipelineOperatorHandle handle);

// Get last error message (thread-local or global)
const char* hands_pipeline_operator_get_last_error();

#ifdef __cplusplus
}
#endif

#endif  // MEDIAPIPE_EXAMPLES_DESKTOP_HANDS_PIPELINE_OPERATOR_C_API_H_

