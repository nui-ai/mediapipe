#include "mediapipe/examples/desktop/hands_pipeline_operator_c_api.h"
#include "mediapipe/examples/desktop/hands_pipeline_operator.h"
#include <string>
#include <vector>
#include <sstream>
#include <mutex>
#include <memory>
#include <cstring>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "mediapipe/framework/calculator.pb.h"
#include "mediapipe/examples/desktop/pipeline_output.pb.h"

// initialize protobuf
#include "google/protobuf/text_format.h"
extern "C" void hands_pipeline_operator_init_protobuf() {
    // Force protobuf initialization
    GOOGLE_PROTOBUF_VERIFY_VERSION;
}

// Error handling (thread-local)
static thread_local std::string g_last_error;
static void set_last_error(const std::string& err) { g_last_error = err; }
extern "C" const char* hands_pipeline_operator_get_last_error() { return g_last_error.c_str(); }

struct HandsPipelineOperatorWrapper {
    std::unique_ptr<mediapipe::HandsPipelineOperator> impl;
};

extern "C" HandsPipelineOperatorHandle hands_pipeline_operator_create(
    const char* graph_file_path,
    const char* output_streams_csv) {
    set_last_error("");
    std::vector<std::string> output_streams;
    std::istringstream ss(output_streams_csv ? output_streams_csv : "");
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (!item.empty()) output_streams.push_back(item);
    }
    HandsPipelineOperatorWrapper* wrapper = new HandsPipelineOperatorWrapper;
    auto status_or_op = mediapipe::HandsPipelineOperator::Create(std::string(graph_file_path), output_streams);
    if (!status_or_op.ok()) {
        set_last_error(std::string(status_or_op.status().message()));
        delete wrapper;
        return nullptr;
    }
    wrapper->impl = std::move(status_or_op.value());
    return wrapper;
}

extern "C" void hands_pipeline_operator_destroy(HandsPipelineOperatorHandle handle) {
    if (handle) delete static_cast<HandsPipelineOperatorWrapper*>(handle);
}

extern "C" int hands_pipeline_operator_push_image(
    HandsPipelineOperatorHandle handle,
    const uint8_t* image_data, int width, int height, int channels,
    int64_t timestamp_us) {
    set_last_error("");
    if (!handle || !image_data || width <= 0 || height <= 0 || channels <= 0) {
        set_last_error("Invalid arguments to push_image");
        return -1;
    }
    cv::Mat input_frame(height, width, channels == 3 ? CV_8UC3 : CV_8UC1, (void*)image_data);
    // Copy to avoid lifetime issues
    cv::Mat frame_copy = input_frame.clone();
    auto* wrapper = static_cast<HandsPipelineOperatorWrapper*>(handle);
    absl::Status status = wrapper->impl->push_image(frame_copy, timestamp_us);
    if (!status.ok()) {
        set_last_error(std::string(status.message()));
        return -1;
    }
    return 0;
}

extern "C" int hands_pipeline_operator_wait_for_output(
    HandsPipelineOperatorHandle handle,
    int frame_number,
    char** output_data, size_t* output_size) {
    set_last_error("");
    if (!handle || !output_data || !output_size) {
        set_last_error("Invalid arguments to wait_for_output");
        return -1;
    }
    auto* wrapper = static_cast<HandsPipelineOperatorWrapper*>(handle);
    mediapipe::PipelineOutputData output;
    absl::Status status = wrapper->impl->wait_for_output(&output, frame_number);
    if (!status.ok()) {
        set_last_error(std::string(status.message()));
        return -1;
    }
    std::string serialized;
    if (!output.SerializeToString(&serialized)) {
        set_last_error("Failed to serialize PipelineOutputData");
        return -1;
    }
    *output_size = serialized.size();
    *output_data = (char*)malloc(*output_size);
    if (!*output_data) {
        set_last_error("Failed to allocate output buffer");
        return -1;
    }
    memcpy(*output_data, serialized.data(), *output_size);
    return 0;
}

extern "C" int hands_pipeline_operator_finalize(HandsPipelineOperatorHandle handle) {
    set_last_error("");
    if (!handle) {
        set_last_error("Invalid handle to finalize");
        return -1;
    }
    auto* wrapper = static_cast<HandsPipelineOperatorWrapper*>(handle);
    absl::Status status = wrapper->impl->finalize();
    if (!status.ok()) {
        set_last_error(std::string(status.message()));
        return -1;
    }
    return 0;
}
