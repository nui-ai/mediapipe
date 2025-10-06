// Copyright 2019 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// C API test: clone of hand_tracking_pipeline_run_main.cc using the C API
#include <cstdlib>
#include <fstream>
#include <vector>
#include <string>
#include <iostream>
#include <memory>

#include "mediapipe/examples/desktop/hands_pipeline_operator_c_api.h"
#include "mediapipe/examples/desktop/pipeline_output.pb.h"
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/calculator.pb.h"
#include "mediapipe/util/resource_util.h"
#include <google/protobuf/util/delimited_message_util.h>
#include <google/protobuf/util/message_differencer.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>

#include <opencv2/opencv.hpp>

constexpr char kInputStream[] = "image";
constexpr char kOutputProtoFilename[] = "output_data_cpp.pb";
constexpr char kReferenceProtoFilename[] = "output_data_v0.10.13.pb";

ABSL_FLAG(std::string, graph_file, "", "Name of pipeline pbtxt file.");
ABSL_FLAG(std::string, input_video_path, "", "Full path of video to load. If not provided, attempt to use a webcam.");
ABSL_FLAG(std::string, output_video_path, "", "Full path of where to save result (.mp4 only). If not provided, show result in a window.");

// helper function to read reference data from file c style (we could just have used cpp)
bool ReadReferenceData(const std::string& filename, std::vector<mediapipe::PipelineOutputData>& out) {
    std::ifstream input(filename, std::ios::binary);
    if (!input.is_open()) {
        std::cerr << "Failed to open reference file: " << filename << std::endl;
        return false;
    }
    google::protobuf::io::IstreamInputStream zero_copy_input(&input);
    bool clean_eof = false;
    int msg_count = 0;
    while (true) {
        mediapipe::PipelineOutputData msg;
        std::streampos pos = input.tellg();
        if (!google::protobuf::util::ParseDelimitedFromZeroCopyStream(&msg, &zero_copy_input, &clean_eof)) {
            if (msg_count == 0) {
                std::cerr << "Failed to parse any messages from " << filename << " (parse error at file offset " << pos << ")" << std::endl;
                return false;
            } else {
                break;
            }
        }
        if (clean_eof) break;
        out.push_back(msg);
        ++msg_count;
    }
    std::cout << "Successfully loaded " << msg_count << " reference records from " << filename << std::endl;
    return true;
}

int main(int argc, char** argv) {

    google::InitGoogleLogging(argv[0]);

    ABSL_LOG(INFO) << "this is the c-api pipeline runner";

    absl::ParseCommandLine(argc, argv);

    // Load the reference output data (we could just read it using cpp)
    std::vector<mediapipe::PipelineOutputData> reference_data;
    ReadReferenceData(kReferenceProtoFilename, reference_data);

    // load the graph definition from its pbtxt file
    std::string graph_protobuf_definition;
    if (!mediapipe::file::GetContents(absl::GetFlag(FLAGS_graph_file), &graph_protobuf_definition).ok()) {
        std::cerr << "Failed to load graph file: " << absl::GetFlag(FLAGS_graph_file) << std::endl;
        return EXIT_FAILURE;
    }
    mediapipe::CalculatorGraphConfig pipeline_definition = mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(graph_protobuf_definition);
    std::string pipeline_definition_serialized;
    if (!pipeline_definition.SerializeToString(&pipeline_definition_serialized)) {
        std::cerr << "Failed to serialize CalculatorGraphConfig" << std::endl;
        return EXIT_FAILURE;
    }

    // output stream names as single string for c api simplicity
    const std::string output_streams_csv = "multi_hand_landmarks,multi_hand_world_landmarks,multi_handedness";

    // instantiate the graph operator object
    HandsPipelineOperatorHandle pipeline_operator = hands_pipeline_operator_create(
        pipeline_definition_serialized.data(), pipeline_definition_serialized.size(), output_streams_csv.c_str());
    if (!pipeline_operator) {
        std::cerr << "Failed to create HandsPipelineOperator via C API: " << hands_pipeline_operator_get_last_error() << std::endl;
        return EXIT_FAILURE;
    }

    // Open video/camera
    cv::VideoCapture capture;
    const bool video_file_input = !absl::GetFlag(FLAGS_input_video_path).empty();
    if (video_file_input) {
        capture.open(absl::GetFlag(FLAGS_input_video_path));
    } else {
        capture.open(0);
    }
    if (!capture.isOpened()) {
        std::cerr << "Failed to open video/camera" << std::endl;
        hands_pipeline_operator_destroy(pipeline_operator);
        return EXIT_FAILURE;
    }

    std::ofstream output_proto_file(kOutputProtoFilename, std::ios::binary | std::ios::trunc);
    if (!output_proto_file.is_open()) {
        std::cerr << "Failed to open output proto file: " << kOutputProtoFilename << std::endl;
        hands_pipeline_operator_destroy(pipeline_operator);
        return EXIT_FAILURE;
    }

    for (int i = 0; i < 999999; ++i) {
        cv::Mat input_frame_raw;
        capture >> input_frame_raw;
        if (input_frame_raw.empty()) {
            if (!video_file_input) {
                std::cerr << "Empty frame from camera being ignored" << std::endl;
                continue;
            }
            break;
        }

        cv::Mat input_frame;
        cv::cvtColor(input_frame_raw, input_frame, cv::COLOR_BGR2RGB);
        if (!video_file_input) { cv::flip(input_frame, input_frame, 1); }

        size_t frame_timestamp_us = (double)cv::getTickCount() / (double)cv::getTickFrequency() * 1e6;
        int push_status = hands_pipeline_operator_push_image(
            pipeline_operator,
            input_frame.data, input_frame.cols, input_frame.rows, input_frame.channels(),
            frame_timestamp_us);
        if (push_status != 0) {
            std::cerr << "push_image failed: " << hands_pipeline_operator_get_last_error() << std::endl;
            break;
        }

        char* output_data = nullptr;
        size_t output_size = 0;
        int wait_status = hands_pipeline_operator_wait_for_output(
            pipeline_operator, i, &output_data, &output_size);
        if (wait_status != 0) {
            std::cerr << "wait_for_output failed: " << hands_pipeline_operator_get_last_error() << std::endl;
            break;
        }

        mediapipe::PipelineOutputData stream_data_msg;
        if (!stream_data_msg.ParseFromArray(output_data, output_size)) {
            std::cerr << "Failed to parse PipelineOutputData from C API output" << std::endl;
            free(output_data);
            break;
        }
        free(output_data);

        // write the current frame output to file
        google::protobuf::util::SerializeDelimitedToOstream(stream_data_msg, &output_proto_file);

        // compare with reference data if available
        if (!reference_data.empty()) {
            if (i < reference_data.size()) {
                google::protobuf::util::MessageDifferencer differ;
                std::string diff;
                differ.ReportDifferencesToString(&diff);
                if (!differ.Compare(stream_data_msg, reference_data[i])) {
                    std::cerr << "Pipeline output at frame " << i << " is different than the reference output:\n" << diff << std::endl;
                    std::cerr << "Terminating early due to difference in output at frame " << i << std::endl;
                    break;
                } else {
                    std::cout << "Pipeline output for frame " << i << " is identical to its reference output." << std::endl;
                }
            } else {
                std::cerr << "Reference output file doesn't have data for frame " << i << std::endl;
            }
        }
    }
    output_proto_file.close();
    int finalize_status = hands_pipeline_operator_finalize(pipeline_operator);
    if (finalize_status != 0) {
        std::cerr << "Error during mediapipe graph finalization: " << hands_pipeline_operator_get_last_error() << std::endl;
    }
    hands_pipeline_operator_destroy(pipeline_operator);
    return EXIT_SUCCESS;
}
