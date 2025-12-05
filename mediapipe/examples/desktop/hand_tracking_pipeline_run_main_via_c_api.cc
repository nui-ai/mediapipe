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
// drives the C API of operating our mediapipe pipeline:
// practically a clone of hand_tracking_pipeline_run_main.cc,
// which unlike the former does not directly use the C++ implementation of the pipeline operator,
// but the C api for it.

#include <cstdlib>
#include <fstream>
#include <filesystem>
#include <vector>
#include <string>
#include <iostream>
#include <memory>

#include "mediapipe/examples/desktop/hands_pipeline_operator_c_api.h" // the C api header
#include "mediapipe/examples/desktop/pipeline_output.pb.h"
#include "mediapipe/examples/desktop/pipeline_output_util.h"

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/calculator.pb.h"
#include "mediapipe/util/resource_util.h"

#include <google/protobuf/util/delimited_message_util.h>
#include <google/protobuf/util/message_differencer.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>

#include <opencv2/opencv.hpp>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"


constexpr char kInputStream[] = "image";
constexpr char kOutputProtoFilename[] = "output_data_cpp.pb";
constexpr char kReferenceProtoFilename[] = "output_data_two_hands_num_hands_3_v0.10.13.pb";

ABSL_FLAG(std::uint32_t, max_num_hands, 0, "maximum number of hands to track");
ABSL_FLAG(std::string, graph_file, "", "Name of pipeline pbtxt file.");
ABSL_FLAG(std::string, input_video_path, "", "Full path of video to load. If not provided, will attempt to use webcam input (not tested).");
ABSL_FLAG(std::string, output_video_path, "", "Full path of where to save result (.mp4 only). If not provided, show result in a window (not tested).");

// Validates and logs the PROJECT_ROOT_DIR env variable.
void ValidateProjectRootDirectoryOrExit() {
    const char* root_dir = std::getenv("PROJECT_ROOT_DIR");
    if (!root_dir || std::string(root_dir).empty()) {
        std::cerr << "ERROR: the environment variable PROJECT_ROOT_DIR must be set.";
        std::exit(EXIT_FAILURE);
    }
    std::string dir_str(root_dir);
    if (!std::filesystem::path(dir_str).is_absolute()) {
        std::cerr << "ERROR: the environment variable PROJECT_ROOT_DIR must be an absolute path, but got: '" << dir_str << "'\n";
        std::exit(EXIT_FAILURE);
    }
    if (!std::filesystem::exists(dir_str)) {
        std::cerr << "ERROR: the path provided by the environment variable PROJECT_ROOT_DIR does not exist: '" << dir_str << "'\n";
        std::exit(EXIT_FAILURE);
    }
    if (!std::filesystem::is_directory(dir_str)) {
        std::cerr << "ERROR: the value provided by the environment variable PROJECT_ROOT_DIR is not a directory: '" << dir_str << "'\n";
        std::exit(EXIT_FAILURE);
    }
    std::cerr << "PROJECT_ROOT_DIR is set to: '" << dir_str << "'\n";
}

bool get_project_root_dir(const std::string &filename, std::string &value1) {
    const char* root_dir = std::getenv("PROJECT_ROOT_DIR");
    if (root_dir && std::string(root_dir).length() > 0) {
        value1 = (std::filesystem::path(root_dir) / filename).string();
        return true;
    }
    return false;
}

// Returns the full path by prefixing with PROJECT_ROOT_DIR if set and path is not absolute.
std::string GetProjectRootedPath(const std::string& filename) {
    if (filename.empty()) {
        throw std::invalid_argument("filename cannot be empty");
    }
    if (std::filesystem::path(filename).is_absolute()) {
        return filename;
    }
    std::string project_root_dir;
    if (get_project_root_dir(filename, project_root_dir)) {
        return project_root_dir;
    }

    throw std::runtime_error("a relative path cannot be resolved when the PROJECT_ROOT_DIR environment variable is not set");
}

// helper function to read the reference output data from file c style (we could just have used cpp)
bool ReadReferenceData(const std::string& filename, std::vector<hand_tracking_mp_lean::PipelineOutputData>& out) {
    std::ifstream input(filename, std::ios::binary);
    if (!input.is_open()) {
        std::cerr << "Failed to open reference file: " << filename << std::endl;
        return false;
    }
    google::protobuf::io::IstreamInputStream zero_copy_input(&input);
    bool clean_eof = false;
    int msg_count = 0;
    while (true) {
        hand_tracking_mp_lean::PipelineOutputData msg;
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

/// runs the hands pipeline through it's C api, thus being an example of usage of our C api for pipeline running,
/// and providing validation for the C API's operation. although this main is not strictly the C subset of C++,
/// it uses many C++ language features, it drives the C api thus providing only soft exemplification and soft
/// validation of usage of the C API which it drives.
///
/// a pure C language test (usage example) may also be written.
/// the C API is currently actively consumed from rust FFI,
int main(int argc, char** argv) {

    google::InitGoogleLogging(argv[0]);

    ABSL_LOG(INFO) << "this is the C api pipeline runner";
    ABSL_LOG(INFO) << "working directory: " << std::filesystem::current_path();

    absl::ParseCommandLine(argc, argv);

    ValidateProjectRootDirectoryOrExit();

    // load the reference output data (we could just read it using cpp)
    std::vector<hand_tracking_mp_lean::PipelineOutputData> reference_data;
    ReadReferenceData(GetProjectRootedPath(kReferenceProtoFilename), reference_data);

    // output stream names as single string for c api simplicity
    const std::string output_streams_csv = "multi_hand_landmarks,multi_hand_world_landmarks,multi_handedness";

    // open an output file for the inferences
    std::ofstream output_proto_file(kOutputProtoFilename, std::ios::binary | std::ios::trunc);
    if (!output_proto_file.is_open()) {
        std::cerr << "Failed to open output proto file: " << kOutputProtoFilename << std::endl;
        return EXIT_FAILURE;
    }

    // open the input video stream
    cv::VideoCapture capture;
    const bool video_file_input = !absl::GetFlag(FLAGS_input_video_path).empty();
    if (video_file_input) {
        capture.open(GetProjectRootedPath(absl::GetFlag(FLAGS_input_video_path)));
    } else {
        capture.open(0);
    }
    if (!capture.isOpened()) {
        std::cerr << "Failed to open video/camera" << std::endl;
        return EXIT_FAILURE;
    }

    // instantiate a pipeline operator from the C api
    HandsPipelineOperatorOpaqueHandle pipeline_operator = hands_pipeline_operator_create(
        absl::GetFlag(FLAGS_max_num_hands),
        GetProjectRootedPath(absl::GetFlag(FLAGS_graph_file)).c_str(),
        output_streams_csv.c_str());
    if (!pipeline_operator) {
        std::cerr << "Failed to create HandsPipelineOperator via C api: " << hands_pipeline_operator_get_last_error() << std::endl;
        return EXIT_FAILURE;
    }

    // loop the input video/camera
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

        hand_tracking_mp_lean::PipelineOutputData stream_data_msg;
        if (!stream_data_msg.ParseFromArray(output_data, output_size)) {
            std::cerr << "Failed to parse PipelineOutputData from C API output" << std::endl;
            free(output_data);
            break;
        }
        free(output_data);

        // write the current frame's pipeline output to file
        google::protobuf::util::SerializeDelimitedToOstream(stream_data_msg, &output_proto_file);

        // compare it with the corresponding frame's output from the reference output data file
        if (!reference_data.empty()) {
            if (i < reference_data.size()) {
                google::protobuf::util::MessageDifferencer differ;
                std::string diff;
                differ.ReportDifferencesToString(&diff);
                if (!differ.Compare(stream_data_msg, reference_data[i])) {
                    std::cerr << "Pipeline output at frame " << i << " is different than the reference output:\n" << diff << std::endl;
                    std::cerr << "Terminating early due to difference in output at frame " << i << std::endl;
                    break;
                }
            } else {
                std::cerr << "warning: could not compare pipeline output for frame " << i << ":the reference output data file doesn't have data for frame " << i << std::endl;
            }
        }
    }
    output_proto_file.close();
    int pipeline_finalize_status = hands_pipeline_operator_finalize(pipeline_operator);
    if (pipeline_finalize_status != 0) {
        std::cerr << "error encountered during C API finalization of mediapipe pipeline wrapper: " << hands_pipeline_operator_get_last_error() << std::endl;
    }
    ABSL_LOG(INFO) << "done processing all input";
    return EXIT_SUCCESS;
}
