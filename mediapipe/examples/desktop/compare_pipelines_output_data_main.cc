// Copyright 2025 The MediaPipe Authors.
//
// Compares two pipeline output protobuf files record by record.
#include <fstream>
#include <iostream>
#include <string>
#include "mediapipe/examples/desktop/pipeline_output.pb.h"
#include "google/protobuf/util/message_differencer.h"

using mediapipe::PipelineOutputData;
using google::protobuf::util::MessageDifferencer;

// Reads a stream of PipelineOutputData messages from a file.
bool ReadPipelineOutputDataStream(const std::string& filename, std::vector<PipelineOutputData>& out) {
    std::ifstream input(filename, std::ios::binary);
    if (!input.is_open()) {
        std::cerr << "Failed to open " << filename << std::endl;
        return false;
    }
    while (input.peek() != EOF) {
        PipelineOutputData msg;
        if (!msg.ParseFromIstream(&input)) {
            std::cerr << "Failed to parse message from " << filename << std::endl;
            return false;
        }
        out.push_back(msg);
    }
    return true;
}

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <output_data_cpp.pb> <output_data_python.pb>" << std::endl;
        return 1;
    }
    std::vector<PipelineOutputData> cpp_records, python_records;
    if (!ReadPipelineOutputDataStream(argv[1], cpp_records)) return 2;
    if (!ReadPipelineOutputDataStream(argv[2], python_records)) return 3;

    size_t n = std::min(cpp_records.size(), python_records.size());
    bool all_equal = true;
    for (size_t i = 0; i < n; ++i) {
        if (!MessageDifferencer::Equals(cpp_records[i], python_records[i])) {
            all_equal = false;
            std::cout << "Record " << i << " differs:" << std::endl;
            std::string diff;
            MessageDifferencer differ;
            differ.ReportDifferencesToString(&diff);
            differ.Compare(cpp_records[i], python_records[i]);
            std::cout << diff << std::endl;
        }
    }
    if (cpp_records.size() != python_records.size()) {
        std::cout << "Warning: record count differs: " << cpp_records.size() << " vs " << python_records.size() << std::endl;
        all_equal = false;
    }
    if (all_equal) {
        std::cout << "All records are equal." << std::endl;
        return 0;
    } else {
        std::cout << "Differences found." << std::endl;
        return 4;
    }
}

