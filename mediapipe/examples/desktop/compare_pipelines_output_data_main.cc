// Compares two pipeline output protobuf files record by record,
// showing any difference via the protobuf google library's MessageDifferencer diff reporting utility.

#include <fstream>
#include <iostream>
#include <string>
#include "mediapipe/examples/desktop/pipeline_output.pb.h"
#include "google/protobuf/util/message_differencer.h"
#include "google/protobuf/io/zero_copy_stream_impl.h"
#include "google/protobuf/util/delimited_message_util.h"

using mediapipe::PipelineOutputData;
using google::protobuf::util::MessageDifferencer;

// Reads a stream of PipelineOutputData messages from a file.
bool ReadPipelineOutputDataStream(const std::string& filename, std::vector<PipelineOutputData>& out) {
    std::ifstream input(filename, std::ios::binary);
    if (!input.is_open()) {
        std::cerr << "Failed to open " << filename << std::endl;
        return false;
    }
    google::protobuf::io::IstreamInputStream zero_copy_input(&input);
    bool clean_eof = false;
    int msg_count = 0;
    while (true) {
        PipelineOutputData msg;
        std::streampos pos = input.tellg();
        if (!google::protobuf::util::ParseDelimitedFromZeroCopyStream(&msg, &zero_copy_input, &clean_eof)) {
            if (msg_count == 0) {
                std::cerr << "Failed to parse any messages from " << filename << " (parse error at file offset " << pos << ")" << std::endl;
                return false;
            } else {
                // EOF or trailing bytes after all messages parsed; treat as normal
                break;
            }
        }
        if (clean_eof) {
            // End of file reached after last message
            break;
        }
        out.push_back(msg);
        ++msg_count;
    }
    std::cerr << "Successfully parsed " << msg_count << " messages from " << filename << std::endl;
    return true;
}

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <output_data_cpp.pb> <output_data_python.pb>" << std::endl;
        return 1;
    }
    std::vector<PipelineOutputData> current, reference;
    if (!ReadPipelineOutputDataStream(argv[1], current)) return 2;
    if (!ReadPipelineOutputDataStream(argv[2], reference)) return 3;

    size_t n = std::min(current.size(), reference.size());
    bool all_equal = true;
    int different_pairs = 0;
    for (size_t i = 0; i < n; ++i) {
        if (!MessageDifferencer::Equals(current[i], reference[i])) {
            all_equal = false;
            ++different_pairs;
            std::cout << "Record " << i << " differs:" << std::endl;
            std::string diff;
            MessageDifferencer differ;
            differ.ReportDifferencesToString(&diff);
            differ.Compare(current[i], reference[i]);
            std::cout << diff << std::endl;
        }
    }
    if (current.size() != reference.size()) {
        std::cout << "Warning: record count differs: " << current.size() << " vs " << reference.size() << std::endl;
        all_equal = false;
    }
    if (all_equal) {
        std::cout << "All records equal." << std::endl;
        return 0;
    } else {
        std::cout << different_pairs << " records are different." << std::endl;
        return 4;
    }
}
