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

#ifndef MEDIAPIPE_CALCULATORS_TFLITE_SSD_ANCHORS_CALCULATOR_UTILS_H_
#define MEDIAPIPE_CALCULATORS_TFLITE_SSD_ANCHORS_CALCULATOR_UTILS_H_

#include <fstream>
#include <string>
#include <vector>

#include "mediapipe/calculators/tflite/ssd_anchors_calculator.pb.h"
#include "mediapipe/framework/formats/object_detection/anchor.pb.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/statusor.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/port/file_helpers.h"

namespace mediapipe {

// Helper functions for generating SSD anchors
class SsdAnchorsCalculatorUtils {
 public:
  // Generate anchors based on the options - moved from private to public
  static absl::Status GenerateAnchors(
      std::vector<Anchor>* anchors, const SsdAnchorsCalculatorOptions& options);

 private:
  // Helper method to load options from YAML config file
  static absl::Status LoadOptionsFromFile(
      const std::string& config_path,
      SsdAnchorsCalculatorOptions* options);

  // Generate multi-scale anchors based on the options
  static absl::Status GenerateMultiScaleAnchors(
      std::vector<Anchor>* anchors, const SsdAnchorsCalculatorOptions& options);
};

}  // namespace mediapipe

#endif  // MEDIAPIPE_CALCULATORS_TFLITE_SSD_ANCHORS_CALCULATOR_UTILS_H_
