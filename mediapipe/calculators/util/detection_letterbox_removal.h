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

#include <cmath>
#include <vector>

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/formats/location.h"
#include "mediapipe/framework/port/ret_check.h"

namespace mediapipe {
    namespace {

        // Processes detections from a letterboxed image to adjust their locations
        // to the corresponding non-letterboxed image.
        std::unique_ptr<std::vector<Detection>> UnLetterBox(
            const std::vector<Detection>& input_detections,
            const std::array<float, 4>& letterbox_padding) {

            const float left = letterbox_padding[0];
            const float top = letterbox_padding[1];
            const float left_and_right = letterbox_padding[0] + letterbox_padding[2];
            const float top_and_bottom = letterbox_padding[1] + letterbox_padding[3];

            auto output_detections = absl::make_unique<std::vector<Detection>>();
            for (const auto& detection : input_detections) {
                Detection new_detection;
                new_detection.CopyFrom(detection);
                LocationData::RelativeBoundingBox* relative_bbox = new_detection.mutable_location_data()->mutable_relative_bounding_box();

                relative_bbox->set_xmin(
                    (detection.location_data().relative_bounding_box().xmin() - left) / (1.0f - left_and_right));
                relative_bbox->set_ymin(
                    (detection.location_data().relative_bounding_box().ymin() - top) / (1.0f - top_and_bottom));
                // The size of the bounding box will change as well.
                relative_bbox->set_width(
                    detection.location_data().relative_bounding_box().width() / (1.0f - left_and_right));
                relative_bbox->set_height(
                    detection.location_data().relative_bounding_box().height() / (1.0f - top_and_bottom));

                // Adjust keypoints as well.
                for (int i = 0; i < new_detection.mutable_location_data()->relative_keypoints_size(); ++i) {
                    auto* keypoint = new_detection.mutable_location_data()->mutable_relative_keypoints(i);
                    const float new_x = (keypoint->x() - left) / (1.0f - left_and_right);
                    const float new_y = (keypoint->y() - top) / (1.0f - top_and_bottom);
                    keypoint->set_x(new_x);
                    keypoint->set_y(new_y);
                }

                output_detections->emplace_back(new_detection);
            }

            return output_detections;
        }

    }
}

