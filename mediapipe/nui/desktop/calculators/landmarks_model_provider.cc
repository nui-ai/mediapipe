// Copyright 2024 The MediaPipe Authors.
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

#include <memory>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "mediapipe/calculators/util/resource_provider_calculator.pb.h"
#include "mediapipe/framework/api2/node.h"
#include "mediapipe/framework/api2/packet.h"
#include "mediapipe/framework/api2/port.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/status_macros.h"
#include "mediapipe/framework/resources.h"

#include "mediapipe/nui/desktop/calculators/landmarks_model_provider.h"

namespace mediapipe::api2 {

absl::Status LandmarksModelProvider::Open(CalculatorContext* cc) {
  // Always load the fixed model file, ignore side packets and options.
  constexpr absl::string_view kModelPath = "mediapipe/modules/hand_landmark/hand_landmark_full.tflite";
  Resources::Options res_opts = {};
  // Default to binary mode for tflite.
  res_opts.read_as_binary = true;

  MP_ASSIGN_OR_RETURN(std::unique_ptr<Resource> res,
                      cc->GetResources().Get(kModelPath, res_opts));
  Packet<Resource> res_packet = api2::PacketAdopting(std::move(res));
  kResources(cc)[0].Set(std::move(res_packet));
  return absl::OkStatus();
}

MEDIAPIPE_REGISTER_NODE(LandmarksModelProvider)

}
